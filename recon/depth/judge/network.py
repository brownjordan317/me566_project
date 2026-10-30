"""The learned judge: scores any number of depth candidates per pixel, in any order.

Each candidate's evidence (its own, the image's, and its relation to the others: recon.depth.judge
.evidence) goes through one shared convolutional encoder, so the judge sees each candidate's
neighbourhood and treats every candidate alike. At each pixel the candidates then attend to one another
(a set transformer without positional encoding: Lee et al., "Set Transformer", ICML 2019), with one
learned "none of these" token alongside, and each candidate and the none token get a score. Nothing
depends on how many candidates there are or which is which, so a new source joins without retraining
the architecture. A confidence head on the none token's output predicts whether the judged depth is
within the training tolerance.

With photo_features > 0, a small strided network reads the full-resolution photo (PHOTO_SCALE times the
evidence grid) down to the grid, so detail finer than a grid cell (edges, studs, texture) reaches the
judge; its output joins the image evidence every candidate sees.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def block(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.GroupNorm(8, outputs), nn.GELU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.GroupNorm(8, outputs), nn.GELU())


PHOTO_SCALE = 8                       # full-resolution photo pixels per evidence-grid cell (stride 4, half-size working frame)


class PhotoStem(nn.Module):
    """Full-resolution RGB (B, 3, 8h, 8w) to (B, outputs, h, w) through three stride-2 stages."""

    def __init__(self, outputs: int = 32):
        super().__init__()
        stage = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.GroupNorm(8, o), nn.GELU(),
                                           nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU())
        self.stages = nn.Sequential(stage(3, 16), stage(16, 32), stage(32, outputs))

    def forward(self, photo: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        x = self.stages(photo)
        return x if x.shape[-2:] == size else F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class Encoder(nn.Module):
    """A three-level U-Net from per-pixel evidence to a per-pixel embedding."""

    def __init__(self, inputs: int, width: int = 64):
        super().__init__()
        self.down1, self.down2, self.down3 = block(inputs, 32), block(32, 64), block(64, 96)
        self.up2, self.up1 = block(96 + 64, 64), block(64 + 32, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.down1(x)
        b = self.down2(F.max_pool2d(a, 2, ceil_mode=True))
        c = self.down3(F.max_pool2d(b, 2, ceil_mode=True))
        b = self.up2(torch.cat([F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b], 1))
        return self.up1(torch.cat([F.interpolate(b, size=a.shape[-2:], mode='bilinear', align_corners=False), a], 1))


class JudgeNet(nn.Module):
    def __init__(self, candidate_features: int, image_features: int, width: int = 64, heads: int = 4,
                 layers: int = 2, photo_features: int = 0):
        super().__init__()
        self.photo = PhotoStem(photo_features) if photo_features else None
        self.encoder = Encoder(candidate_features + image_features + photo_features, width)
        self.none = nn.Parameter(torch.zeros(width))
        layer = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0., batch_first=True, norm_first=True)
        self.attention = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.score = nn.Linear(width, 1)
        self.confidence = nn.Linear(width, 1)

    def forward(self, candidates: torch.Tensor, image: torch.Tensor, present: torch.Tensor,
                photo: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """candidates (B, C, F, h, w), image (B, I, h, w), present (B, C, h, w) bool, and with photo_features
        the full-resolution photo (B, 3, PHOTO_SCALE h, PHOTO_SCALE w), RGB in 0-1.

        Returns logits (B, C + 1, h, w), the last channel "none of these" (absent candidates get -inf),
        and a confidence logit (B, h, w)."""
        B, C, _, h, w = candidates.shape
        if self.photo is not None:
            image = torch.cat([image, self.photo(photo, (h, w))], 1)
        inputs = torch.cat([candidates, image[:, None].expand(-1, C, -1, -1, -1)], 2).flatten(0, 1)
        embedding = self.encoder(inputs).reshape(B, C, -1, h, w)                  # (B, C, E, h, w)
        tokens = embedding.permute(0, 3, 4, 1, 2).reshape(B * h * w, C, -1)
        tokens = torch.cat([tokens, self.none.expand(B * h * w, 1, -1)], 1)
        absent = torch.cat([~present.permute(0, 2, 3, 1).reshape(B * h * w, C),
                            torch.zeros(B * h * w, 1, dtype=torch.bool, device=present.device)], 1)
        attended = self.attention(tokens, src_key_padding_mask=absent)
        logits = self.score(attended)[..., 0].masked_fill(absent, float('-inf'))
        confidence = self.confidence(attended[:, -1])[..., 0]
        return (logits.reshape(B, h, w, C + 1).permute(0, 3, 1, 2), confidence.reshape(B, h, w))
