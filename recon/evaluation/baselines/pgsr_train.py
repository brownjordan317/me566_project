#!/usr/bin/env python3
"""Run PGSR's train.py unchanged, with GPU pixel sampling and an optional depth-prior loss.

Called by recon.evaluation.baselines.pgsr with the PGSR checkout as working directory
and train.py's own arguments, plus two of ours (removed before train.py sees them):

  --depth-prior DIR   <image name>.npy per training view: our depth in PGSR's frame
                      (normalised units, principal point centred, cropped), 0 = none;
  --depth-weight W    weight of the prior term;
  --depth-clip C      cap each pixel's relative difference at C (a robust prior: where our
                      depth is clearly wrong it stops pulling); 0 = no cap.
  --depth-decay N     fade the weight linearly to zero over the first N iterations (0 = constant).


GPU sampling: each iteration PGSR picks 102,400 pixels for its multi-view loss with
numpy.random.choice(n, k, replace=False), which shuffles all n candidates on the CPU
(7.4% of a profiled run, and a GPU stall). That call pattern (an integer population,
no replacement, no weights) returns the first k of a GPU random permutation instead,
the same distribution; PGSR only uses it to index a GPU tensor. Other calls pass through.

Depth prior (after DRGS, Chung et al., "Depth-Regularized Optimization for 3D Gaussian
Splatting in Few-Shot Images", CVPR workshops 2024; our implementation): PGSR's loop is
not edited. gaussian_renderer.render is wrapped to keep the first gradient-tracking
render after each backward pass, which in train.py is the iteration's training view, and
asked for its plane depth from the first iteration (PGSR itself requests it only once its
geometric losses start);
torch.Tensor.backward is wrapped so that, on train.py's scalar loss, W times the mean
relative difference between that render's plane depth and the prior (where the prior
exists) is added before the gradients are computed.
"""
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

_choice = np.random.choice


def choice(a, size=None, replace=True, p=None):
    if not replace and p is None and size is not None and np.ndim(a) == 0:
        return torch.randperm(int(a), device='cuda')[:int(size)]
    return _choice(a, size, replace, p)


def take(argv: list[str], name: str):
    """Remove `name VALUE` from argv and return VALUE (None if absent)."""
    if name not in argv:
        return None
    index = argv.index(name)
    value = argv[index + 1]
    del argv[index:index + 2]
    return value


def install_depth_prior(directory: Path, weight: float, clip: float = 0., decay: int = 0) -> None:
    import gaussian_renderer
    render, backward = gaussian_renderer.render, torch.Tensor.backward
    state, priors, step = {}, {}, [0]

    def prior(name: str):
        if name not in priors:
            path = directory / f'{name}.npy'
            priors[name] = torch.as_tensor(np.load(path), device='cuda') if path.is_file() else None
        return priors[name]

    def recording_render(camera, *args, **kwargs):
        training_view = torch.is_grad_enabled() and 'camera' not in state
        if training_view:
            kwargs['return_plane'] = True       # depth from the first iteration; PGSR's own losses start later
        package = render(camera, *args, **kwargs)
        if training_view:
            state['camera'], state['package'] = camera, package
        return package

    def backward_with_prior(tensor, *args, **kwargs):
        if state and tensor.dim() == 0 and tensor.requires_grad:
            target = prior(state['camera'].image_name)
            depth = state['package']['plane_depth'].squeeze()
            if target is not None and target.shape == depth.shape:
                valid = target > 0
                if valid.any():
                    relative = (depth[valid] - target[valid]).abs() / target[valid]
                    if clip > 0:
                        relative = relative.clamp(max=clip)
                    scale = max(0., 1 - step[0] / decay) if decay > 0 else 1.
                    if scale > 0:
                        tensor = tensor + weight * scale * relative.mean()
            step[0] += 1
        state.clear()
        return backward(tensor, *args, **kwargs)

    gaussian_renderer.render = recording_render
    torch.Tensor.backward = backward_with_prior


if __name__ == '__main__':
    np.random.choice = choice
    sys.path.insert(0, os.getcwd())                 # the PGSR checkout
    argv = sys.argv[1:]
    directory, weight, clip = take(argv, '--depth-prior'), take(argv, '--depth-weight'), take(argv, '--depth-clip')
    decay = take(argv, '--depth-decay')
    if directory is not None:
        install_depth_prior(Path(directory), float(weight or 1.0), float(clip or 0.), int(decay or 0))
    sys.argv = ['train.py', *argv]
    runpy.run_path('train.py', run_name='__main__')
