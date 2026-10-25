#!/usr/bin/env python3
"""Reconstruct your own capture (ordered image folder or video) with the full pipeline.

The capture is prepared into a sequence (working frames, full-resolution copies,
masks), then the same `make reconstruct` used for the benchmarks runs on it:
poses, VGGT/MoGe/DA3 depth, photometric fusion and refinement, cross-view check,
TSDF, cleanup, point clouds and texture. The camera model is rescaled so the
median scene depth is fixed (NORMALIZE_DEPTH), because a capture's COLMAP scale
is arbitrary and the pipeline's settings are in map units.

Object masks:
  neither option below   BiRefNet (recon.masks.birefnet) masks the salient object in
                         every frame, which suits a capture centred on one object;
  --masks DIR            one binary mask per source image (same file stem; folders only);
  --mask-command "CMD"   any external program, run once with {frames} replaced by the
                         prepared frames folder and {output} by the folder it must
                         fill with one binary NNNNNNNN.png mask per frame (works for
                         videos too);
  --scene                no masks: the whole frame is the "object" and the scene is
                         reconstructed.
Supplied masks define the object when the most salient thing in view is not it.
--complete-masks fills in parts they miss with BiRefNet foreground that overlaps
them; --refine-masks refines them with SAM-HQ (recon.masks.sam_hq), which only
accepts boundary changes that keep at least 85% overlap.
Outputs land in --output: mesh/ (object.ply, object.glb, scene.ply), points/,
poses/ and metrics/ as for `make reconstruct`.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from recon.common.progress import done, heading
from recon.inputs.capture.prepare import IMAGE_SUFFIXES, is_video, prepare, video_step


def install_masks(masks: Path, sequence: Path, destination: Path) -> int:
    """Copy user masks to destination, matched to frames through the source file names."""
    report = json.loads((sequence / 'sequence.json').read_text())
    files = {path.stem: path for path in masks.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES}
    width, height = report['size']
    installed = 0
    for item in report['mapping']:
        source = files.get(Path(item['source']).stem)
        if source is None:
            raise FileNotFoundError(f"No mask for {item['source']} in {masks}")
        mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(destination / f"{item['frame']:08d}.png"), np.where(mask > 127, 255, 0).astype(np.uint8))
        installed += 1
    return installed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', type=Path, help='Image folder or video file')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--masks', type=Path, help='Object masks named like the source images')
    parser.add_argument('--mask-command',
                        help='External mask generator; {frames} and {output} are substituted')
    parser.add_argument('--complete-masks', action='store_true',
                        help='Complete supplied masks with overlapping BiRefNet foreground')
    parser.add_argument('--refine-masks', action='store_true', help='Refine supplied masks with SAM-HQ')
    parser.add_argument('--mask-dilate', type=int, default=2,
                        help='Grow automatic masks by this many pixels (at a 960 px frame width): a mask '
                             'slightly too tight costs far more than one slightly too loose')
    parser.add_argument('--scene', action='store_true', help='No masks: reconstruct the whole scene')
    parser.add_argument('--max-edge', type=int, default=1200, help='Working-frame size')
    parser.add_argument('--sample-fps', type=float, default=6., help='Frames per second kept from a video')
    parser.add_argument('--normalize-depth', type=float, default=6.5)
    parser.add_argument('make_vars', nargs='*', help='Extra NAME=value settings passed to make')
    args = parser.parse_args()
    source = args.images.resolve()
    if not (source.is_dir() or is_video(source)):
        parser.error(f'No image directory or supported video: {source}')
    if args.masks is not None and is_video(source):
        parser.error('--masks needs an image folder (one mask per image)')
    supplying = args.masks is not None or bool(args.mask_command)
    if args.scene and supplying:
        parser.error('--scene reconstructs without masks; drop --masks/--mask-command')
    if (args.complete_masks or args.refine_masks) and not supplying:
        parser.error('--complete-masks and --refine-masks work on supplied masks')
    if args.complete_masks and args.refine_masks:
        parser.error('Choose one of --complete-masks and --refine-masks')
    output = args.output.resolve()
    sequence = output / 'prepared'
    report_path = sequence / 'sequence.json'
    if report_path.is_file():
        old = json.loads(report_path.read_text())
        if is_video(source):
            changed = old.get('video_step') != video_step(source, args.sample_fps)
        else:
            current = {p.name for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
            changed = current != {item['source'] for item in old['mapping']}
        if old['source'] != str(source) or changed or old['max_edge'] != args.max_edge:
            parser.error('Input frames or sampling changed; choose a fresh --output')
        heading('Frames', 'reusing prepared frames')
    else:
        heading('Frames', str(source))
        prepare(source, sequence, args.max_edge, args.sample_fps)
    root = Path(__file__).resolve().parents[2]
    supplied = sequence / 'masks_supplied'
    if args.masks is not None:
        supplied.mkdir(exist_ok=True)
        heading('Masks', f'{install_masks(args.masks.resolve(), sequence, supplied)} supplied')
    elif args.mask_command and not supplied.is_dir():
        supplied.mkdir()
        heading('Masks', 'external mask command')
        command = shlex.split(args.mask_command.format(frames=sequence / 'frames', output=supplied))
        subprocess.run(command, check=True)
    frames = len(list((sequence / 'frames').glob('*.png')))
    if supplied.is_dir():
        count = len(list(supplied.glob('*.png')))
        if count != frames:
            parser.error(f'{count} masks for {frames} frames in {supplied}')

    def birefnet(output: Path, *extra: str) -> Path:
        if len(list(output.glob('*.png'))) == frames and (output.parent / f'{output.name}_birefnet.json').is_file():
            return output                                             # reuse a finished run
        subprocess.run([sys.executable, '-m', 'recon.masks.birefnet', str(sequence / 'frames'),
                        '--output', str(output), '--dilate', str(args.mask_dilate), *extra],
                       check=True, cwd=root)
        return output

    object_mode = not args.scene
    if args.scene:
        heading('Masks', 'none: reconstructing the whole scene')
        masks = sequence / 'masks'                                    # all-white, from prepare
    elif not supplied.is_dir():
        heading('Masks', 'BiRefNet (salient object)')
        masks = birefnet(sequence / 'masks_birefnet')
    elif args.complete_masks:
        heading('Masks', 'BiRefNet completion')
        masks = birefnet(sequence / 'masks_completed', '--guide', str(supplied))
    elif args.refine_masks:
        heading('Masks', 'SAM-HQ refinement')
        masks = sequence / 'masks_refined'
        subprocess.run([sys.executable, '-m', 'recon.masks.sam_hq', str(sequence),
                        '--input-masks', str(supplied), '--output-masks', str(masks),
                        '--checkpoint', str(root / 'runs/models/sam_hq/sam_hq_vit_b.pth')],
                       check=True, cwd=root, stdout=subprocess.DEVNULL)
    else:
        masks = supplied
    # The Makefile reads full-resolution images from DATA/SCENE/image (and treats it as the
    # source of sequence.json); point it at the prepared full-resolution copies.
    image = sequence / 'image'
    if not image.exists():
        image.symlink_to('full', target_is_directory=True)
    os.utime(report_path)                                             # sequence.json stays newest
    make_vars = [f'SCENE={sequence.name}', f'SEQ_ROOT={output}', f'RUN={output}', f'DATA={output}',
                 f'MASKS={masks}',
                 f'NORMALIZE_DEPTH={args.normalize_depth}',
                 f'OBJECT_TSDF={"weighted" if object_mode else "scene"}', *args.make_vars]
    root = Path(__file__).resolve().parents[2]
    subprocess.run(['make', '--no-print-directory', '-C', str(root), *make_vars, 'reconstruct'],
                   check=True, env=os.environ.copy())
    mask_source = ('none (scene)' if args.scene else 'BiRefNet' if not supplied.is_dir() else
                   ('mask folder' if args.masks is not None else 'mask command')
                   + (' + BiRefNet completion' if args.complete_masks else '')
                   + (' + SAM-HQ' if args.refine_masks else ''))
    result = {'source': str(source), 'object_mode': object_mode, 'masks': mask_source, 'make_vars': make_vars,
              'meshes': sorted(str(p) for p in (output / 'mesh').glob('*') if p.suffix in ('.ply', '.glb'))}
    (output / 'construction.json').write_text(json.dumps(result, indent=2) + '\n')
    done(f"Construct: {'object' if object_mode else 'scene'} reconstruction in {output}")


if __name__ == '__main__':
    sys.exit(main())
