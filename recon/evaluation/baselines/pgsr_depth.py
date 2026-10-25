#!/usr/bin/env python3
"""Save a trained PGSR model's depth for every training view (run inside PGSR's environment).

Called by recon.evaluation.baselines.pgsr with the PGSR (or Fast-PGSR) checkout as working directory;
it uses PGSR's own scene loading and renderer unmodified (the same calls as its
render.py: Scene, GaussianModel, render(...)['plane_depth'], the z-depth its TSDF
fuses) and writes <image name>.npy in the model's normalised units at the training
image size. Mapping back to our frames happens in the caller. No meshing here.
"""
import inspect
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, os.getcwd())                     # the PGSR checkout

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene import Scene                               # noqa: E402


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    optimisation = OptimizationParams(parser)
    parser.add_argument('--iteration', default=-1, type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = get_combined_args(parser)
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device='cuda')
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for view in scene.getTrainCameras():
            # Fast-PGSR's render also takes the training options (its tile multiplier); PGSR's does not.
            extra = (optimisation.extract(args),) if 'args' in inspect.signature(render).parameters else ()
            depth = render(view, gaussians, pipeline.extract(args), background, *extra)['plane_depth'].squeeze()
            np.save(args.output / f'{view.image_name}.npy', depth.float().cpu().numpy())


if __name__ == '__main__':
    main()
