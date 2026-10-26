# Third-party notices

This file records code in `recon/` that was adapted from other projects, with
the licence notices those projects require. Libraries and models that are only
called (COLMAP, LightGlue, SuperPoint, ALIKED, MoGe, VGGT, SAM-HQ, BiRefNet, 2DGS, Open3D, PyTorch,
gsplat, libigl, xatlas, scikit-image, SciPy) keep
their own licences in `third_party/` or their packages; see the README's
"Licences of components" and "Credits and citations" sections.

`recon/depth/gaussian_refine.py` is not listed below: it reimplements PGSR's published method from the
paper and contains none of its code, as its docstring records. The methods the newer stages call are
credited where they are used - the generalised winding number (Jacobson et al., SIGGRAPH 2013; Barill et
al., SIGGRAPH 2018) through libigl in `recon/surface/watertight.py`, shape from silhouette (Laurentini,
TPAMI 1994) for the visual hull there, and xatlas for UV unwrapping in `recon/surface/texture.py`.

# Adapted code

## DTUeval-python

Source: https://github.com/jzhangbs/DTUeval-python (eval.py)
Used in: `recon/evaluation/metrics/dtu.py`: the DTU chamfer protocol and its constants

```text
MIT License

Copyright (c) 2021 Jingyang Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## NeuS

Source: https://github.com/Totoro97/NeuS (models/dataset.py, load_K_Rt_from_P)
Used in: `recon/inputs/dtu/prepare.py`: `decompose`, splitting a DTU projection matrix into intrinsics and pose

```text
MIT License

Copyright (c) 2021 Peng Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## IDR

Source: https://github.com/lioryariv/idr (load_K_Rt_from_P, which NeuS took from IDR)
Used in: `recon/inputs/dtu/prepare.py`: `decompose`

```text
MIT License

Copyright (c) 2020 Lior Yariv

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MobileBrick evaluation

Source: https://github.com/ActiveVisionLab/MobileBrick (evaluations/evaluate_3d.py)
Used in: `recon/evaluation/metrics/mobilebrick.py`: the official 3D protocol (ICP gate, visibility crop, sampling, thresholds, chamfer)

```text
MIT License

Copyright (c) 2023 Active Vision Laboratory

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Tanks and Temples toolbox

Source: https://github.com/isl-org/TanksAndTemples (python_toolbox/evaluation: run.py, registration.py, evaluation.py), in the variant bundled with 2DGS (scripts/eval_tnt)
Used in: `recon/evaluation/metrics/tnt.py`: trajectory alignment, ICP refinement schedule, cropping and precision/recall/F-score, per-scene thresholds

```text
The MIT License (MIT)

Copyright (c) 2017 Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, Vladlen Koltun

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

# Reimplemented behaviour (no code copied)

| Where | What it reproduces | Source |
|---|---|---|
| `recon/cameras/matching.py`: `sequential_pairs` | COLMAP's sequential pair selection with quadratic overlap | COLMAP, `src/colmap/controllers/pairing.cc` (BSD-3-Clause) |
| `recon/depth/multiview/patchmatch.py`: `read_array`, `write_array` | COLMAP's dense depth and normal map file format | COLMAP, `scripts/python/read_write_dense.py` (BSD-3-Clause) |
| `recon/depth/multiview/vggt.py`: `unpad_depth` | Inverse of VGGT's `load_and_preprocess_images(mode='pad')` | VGGT, `vggt/utils/load_fn.py` (VGGT License) |
| `recon/masks/birefnet.py`: `foreground` | Inference preprocessing (1024×1024 resize, ImageNet normalisation, sigmoid of the last output) | BiRefNet model card and `inference.py`, https://github.com/ZhengPeng7/BiRefNet (MIT, Copyright (c) 2024 ZhengPeng) |
| `recon/evaluation/metrics/pose_auc.py` | Relative pose errors and AUC (`calculate_auc_np`, `rotation_angle`, `translation_angle`) | Depth Anything 3, `src/depth_anything_3/bench/utils.py` (Apache-2.0, ByteDance), after VGGT's CO3D evaluation and PoseDiffusion |
| `recon/evaluation/baselines/twodgs.py`: `cull` | 2DGS's DTU mesh culling with dilated masks | 2DGS, `scripts/eval_dtu/evaluate_single_scene.py` (Gaussian-Splatting License) |
| `recon/evaluation/baselines/vggt_feedforward.py` | Inverse of VGGT's padded preprocessing for intrinsics; pose decoding via VGGT's own `pose_encoding_to_extri_intri` | VGGT, `vggt/utils/load_fn.py`, `vggt/utils/pose_enc.py` (VGGT License) |
| `recon/evaluation/metrics/mesh.py`: `pose_similarity` | Chordal L2 rotation averaging | Hartley et al., "Rotation Averaging", IJCV 2013 (method) |
| `recon/surface/tsdf.py` | VoxelBlockGrid integration sequence | Open3D, `examples/python/t_reconstruction_system/integrate.py` (MIT) |
