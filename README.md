# Calibrated object reconstruction

This project reconstructs one rigid object from a calibrated image sequence. The supported offline path matches each frame to its sequential neighbours with full-image SuperPoint + LightGlue and solves COLMAP poses with the dataset's fixed per-frame intrinsics, then uses MoGe-3 for per-image depth, multiview refinement inside the object, full-scene TSDF fusion, and mask votes to extract the object. The calibrated benchmark path also applies a cross-view depth check; its measured benefits vary by scene. It is an offline reconstruction pipeline, not a live SLAM tracker.

The MobileBrick benchmark supplies per-frame intrinsics, RGB images, and object masks. Reference poses and meshes are used only by `make evaluate`. The model's scale comes from monocular reconstruction; benchmark millimeter values use similarity alignment to the reference poses and do not prove independent absolute scale.

## Run

Create a Python environment with CUDA-enabled PyTorch if you need MoGe-3 inference and LightGlue on the GPU. COLMAP is a separate executable. From the project root:

```bash
make setup                         # one-time downloads and Python dependencies
make download                      # one-time MobileBrick download
make prepare                       # half-resolution RGB, masks, and calibration
make reconstruct TSDF_DEVICE=cuda  # offline once models and data are cached
make evaluate                      # compare with the reference scan
make benchmark                     # reconstruct and evaluate all 18 test scenes
make test
```

`TSDF_DEVICE` defaults to `cuda`; use `TSDF_DEVICE=cpu` with a CPU-only Open3D build. The Makefile detects the local `/tmp/colmap-local/usr/bin/colmap` installation if present; otherwise set `COLMAP=/path/to/colmap`. For a different MobileBrick scene, set `SCENE=scene_name`. To use an explicit MoGe checkpoint, set `MOGE_CHECKPOINT=/path/to/model.pt`. `make status` prints the selected masks and mesh path. Completed stages are reused. When you change the mask directory, run `make rebuild-object MASKS=/path/to/masks` so the hybrid depths and meshes are regenerated. Set `RUN=/path/to/dir` to write a scene's outputs somewhere other than beside its prepared sequence.

`make benchmark` reconstructs and scores every downloaded scene with the default pipeline (VGGT and DA3 object depth chosen per region by photometric evidence, MoGe depth for the background) and its supplied masks, so scenes are compared on equal terms. Each scene's outputs and log go under `runs/real/mobilebrick_sweep/` (set `BENCH_ROOT` to change this), a failed scene does not stop the sweep, and `summary_object.md` (and `.csv`; `summary_object_dtu.md` for DTU) there tabulate pose error, surface error, coverage, and open edges per scene. Set `BENCH_SCENES="jeep audi"` to run a subset. Two scenes run at once (`BENCH_JOBS=2`): GPU stages take turns through a lock file while the other scene's COLMAP mapping runs on the CPU, which cut four scenes from about 400 s to 265 s. DTU is prepared once with `make prepare-dtu` (sources under `DTU_SOURCE`, default `/media/megrad/Data_Storage1/datasets/dtu`). For DTU, add `DATA=runs/real/dtu_dataset SEQ_ROOT=runs/real/dtu_benchmark PER_FRAME_FOCAL=`; the tables then report chamfer, accuracy, and completeness. DTU's robot-arm camera has a fixed lens, so it is run with one shared focal length: per-frame focal invents drift there (scan24 chamfer 0.64 mm shared vs 1.37 mm per-frame), while it helps MobileBrick's phone captures.

Every table also reports **shape** metrics. The official scores place the mesh using a similarity fitted to our camera centres. From RGB alone, absolute scale cannot be known, and a 1% focal or scale offset otherwise reads as millimetres of surface error. The shape metrics refine that placement, scale included, by fitting the surface itself (ICP), then score again: *Shape median/p90* for MobileBrick and *Shape chamfer* for DTU, which uses the official protocol after refinement. *Size error* is the scale that refinement removed; it is positive when our object comes out too large. Use the official columns to compare with published results, and the shape columns to judge geometry. On the current sweeps, MobileBrick's median surface error is 1.87 mm against a shape median of 0.72 mm, and DTU's median chamfer is 0.68 mm against a shape chamfer of 0.57 mm.

MobileBrick accuracy is measured from points sampled on our mesh to the reference **surface** (its triangles). Before 2026-10-10 it was measured to the nearest reference *vertex*. The reference meshes are remeshed at about 1 mm edges, so that scored even the reference itself at 0.38 mm median and 0.56 mm p90, and hid gains below that floor. All MobileBrick numbers in this README use the corrected metric. Coverage (reference vertices within 10 mm) and the DTU metric are unaffected.

On MobileBrick the estimated focal is about 1.5% longer than the dataset's phone-reported value, yet the shape after scaled alignment is the same with either focal (jeep, audi, convertible and big_ben agree within 0.04 mm). The offset only changes overall size. Per-frame focal estimates follow the phone's real focus breathing, about a 1.6% range within a capture (median correlation 0.82).

### Progress output

Every stage prints a coloured `▶ <stage>` label, then tqdm bars for per-frame and per-pair loops and halo spinners for long steps without measurable progress (COLMAP verification and refinement, model loading); COLMAP mapping shows registered images as a bar. Each stage ends with one green ✓ summary line; full details stay in its JSON report. `make benchmark` shows a live rich table of every scene's stage and elapsed time, then the summary table. `VERBOSE=1` also prints the underlying commands. When output goes to a file, or with `NO_PROGRESS=1`, bars and spinners become one plain line when a step starts and one when it ends, which is what the per-scene benchmark logs contain.

### Output layout

Single-scene runs write to `runs/real/mobilebrick_runs/<scene>/` (DTU: `runs/real/dtu_runs/<scan>/`; `RUN=dir` overrides), separate from the prepared inputs. After the last target on the command line finishes, intermediates are deleted and each scene keeps only:

```text
<run>/
  mesh/       finished meshes for Blender: object.ply (the object) and scene.ply (full scene
              with background); comparison arms add moge*, vggt_full*, dense*
  points/     coloured point clouds with normals: object.ply and scene.ply
  metrics/    poses.json and object.json against MobileBrick references, object_dtu.json against DTU
  poses/      sparse_txt/ (cameras, poses, sparse points), export/trajectory.csv, export_object/landmarks.ply
```

Object meshes are about 1-5 MB. Full-scene meshes are 10-60 MB each. They keep only surface that at least `SCENE_MIN_VIEWS=3` frames observed; surface seen by one or two frames was mostly floaters and ray-aligned shards at the capture's edge. They also drop pieces smaller than 5% of the main surface and fill tears up to four TSDF voxels across, without smoothing. Raise `SCENE_MIN_VIEWS` (for example to 5) for a cleaner but smaller scene; on jeep the object mesh was identical at 1, 3, and 5. Set `SCENE_MESHES=` (empty) to skip the full-scene meshes.

Point clouds come from the consistency-filtered depth. A pixel is kept only if at least three other views see a surface at the same depth (within 1%). Kept pixels are averaged into voxels: 1/500 of the object's diagonal for `object.ply` and four times that for `scene.ply`, so each cloud is roughly 0.5-1M points (about 10-25 MB) at any map scale. The object cloud uses the same mask vote as the object mesh. It is slightly noisier but more complete. On jeep it scored a 0.90 mm median against the 0.77 mm mesh, but covered 81% of the reference within 2 mm against 70%. On DTU scan24 its chamfer was 0.81 mm against the mesh's 0.66 mm, with better completeness (0.59 against 0.73 mm). Set `POINT_CLOUDS=` (empty) to skip them.

All final PLYs use 32-bit float positions and normals plus 8-bit RGB vertex colours. Open3D's default doubles made files twice as large. In Blender, colours import as a colour attribute: set the viewport shading colour to *Attribute*, or feed a *Color Attribute* node into Base Color. Pruned stages count as done, so rerunning a finished scene rebuilds nothing. Set `KEEP_INTERMEDIATE=1` to keep `depth/` (per-frame depth maps for each stage), `work/` (full-scene and uncleaned object meshes), and the COLMAP database. Run several targets in one command (`make reconstruct reconstruct-vggt`) so shared stages are computed once before pruning. `make construct` uses the same `poses/`, `depth/`, `work/`, `mesh/` names but keeps its intermediates.

## Fusing depth models (default; `FUSION=` turns it off)

`make reconstruct` adds DA3-Giant depth (input resolution 756, `DA3_RES`), conditioned on our COLMAP cameras, to VGGT's. It then picks between the two region by region (`recon/depth/fusion/filter.py`). Each candidate's depth is scored by how well 9×9 full-resolution image patches match across six neighbouring views (normalized cross-correlation, best three views averaged). Scores are pooled over 17×17 working pixels, and every object pixel takes the depth of the winner. A single patch is too weak to decide; pooled over a region, the evidence holds. DA3 and the fusion add about 10–20 s per scene.

Why it works: on the development scenes (jeep, audi, convertible, big_ben), giving the pipeline perfect depth halves the shape error, while perfect poses change nothing. That holds on all 18 MobileBrick scenes (`make stage-ranking`). VGGT and DA3 are about equally accurate but often wrong in different places. Choosing the better model per pixel with hindsight would close most of the gap; the photometric filter recovers a real part of it without hindsight:

| Scene | Shape median / p90, VGGT only (`FUSION=`) | With fusion |
|---|---|---|
| jeep | 0.49 / 1.62 mm | 0.39 / 1.32 mm |
| audi | 0.65 / 1.73 | 0.57 / 1.54 |
| convertible | 0.98 / 3.29 | 0.86 / 3.10 |
| big_ben | 1.23 / 3.93 | 0.95 / 3.04 |

On the full benchmarks (`runs/real/*_sweep`, with the earlier runs kept in `*_sweep_nofusion`), fusion improved 17 of 18 MobileBrick scenes. The median shape error went from 0.862 to 0.772 mm and p90 from 3.26 to 3.05 mm; the nine scenes not used in development improved like the four that were. On DTU, which played no part in its development, 11 of 15 scans improved. The mean chamfer went from 0.965 to 0.916 mm and the median from 0.762 to 0.699 mm. scan63 and scan83, already the hardest, got worse (1.37 to 1.49 mm and 2.05 to 2.17 mm).

After the choice, a photometric plane sweep (`recon/depth/fusion/refine.py`, on by default, `REFINE=` skips it) fine-tunes each object pixel within ±0.6% of its depth in 13 steps. Each step is scored like the choice, the scores are averaged over 15×15 working pixels, and the best step is interpolated between neighbouring steps. A pixel only moves if the images prefer the new depth by a margin. The neighbouring views are taken 10–35° away rather than closest: big_ben's 127 frames are only about 3° apart, and at that angle a millimetre of depth moves a patch by a fraction of a pixel. On the development scenes it lowered the shape median by 6–16% (big_ben 0.94 to 0.85 mm, audi 0.57 to 0.48 mm). The p90 improved on three scenes and worsened on convertible (3.11 to 3.36 mm), a glossy car whose reflections mislead patch matching. On the full benchmarks (fusion alone kept in `*_sweep_norefine`) it lowered the MobileBrick median shape error from 0.772 to 0.723 mm (17 of 18 scenes) and p90 from 3.05 to 2.80 mm. The p90 worsened on 5 scenes: aston, boat, camera, london_bus and satellite. On DTU it lowered the mean chamfer from 0.916 to 0.884 mm, the median from 0.699 to 0.679 mm and the mean shape chamfer from 0.658 to 0.601 mm, improving 11 of 15 scans; the other four are within 0.01 mm. Tilting patches to the surface normal (`--slanted`, on) made no measurable difference.

The object surface then comes from a confidence-weighted object TSDF (`recon/surface/weighted_tsdf.py`; `OBJECT_TSDF=scene` restores extraction from the scene TSDF). Each depth pixel counts by how well VGGT and DA3 agree there and by its viewing angle. Surface still needs three observing frames, so the weights change the average but never delete surface, and the band narrows to 0.06 map units. DA3 now runs at 756 px (`DA3_RES`).

On the full benchmarks (previous default in `*_sweep_v2`), MobileBrick improved on 16 of 18 scenes:
- F-score at 1 mm: 50.4% to 53.5%
- F-score at 2 mm: 72.4% to 74.6%
- recall at 1 mm: 42.8% to 46.7%

DTU's shape chamfer improved on all 15 scans (mean 0.601 to 0.515 mm). The official numbers stayed flat (DTU mean chamfer 0.884 to 0.889 mm), because they place the mesh using our estimated cameras, and that placement error now outweighs the remaining shape error. A tighter cross-view check (1% instead of 3%, `CONSISTENCY_TOLERANCE`) and a robust second averaging pass in the object volume then raised the MobileBrick F-score at 1 mm to 54.1%, the shape median to 0.671 mm, and the DTU shape chamfer to 0.505 mm (`*_sweep_v3` holds the runs before this change). MobileBrick tables report precision, recall and F-score at 1 and 2 mm after shape alignment. Accuracy-only scores reward deleting difficult surface; the F-score does not.

Single-image models (Depth Pro, Depth Anything V2, Marigold) were tested as extra candidates and removed. Their papers claim sharper edges and finer detail, but measured per region (outline, depth jumps, steep, textured, flat) none beat VGGT anywhere, and fine-scale (high-pass) detail is within 0.1–0.3 mm for every model. The filter picks them for under 1% of pixels, so they are left out of the default fusion. Smooth per-frame correction fields, even fitted to the true depth, recover only about 40% of the depth gap, so the remaining error is not smooth distortion.

## Reconstruct your own capture

For a folder of JPG or PNG images, or a video:

```bash
make construct /path/to/images        # or: make construct video.mp4, or SCAN=path
```

The capture is prepared under `runs/scans/<name>/prepared/`:
- **Frames:** an image folder is sorted numerically. A video is sampled at `SCAN_FPS` (default 6), keeping the sharpest frame of each window.
- **Working frames** are resized to at most `SCAN_MAX_EDGE` (1200).
- **Full-resolution copies** go to `full/`, for feature matching, photometric fusion and texture.

The same `make reconstruct` used for the benchmarks then runs on it: cameras, depth, fusion and refinement, cross-view check, TSDF, cleanup, point clouds and texture. Outputs land in `runs/scans/<name>/` (`SCAN_OUTPUT` changes this): `mesh/object.ply`, `mesh/object.glb` (textured), `mesh/scene.ply` and `points/`.

A capture's COLMAP scale is arbitrary, while the pipeline's settings are in map units. So the camera model is rescaled until the median scene depth is 6.5 units (`NORMALIZE_DEPTH`), about where MobileBrick and DTU sit. The mesh still has **arbitrary scale**.

Object masks:
- **By default**, [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) masks the most prominent object in every frame (`prepared/masks_birefnet/`). That suits a capture centred on one object; see [Automatic masks](#automatic-masks-birefnet) for how it compares with hand-made masks.
- **`SCAN_MASKS=/path/to/masks`:** your own masks, one binary mask per source image with the same file stem (image folders). Supply masks when the most prominent thing in view is not the object you want, or is only part of it.
- **`SCAN_MASK_COMMAND='program {frames} {output}'`:** any external mask generator (a video object segmentation tool, for example). It runs once, with `{frames}` replaced by the prepared frames folder and `{output}` by the folder it must fill with one binary `NNNNNNNN.png` per frame. This also works for videos.
- **`SCAN_COMPLETE_MASKS=1`:** fills in parts your masks miss (a tracker dropping wheels, say) with BiRefNet foreground that overlaps them. Other prominent things in view are left out.
- **`SCAN_REFINE_MASKS=1`:** refines your masks' edges with SAM-HQ (`make setup-masks` first). It only accepts boundary changes that keep at least 85% overlap.
- **`SCAN_SCENE=1`:** no masks. The whole frame is treated as the object, and the scene is reconstructed.

Run `make setup` once first; construction uses cached model weights and a CUDA GPU.

## Inputs and capture

Reconstruction needs only RGB frames and object masks. Camera intrinsics are estimated from the images (one shared focal length; `INTRINSICS=dataset` uses a dataset's own cameras for comparison). Poses are solved on the full-resolution originals with up to 2048 keypoints from each of two scales (`FEATURE_SCALES=1,0.5`, `MAX_KEYPOINTS=2048`; 4096 per scale doubled mapping time with no accuracy gain): fine keypoints add precision where detail is distinctive, while coarse ones stay matchable on repetitive texture such as patterned rugs.

**Focal length is estimated per frame.** Phone lenses refocus during capture, changing the focal length by about 1%, which a single shared focal cannot follow; on MobileBrick's iPhone captures a shared focal triples camera-pose error. After solving with one shared focal, each frame's focal is refined against its own observations. On fixed-focus cameras this invents about 1% of drift and costs some pose accuracy (DTU scan24: 0.98 to 1.53 mm); set `PER_FRAME_FOCAL=` to keep one shared focal for such cameras. Dataset intrinsics are never used unless you pass `INTRINSICS=dataset` for a comparison.

## Masks

`make reconstruct` uses existing SAM-HQ refined masks at `sam_hq/masks/` when available; otherwise it uses the MobileBrick supplied masks at `masks/`. You can select any prepared binary mask directory with `MASKS=/path/to/masks`. All masks must match the frame size and use eight-digit PNG names.

To refine masks produced by a separate tracker, run `make setup-masks` once, then:

```bash
make refine-masks TRACKED_MASKS=/path/to/tracked/masks
make rebuild-object MASKS=runs/real/mobilebrick_benchmark/london_bus/sam_hq/masks
```

The [SAM-HQ](https://github.com/SysCV/sam-hq) adapter uses tracker masks as prompts and limits changes to their boundaries. It never uses reference masks as prompts. For isolated geometry work, the dataset's supplied masks let you skip tracking entirely. Source checkouts under `third_party/` are kept unmodified.

### Automatic masks (BiRefNet)

`python3 -m recon.masks.birefnet FRAMES --output DIR` writes one mask per frame from BiRefNet, a high-resolution foreground segmenter (MIT). Add `--guide MASKS` to complete existing masks instead: only foreground pieces that overlap the supplied mask of the same frame are kept. The model code is fetched from Hugging Face and pinned to a reviewed revision; `make setup` downloads it.

Against the datasets' own masks, BiRefNet's masks averaged IoU 0.93 over the 18 MobileBrick scenes (0.86–0.97) and 0.99 on 11 of the 15 DTU scans. Reconstructions from them, at default settings:

| | MobileBrick F@1 / F@2, median of 18 | DTU chamfer, mean of 14 (all but scan37) |
|---|---|---|
| Dataset masks | 54.2 / 75.2 | 0.863 mm |
| BiRefNet masks | 54.9 / 77.2 | 0.868 mm |

BiRefNet beat the dataset masks on 12 MobileBrick scenes (convertible +4.5 F@1, audi +4.1, jeep +3.3). It loses where "most prominent object" is the wrong definition: on DTU scan37 it masks the tools but not the bricks under them, which DTU scores as part of the object (chamfer 4.05 against 1.22 mm), and on big_ben it misses parts of the tower in close-ups (F@1 27.6 against 38.5).

**Stray pieces.** A mask that covers something else for a few frames (BiRefNet took a chair for the object in four big_ben frames) passes the per-point mask vote, because those few views agree, and leaves a separate piece in the mesh. Object extraction therefore drops pieces masked in fewer than a quarter as many views as the main piece (`OBJECT_SUPPORT`, default .25; 0 turns it off). On big_ben the chair was masked in 8 views and the tower in 53. With the dataset masks the filter changed no scene by more than 0.2 F@1 on MobileBrick or 0.01 mm chamfer on DTU.

## Measured London bus result

The best measured mesh is the SAM-HQ mask variant at the path above. With 57 registered images, SuperPoint + LightGlue camera centers had **1.665 mm median** error after similarity alignment to the reference poses. The extracted object mesh had **1.439 mm median / 5.004 mm 90th-percentile** sampled surface distance, **97.73%** reference vertex coverage within 10 mm, and **3,090** boundary edges. The supplied-mask extraction from the same scene mesh had **1.521 / 5.834 mm**, **97.81%** coverage, and **2,344** boundary edges. These are different tradeoffs between surface accuracy and open edges; the measured best surface accuracy uses SAM-HQ.

The full-scene integration maps each MoGe depth pixel through the frame's calibrated intrinsics and optimized COLMAP pose. It rejects large relative depth jumps before TSDF fusion. Object extraction projects scene faces into multiple views, checks depth visibility, and retains faces supported by masks. Mask voting can itself introduce open boundaries. There is no claimed online loop closure or metric-scale guarantee.

## Code layout

The code is one package per pipeline component, in pipeline order. Each package's `__init__.py` describes it, and `tests/` mirrors the layout.

| Path | Component |
|---|---|
| `recon/inputs/` | 1. Inputs: `mobilebrick/` (download, prepare), `dtu/` (prepare), `capture/` (your own image folder or video) |
| `recon/masks/` | 2. Object segmentation: `birefnet` (automatic masks, or completion of supplied ones), `sam_hq` (optional boundary refinement of supplied masks) |
| `recon/cameras/` | 3. Calibration and poses: `matching` (SuperPoint + LightGlue), `sfm` (COLMAP), `solve` (stage entry point), `export`, `reference` (reference-pose substitution) |
| `recon/depth/` | 4. Depth: `grounding` (fit to sparse points), `monocular/` (MoGe-3), `multiview/` (VGGT, DA3, VGGT-Ω adapter), `fusion/` (photometric selection between models and plane-sweep refinement), `combine`, `consistency` |
| `recon/surface/` | 5. Fusion and extraction: `tsdf`, `weighted_tsdf`, `object` (mask votes), `clean`, `components`, `points`, `texture` |
| `recon/evaluation/` | 6. Measurement: `metrics/` (mesh, MobileBrick official, DTU, Tanks and Temples, poses, pose AUC, reprojection), `diagnostics/` (stage ranking with reference depth and poses), `benchmark/` (runner, summaries, comparison, timing, report), `baselines/` (VGGT feed-forward, 2DGS; comparisons only) |
| `recon/apps/` | End-to-end entry points (`construct`: your own capture; `view_ply`: viewer) |
| `recon/common/` | Shared helpers: cameras, COLMAP text models, PLY output, progress display, trajectories |
| `Makefile` | Supported setup, reconstruction, evaluation, and test commands |
| `calibration/` | Personal camera calibration, separate from MobileBrick intrinsics |
| `third_party/` | Source checkouts for academic baselines only (e.g. 2DGS); ignored by Git. The pipeline's models are pinned pip packages in `requirements.txt` |
| `runs/` | Downloaded data, checkpoints, and generated outputs; ignored by Git |

## Credits and citations

Code adapted from other projects is credited where it is used and listed, with the required licence notices, in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The methods, models and datasets this project builds on:

- **Structure from motion:** J. L. Schönberger and J.-M. Frahm, "Structure-from-Motion Revisited", CVPR 2016 (COLMAP). Stereo-fusion support rule: J. L. Schönberger et al., "Pixelwise View Selection for Unstructured Multi-View Stereo", ECCV 2016.
- **Features and matching:** D. DeTone, T. Malisiewicz and A. Rabinovich, "SuperPoint: Self-Supervised Interest Point Detection and Description", CVPR Workshops 2018. P. Lindenberger, P.-E. Sarlin and M. Pollefeys, "LightGlue: Local Feature Matching at Light Speed", ICCV 2023.
- **Depth:** R. Wang et al., "MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision", CVPR 2025 (MoGe-3 weights from https://github.com/microsoft/MoGe). J. Wang et al., "VGGT: Visual Geometry Grounded Transformer", CVPR 2025. Scale-and-shift alignment: R. Ranftl et al., "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer", TPAMI 2022.
- **Multi-view checks:** P. Merrell et al., "Real-Time Visibility-Based Fusion of Depth Maps", ICCV 2007. R. T. Collins, "A Space-Sweep Approach to True Multi-Image Matching", CVPR 1996. D. Scharstein and R. Szeliski, "A Taxonomy and Evaluation of Dense Two-Frame Stereo Correspondence Algorithms", IJCV 2002 (cost aggregation). M. Bleyer, C. Rhemann and C. Rother, "PatchMatch Stereo - Stereo Matching with Slanted Support Windows", BMVC 2011.
- **Fusion and meshing:** B. Curless and M. Levoy, "A Volumetric Method for Building Complex Models from Range Images", SIGGRAPH 1996. W. Dong et al., "ASH: A Modern Framework for Parallel Spatial Hashing in 3D Perception", TPAMI 2023. Q.-Y. Zhou, J. Park and V. Koltun, "Open3D: A Modern Library for 3D Data Processing", arXiv 2018. G. Taubin, "A Signal Processing Approach to Fair Surface Design", SIGGRAPH 1995.
- **Fused depth models:** H. Lin et al., "Depth Anything 3: Recovering the Visual Space from Any Views", 2025. A. Bochkovskii et al., "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second", ICLR 2025. L. Yang et al., "Depth Anything V2", NeurIPS 2024. B. Ke et al., "Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation", CVPR 2024 (Marigold). Spline depth correction: J. Kopf, X. Rong and J.-B. Huang, "Robust Consistent Video Depth Estimation", CVPR 2021.
- **Segmentation:** P. Zheng et al., "Bilateral Reference for High-Resolution Dichotomous Image Segmentation", CAAI Artificial Intelligence Research 2024 (BiRefNet). L. Ke et al., "Segment Anything in High Quality", NeurIPS 2023.
- **Alignment for scoring:** S. Umeyama, "Least-Squares Estimation of Transformation Parameters Between Two Point Patterns", TPAMI 1991. P. J. Besl and N. D. McKay, "A Method for Registration of 3-D Shapes", TPAMI 1992.
- **Datasets and protocols:** K. Li et al., "MobileBrick: Building LEGO for 3D Reconstruction on Mobile Devices", CVPR 2023. R. Jensen et al., "Large Scale Multi-view Stereopsis Evaluation", CVPR 2014, and H. Aanæs et al., "Large-Scale Data for Multiple-View Stereopsis", IJCV 2016 (DTU). The DTU images, masks and cameras come from the NeuS release: P. Wang et al., "NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction", NeurIPS 2021, building on L. Yariv et al., "Multiview Neural Surface Reconstruction by Disentangling Geometry and Appearance", NeurIPS 2020 (IDR).

## Licences of components

This project's own code may be released openly. Its default pipeline also uses models with **non-commercial** licences. That is fine for research and course use, but anyone running the full pipeline inherits those terms. Weights are downloaded by `make setup` and never committed. Each non-commercial part sits in its own stage, so it can be swapped for a commercially licensed alternative if needed.

| Component | Stage | Licence | Commercial alternative |
|---|---|---|---|
| COLMAP | Poses | BSD-3-Clause | — |
| LightGlue (code and matcher weights) | Matching | Apache-2.0 | — |
| SuperPoint weights | Keypoints | Magic Leap, non-commercial | ALIKED (BSD-3-Clause), DISK (Apache-2.0) |
| VGGT-1B | Object depth | CC BY-NC 4.0 | DA3-Base, MapAnything (Apache-2.0), VGGT-1B-Commercial |
| DA3-Giant (code Apache-2.0) | Object depth with `FUSION=1` | CC BY-NC 4.0 | DA3-Base (Apache-2.0) |
| MoGe-3 | Background depth | MIT | — |
| BiRefNet (code and weights) | Masks for `make construct` without supplied masks; optional mask completion | MIT | — |
| SAM-HQ | Optional mask refinement | Apache-2.0 | — |
| Open3D, PyTorch, SciPy, rich, tqdm, halo | Fusion, utilities | MIT / BSD | — |
| MASt3R, VGGT-Ω, DA3-Large/Giant, 2DGS (`recon/evaluation/baselines`) | Comparisons only | Non-commercial (2DGS: Gaussian-Splatting License) | — |

Code from non-commercial repositories is called as a separate dependency (a pip package, or a checkout under `third_party/` for comparisons), never copied into this project's files.

## DTU compared with published methods

Mean chamfer over the standard 15 scans (mm, official protocol). Published methods are given DTU's calibrated cameras; ours estimates them unless stated.

| Method | Cameras | Chamfer | Time per scan |
|---|---|---|---|
| 3DGS | given | 1.96 | 11 min |
| NeRF | given | 1.49 | >12 h |
| SuGaR | given | 1.33 | ~1 h |
| VolSDF | given | 0.86 | >12 h |
| NeuS | given | 0.84 | >12 h |
| 2DGS | given | 0.80 | 19 min |
| GOF | given | 0.74 | 1–2 h |
| QGS | given | 0.73 | 50 min |
| Neuralangelo | given | 0.61 | >128 h |
| PGSR | given | 0.53 | 1 h |
| DiGS | given | 0.46 | 1.2 h |
| Voxurf | COLMAP poses, true intrinsics | 3.48 | — |
| Voxurf + PoRF pose refinement | COLMAP poses refined, true intrinsics | 0.85 | +2.5 h refinement |
| **Ours** | estimated (intrinsics and poses) | **0.886** | ~4–7 min |
| **Ours** | true intrinsics, our poses (orientation alignment) | **0.594** | ~4–7 min |
| **Ours** | given | **0.556** | ~4–7 min |

Our surface error after scaled ICP is about 0.51 mm in all three settings: the differences are camera placement. DTU's camera has its principal point about 30 px off centre, which self-calibration from a narrow lens can't recover; with its intrinsics known, our own poses come within 0.04 mm of the calibrated ones. Sources: 2DGS (Huang et al., SIGGRAPH 2024), QGS (Zhang et al., 2024), DiGS (2025), PoRF (Bian et al., ICLR 2024).
