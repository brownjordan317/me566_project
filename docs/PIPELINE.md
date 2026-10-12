# Pipeline reference

How the reconstruction pipeline works, what each part does, the conventions it relies on, and what has been measured. This is the working reference for development: **update it in the same change whenever a stage, default, output or key result changes.** The user-facing guide is `README.md`; this file is the engineering view.

Last updated: 2026-10-12.

## Goal and constraints

- **Input:** an ordered RGB image sequence (or video) of one object plus a binary object mask per frame. `make construct` generates masks with BiRefNet when the user supplies none (`recon/masks/birefnet.py`); benchmarks use the datasets' masks. External mask generators plug in only through `SCAN_MASK_COMMAND`. No camera calibration is assumed. Dataset intrinsics and poses are used only for evaluation and diagnostics.
- **Output:** an object mesh, a full-scene mesh, and coloured point clouds of the object and scene, all as 32-bit binary PLY.
- **Benchmarks:** MobileBrick (18 test scenes, phone captures, CAD reference meshes) and DTU (15 scans, robot-arm camera, laser-scan references).
- **Licences:** non-commercial models are allowed (research project), but each must sit in its own swappable stage and be listed in the README licence table. Weights are downloaded by `make setup` and never committed. Code from non-commercial repositories is never copied into `recon/`.
- **Citations:** any code ported from, or behaviour reimplemented after, another project is credited in its docstring (repository, file, licence, copyright) and in `THIRD_PARTY_NOTICES.md`. Method ideas are cited in docstrings and in the README "Credits and citations" section.
- **Data:** datasets live on the data drive (`/media/megrad/Data_Storage1/datasets`, symlinked into `runs/real/`). Runs and derived data live on the system drive under `runs/`.

## Pipeline at a glance

```text
frames + masks ─▶ cameras ─▶ depth candidates ─▶ fusion ─▶ refinement ─▶ consistency ─▶ TSDF ─▶ object ─▶ cleanup ─▶ mesh/, points/
                 (COLMAP)   VGGT, DA3, MoGe    (per region)  (plane sweep)  (free-space)           (mask votes)
```

| # | Stage (Makefile label) | Module | Output in `RUN/` |
|---|---|---|---|
| 1 | Camera poses | `recon.cameras.solve` | `poses/sparse_txt/`, `poses/summary.json` |
| 2 | Exporting poses | `recon.cameras.export` | `poses/export/trajectory.csv`, `poses/export_object/landmarks.ply` |
| 3 | MoGe depth | `recon.depth.monocular.moge` | `depth/moge_raw/` (object-anchored) |
| 4 | MoGe grounding (scene) | same, `--anchor-region all --reuse-raw-dir` | `depth/moge_scene/` |
| 5 | VGGT depth | `recon.depth.multiview.vggt` | `depth/vggt/` |
| 6 | Combining depth | `recon.depth.combine` | `depth/hybrid/` (VGGT in mask, MoGe-scene outside) |
| 7 | DA3 depth | `recon.depth.multiview.da3` | `depth/da3/` |
| 8 | Photometric fusion | `recon.depth.fusion.filter` | `depth/fused/` |
| 9 | Photometric refinement | `recon.depth.fusion.refine` | `depth/fused_refined/` |
| 10 | Cross-view depth check | `recon.depth.consistency` | `depth/consistent/` |
| 11 | TSDF fusion | `recon.surface.tsdf` | `work/scene.ply` |
| 11b | Weighted object TSDF | `recon.surface.weighted_tsdf` | `work/object_tsdf.ply` |
| 12 | Object extraction | `recon.surface.object` | `work/object.ply` (from `object_tsdf.ply`; `OBJECT_TSDF=scene` uses `scene.ply`) |
| 13 | Mesh cleanup | `recon.surface.clean` | `mesh/object.ply` |
| 14 | Scene mesh cleanup | `recon.surface.clean` | `mesh/scene.ply` |
| 15 | Point clouds | `recon.surface.points` | `points/object.ply`, `points/scene.ply` |
| 16 | Texture | `recon.surface.texture` | `mesh/object.glb` |
| — | Scoring | `recon.evaluation.metrics.*` | `metrics/*.json` |

Stages 7–9 run with `FUSION=1` (default); stage 9 runs with `REFINE=1` (default). With `FUSION=`, stage 10 reads `depth/hybrid/` directly. `OBJECT_DEPTH_INPUT` in the Makefile names the depth that stages 10, 12 and 15 consume.

## Stages in detail

### 1–2. Cameras (`recon/cameras/`)

- **Features** (`matching.py`): SuperPoint keypoints on the full-resolution dataset images (`FEATURE_FRAMES`), at scales 1 and 0.5 (`FEATURE_SCALES`), up to 2048 per scale (`MAX_KEYPOINTS`). Fine scales add precision; coarse scales stay matchable on repetitive texture.
- **Pairs:** COLMAP-style sequential pairs, 10 neighbours plus power-of-two jumps. `LOOP_PAIRS=1` also wraps the sequence end to its start. It was benchmarked as neutral and is off.
- **Matching:** LightGlue, raw matches imported into a COLMAP database (`sfm.py`). Geometric verification runs in parallel `matches_importer` processes on database copies. The WAL checkpoint before copying is required, or the copies are empty.
- **Mapping:** the COLMAP 3.9 mapper (`/tmp/colmap-local`, library path exported by the Makefile). Global bundle adjustment ratios are 1.4 (29% faster, same accuracy). There is one shared SIMPLE_PINHOLE focal, and the principal point stays at the centre.
- **Per-frame focal** (`PER_FRAME_FOCAL=1`, MobileBrick default): split cameras, then bundle-adjust focal lengths per frame. It tracks phone focus breathing (true range about 1.6%, correlation 0.82). DTU's fixed lens must use `PER_FRAME_FOCAL=`, because per-frame focal invents drift there.
- **Rescale:** the model is solved at feature resolution, then rescaled to the working frames (`rescale_text_model`).
- **Reference substitution** (`reference.py`, diagnostic): reference poses at map scale, cameras from the database (true K with `INTRINSICS=dataset`), re-triangulated, rescaled to working frames.

### 3–9. Depth (`recon/depth/`)

- **Grounding** (`grounding.py`): every prediction is fitted to the sparse COLMAP points observed in that frame, with a robust scale-and-shift fit. Object depth uses object-mask anchors; background uses all anchors. Frames are rejected when the fit collapses: `scale_share < 0.5` for metric models, or relative anchor error above 3% (`candidates.py`, `da3.py`).
- **MoGe-3** (`monocular/moge.py`): single-image depth from the full frame. Object-anchored output goes to `moge_raw`, scene-anchored to `moge_scene`, the background source. Frames with fewer than 30 object anchors are skipped, not fatal.
- **VGGT** (`multiview/vggt.py`): multi-view depth in windows of 16 frames, step 8. `unpad_depth` inverts VGGT's padding. Grounded with object anchors.
- **Combine** (`combine.py`): VGGT depth inside the mask, MoGe-scene depth outside. This is the "hybrid".
- **DA3** (`multiview/da3.py`): DA3-Giant fed our COLMAP extrinsics (4×4 world-to-camera) and intrinsics. It returns depth within about 2% of map scale, which is then grounded. Windows of 16, step 8, input resolution 756 (`DA3_RES`), about 0.3 s per frame. 756 beat 504 on all four development scenes: per-pixel median 6–12% and p90 11–17% lower, fused-mesh p90 5–11% lower, coverage up 0.2–2 points. 1008 runs out of memory on 24 GB, even with 8-frame windows.
- **Fusion filter** (`fusion/filter.py`): for VGGT and DA3, scores each candidate's depth on a stride-2 grid of object pixels by photometric NCC (`fusion/photometric.py`).
  - Patches are 9×9 at full resolution, compared against the 6 most similar views, with the best 3 averaged.
  - Scores are pooled over 17×17 working pixels, plus a per-model prior (0 for both).
  - Each object pixel takes the winner's depth; mask pixels without a candidate are cleared.
  - It also writes `*_confidence.npy`, the margin of winner over runner-up.
- **Refinement** (`fusion/refine.py`): plane sweep of ±0.6% in 13 steps around the fused depth.
  - NCC with slanted patches (normals from the fused depth), aggregated over 15×15, a quadratic penalty of 0.02, and parabolic sub-step interpolation.
  - A pixel moves only if its gain exceeds 0.01.
  - Neighbours are taken 10–35° away (`views.neighbours(band=...)`); close views can't resolve millimetres.
- **Consistency** (`consistency.py`): each pixel is projected into 24 reference views spread over the sequence. It's dropped if more views see through it than support it. The agreement tolerance is 1% (`CONSISTENCY_TOLERANCE`; was 3%, about 12 mm at jeep's distance, which let 1–3 mm-off depth into the TSDF).
- **Not in the default path:** `multiview/omega.py` (VGGT-Ω adapter, kept until its gated weights are available). `grounding.ground` (affine fit in depth or inverse-depth space) is shared by DA3 and VGGT-Ω.
- **Removed 2026-10-12** (rejected experiments; findings below; code archived in `/media/megrad/Data_Storage1/me566_archive/2026-10-12_code_cleanup/project_before_cleanup.tar.gz`): single-image candidates (Depth Pro, Depth Anything V2, Marigold; `candidates.py`), smooth bias fields (`bias.py`), the agreement selector (`fusion/agreement.py`), MVInverse albedo (`fusion/intrinsics.py`), MapAnything and π³ adapters, the old MoGe patch refinement and propagation (`multiview/refine.py`, `propagate.py`), and the Makefile's `reconstruct-moge`, `reconstruct-dense`, `reconstruct-vggt-full` arms with their `evaluate-*` targets.

### 10–15. Surface (`recon/surface/`)

- **TSDF** (`tsdf.py`): Open3D VoxelBlockGrid on CUDA.
  - Voxel 0.02 map units, truncation 0.12.
  - `--min-views 3` (`SCENE_MIN_VIEWS`) keeps surface seen by at least 3 frames, which removes floaters.
  - The depth discontinuity filter `tau` is 0.03; depth range 0.5–30.
  - Hash capacity is 3× the estimated block count (minimum 100k) with a GPU memory check, because the CUDA hash map corrupts above about 45% load.
- **Object extraction** (`object.py`): crop the scene mesh to landmark bounds plus a 0.4 margin. Keep triangles whose depth-visible projections fall inside the mask in at least 80% of the views that see them (`VGGT_MASK_VOTE`, at least 2 views), then drop components under 300 faces, and components whose median mask-view count is under 25% of the largest component's (`OBJECT_SUPPORT`, `--min-component-support`). The support filter removes things a mask covered in only a few frames (BiRefNet's chair on big_ben: 8 views against 53); the report lists every component of 1000+ faces with its count.
- **Masks** (`recon/masks/`): `birefnet.py` (default for `make construct`, MIT, pinned Hugging Face revision; `--guide` keeps only foreground pieces overlapping supplied masks), `sam_hq.py` (boundary refinement of supplied masks).
- **Cleanup** (`clean.py`):
  - **Object:** components below 1000 faces removed, holes up to 0.08 filled, 5 Taubin iterations.
  - **Scene:** pieces under 5% of the largest dropped, holes filled, no smoothing.
  - Written as 32-bit PLY (`common/ply.py`). Open3D's `fill_holes` result shares buffers with its tensor mesh, so the tensor mesh must be kept alive.
- **Point clouds** (`points.py`):
  - Back-project the consistent depth; keep pixels that at least 3 other views see within 1%.
  - Voxel-average at 1/500 of the object diagonal (scene ×4), remove outliers, select the object by mask votes.

## Evaluation (`recon/evaluation/`)

- **MobileBrick** (`metrics/mesh.py`): the mesh is placed by a similarity fitted to our camera centres (Umeyama) and scored against the reference mesh vertices.
  - **Official:** surface median, p90 (distance from points sampled on our mesh to the reference *surface*, via Open3D ray casting), and coverage of reference vertices within 10 mm. Until 2026-10-10 accuracy used distance to the nearest reference *vertex*. The CAD meshes are remeshed at about 1 mm edges, so that floor scored the reference itself at 0.38 / 0.56 mm. Numbers in older notes or logs that predate the fix are inflated; re-score kept meshes with `recon.evaluation.metrics.mesh` (it needs the run's `poses/export/trajectory.csv`).
  - **Shape:** the placement is refined by scaled ICP (10 mm then 3 mm) before scoring. Shape median and p90 are the metrics to optimise.
  - `size_error_pct` is the scale removed, positive when ours is too large.
  - **F-scores** (added 2026-10-10): after shape alignment, `precision_{1,2}mm_pct` (our surface within τ of the reference), `recall_{1,2}mm_pct` (reference surface within τ of ours) and `fscore_{1,2}mm_pct`. Accuracy-only metrics reward deleting hard surface; use the F-score to judge accuracy/completeness trade-offs. Earlier sign confusion: the ICP scale maps ours onto the reference.
- **MobileBrick official protocol** (`metrics/mobilebrick.py`, added 2026-10-12, run by `make evaluate` into `metrics/object_official.json`): port of the benchmark's `evaluations/evaluate_3d.py` (MIT). Our mesh is placed by the camera-centre similarity, then the official steps: rigid ICP of reference vertices onto ours (2 cm, 10 iterations, applied only if fitness > 0.99), crop to the scene's `visibility_mask.npy`, 100k Poisson-disk samples per mesh, precision / recall / F1 at 2.5 and 5 mm, chamfer as the *sum* of both mean distances (reported in mm; the paper uses metres). `scaled_*` repeats it after our scaled ICP. Verified: the visibility crop matches the original torch `grid_sample` on >99.9999% of vertices (jeep, big_ben, castle). Compare these numbers, not our F@1/F@2, with published MobileBrick results.
- **DTU** (`metrics/dtu.py`): a port of DTUeval-python. Mesh sampled uniformly (not DTUeval's per-triangle grid), thinned to 0.2 mm, ObsMask and table plane applied, 20 mm cap. Chamfer is the mean of accuracy and completeness. `shape_chamfer_mm` repeats the score after scaled ICP. The official chamfer is the number to compare with published results.
- **Poses** (`metrics/poses.py`, `reprojection.py`): similarity-aligned camera position error, and a COLMAP projection audit.
- **Pose AUC** (`metrics/pose_auc.py`, added 2026-10-12, `metrics/pose_auc.json` from `make evaluate` / `evaluate-dtu`): relative-pose protocol of VGGT / DA3 / PoseDiffusion, reimplemented after DA3's `bench/utils.py` (Apache-2.0). All frame pairs: rotation error and translation-direction error (folded, scale-free); RRA@15, RTA@15, AUC@1/3/5/15/30 (mean over 1° thresholds of the share of pairs whose larger error is below). AUC@30 saturates on our data (≥99%); AUC@1 and AUC@3 discriminate.
- **Tanks and Temples** (`metrics/tnt.py`, added 2026-10-12): port of the official python_toolbox evaluation (MIT) to current Open3D (`o3d.pipelines.registration`), in 2DGS's variant that scores mesh vertices plus face centres. Our camera centres are RANSAC-registered (with scale) to the reference COLMAP trajectory moved by `_trans.txt`, refined by three scaled ICPs in the crop volume, then precision / recall / F at the scene's τ after τ/2 voxel downsampling. Frames match by order (frame i ↔ image i+1 of the image set). Validated on T&T's own COLMAP reconstruction of Barn with its own trajectory: precision 45.6, recall 55.2, F 49.9 at τ = 1 cm, all 410 frames matched.
- **Diagnostics** (`diagnostics/`):
  - `stage_ranking.py` (`make stage-ranking`): reconstructs each scene with ours, truth poses, truth depth, and both.
  - `oracle_depth.py`: renders reference depth into our cameras. `--refine-to OUR_MESH` places the reference by surface fit. Without it, MobileBrick's pose-versus-CAD scale mismatch biases reference depth by up to 3 mm.
  - `reference_surface.py`: Poisson surface from a DTU point cloud, written in metres to `runs/real/dtu_reference_surfaces/`.
  - `recall_loss.py`: assigns each missed reference point to its first failing stage (unobserved, outside mask, cross-view check, inaccurate depth, meshing). Removed in the morning cleanup of 2026-10-12 and restored the same day for the recall work; run it on a run kept with `KEEP_INTERMEDIATE=1`.
  - Removed 2026-10-12 (one-off investigations, findings kept below; archived as above): `complementarity.py`, `depth_consistency.py`, `missing_surface.py` and the `diagnose-*` targets.
- **Benchmark** (`benchmark/run.py`, `make benchmark`): a rich dashboard, two scenes in parallel (`BENCH_JOBS`), a GPU lock file serialising GPU stages. `summarize.py` writes `summary_<metrics>.md`/`.csv`.
- **Comparison** (`benchmark/compare.py`, added 2026-10-12): repeated default runs give each metric's mean ± sd over runs and the median per-scene sd (the noise floor); each ablation arm is paired with default run 1: mean, difference, scenes better/worse, Wilcoxon signed-rank p.
- **Timing** (`benchmark/timing.py`, added 2026-10-12): every Makefile stage appends `<unix time>\t<label>` to `RUN/timing.log` (the `stage` macro); stage time is until the next stage starts, reconstruction ends where scoring starts. With a GPU memory log (`<time>\t<GiB>`, sampled by `cudaMemGetInfo` because NVML can be unavailable) it reports peak memory per stage above idle. Profile runs use `BENCH_JOBS=1` so stages don't wait on the GPU lock.
- **Baselines** (`baselines/`, comparison only, added 2026-10-12):
  - `vggt_feedforward.py`: one VGGT-1B pass over the sequence (≤96 frames, evenly spread) for cameras, intrinsics and depth; written as a COLMAP text model + trajectory + depth (rescaled to median depth 6.5), then our scene TSDF, mask-vote extraction (vote 0.8, support 0.25) and cleanup. Placed by VGGT's camera centres for scoring.
  - `twodgs.py`: 2DGS trained on *our* cameras and sparse points (same-camera comparison), run from `third_party/2d-gaussian-splatting` in its own venv (`~/.venvs/me566-2dgs`, PyTorch 2.7.1 + CUDA 12.8, because the system nvcc is 12.9 and our PyTorch is CUDA 13). The 2DGS paper's DTU recipe: unit-sphere normalisation from our object landmarks, 30k iterations, `--depth_ratio 1 --lambda_dist 1000`, TSDF voxel 0.004 / trunc 0.016, largest cluster, then their DTU mask culling (24 px dilation at 1600 px width, all views). Gaussian-Splatting licence: research/evaluation only. Building its extensions with GCC 13 / CUDA 12.9 needs headers its sources omit; without editing `third_party/`: `CUDA_HOME=/usr/local/cuda-12.9 TORCH_CUDA_ARCH_LIST=8.9 NVCC_APPEND_FLAGS="-include cfloat -include cstdint" CXXFLAGS="-include cstdint" ~/.venvs/me566-2dgs/bin/pip install --no-build-isolation submodules/{simple-knn,diff-surfel-rasterization}`. The 2DGS venv pins **Open3D 0.19.0**: 0.20's legacy `ScalableTSDFVolume` (which 2DGS meshes with) returns empty meshes even for a synthetic plane, while 0.19 works; our own TSDF uses the tensor `VoxelBlockGrid` and is unaffected. Our cameras are SIMPLE_PINHOLE and 2DGS only reads PINHOLE, so the wrapper rewrites them (same focal for x and y). A trained model is reused if meshing has to be redone. Test on DTU scan24 with our cameras: chamfer 0.888 mm (shape 0.676), 574 s training + 13 s meshing, against ours 0.63 (shape 0.36) in about 140 s; 2DGS's published scan24 with true cameras is 0.48. VGGT feed-forward on scan24: chamfer 3.22 (shape 0.85, size −3.1%) in 21 s; its rotations are slightly better than ours there (median relative error 0.28° vs 0.40°), its depth and scale are not.

## Conventions

- **Frames:** `frames/NNNNNNNN.png` (8 digits) at working resolution (MobileBrick 960×720, DTU 800×600). Masks: `masks/NNNNNNNN.png`. Full-resolution dataset images: `DATA/SCENE/image/NNNNNN.{jpg,png}` (6 digits). Match frames by integer stem.
- **Depth files:** `<frame:08d>_grounded.npy`, float32 z-depth in map units, 0 = no depth. Every depth stage directory also has a summary JSON, which the Makefile uses as its target (`moge_inference.json`, `vggt_inference.json`, `hybrid_depth.json`, `candidate.json`, `consistency.json`).
- **Poses:**
  - COLMAP text models store world-to-camera `R, t`; `common/colmap.parse_model` returns them that way.
  - `common/trajectory.load_poses` returns camera-to-world rotation and camera centre: `camera = (X - centre) @ R`, `X = camera @ R.T + centre`.
  - DA3 takes 4×4 world-to-camera matrices.
- **Map units:** COLMAP's arbitrary scale. Roughly 64 mm per unit on jeep. Metric thresholds in code are in map units unless named `_mm`.
- **Progress:** use `recon.common.progress` (`bar`, `spinner`, `done`, `heading`). `NO_PROGRESS=1` or a non-terminal gives plain log lines.
- **Make:** intermediates are `.SECONDARY`. After the last goal, `PRUNE` deletes `depth/`, `work/` and pose scratch unless `KEEP_INTERMEDIATE=1`. GPU stages are prefixed with `$(GPU)` (flock). Command-line variables propagate to the benchmark's sub-makes.
- **Tests:** `make test` / `python3 -m pytest -q`, with `tests/` mirroring `recon/`. Keep all passing; add tests with new modules.
- **COLMAP:** 3.9.1, a user-space extraction of the Ubuntu package (and its missing libraries, without the GLES Qt build) in `~/.local/colmap-3.9.1`; the Makefile's `COLMAP_ROOT` finds it there (or the old `/tmp/colmap-local`) and exports `LD_LIBRARY_PATH`. `/tmp` is wiped on reboot, so nothing long-lived goes there.
- **Viewing outputs:** `make view_ply PATH=file.ply` (or `PLY=file.ply`; `recon/apps/view_ply.py`, 2026-10-12) opens Open3D's native GPU viewer (O3DVisualizer): full meshes and multi-million-point clouds without decimation, vertex or point colours, textures for `.glb`/`.obj`, height shading when uncoloured, settings panel for lighting and point size. A Plotly page was tried first and dropped as too slow for full-resolution outputs. `PATH=` on the make command line replaces the shell's search path for recipes, so the target restores the `PATH` make started with (`HOST_PATH`, captured at parse time) and calls `/usr/bin/env` by absolute path.
- **Dependencies:** pretrained models are libraries, not vendored code: MoGe, LightGlue, VGGT and SAM-HQ (`segment-anything-hq`) are pip packages pinned to the commits the results were measured with (`requirements.txt`); Depth Anything 3 is installed by `make setup` from its pinned commit with `--no-deps`. Nothing in the pipeline reads `third_party/`, which is git-ignored and holds only academic baselines (2DGS) and comparison checkouts. Switching from the old clones to these packages was verified on jeep: F@1 75.82 vs 75.78, shape median 0.362 vs 0.364 mm (run-to-run noise). What this project takes from other work is techniques, reimplemented in `recon/` and credited (THIRD_PARTY_NOTICES.md); copying model architectures in would add bulk without adding anything of ours, and VGGT's licence forbids it.
- **Python:** `python3` is the `~/venv-ardupilot` venv (PyTorch 2.13, CUDA 13). 2DGS has its own venv (see Baselines).
- **After a reboot:** the data drive is not in fstab; mount it with `udisksctl mount -b /dev/nvme0n1p1 --no-user-interaction` (back at `/media/megrad/Data_Storage1`), or every dataset symlink dangles and make tries to re-download. Long queues live in `runs/real/academic/` and run detached (`setsid nohup`).
- **Camera solver options** (2026-10-11, all off by default): `FEATURES=aliked|sift` (default superpoint), `CAMERA_MODEL=pinhole` (separate x/y focal), `REFINE_PP=1`. See Findings: none improved placement.

## Runs and data layout

| Path | Contents |
|---|---|
| `runs/real/mobilebrick_dataset/test/<scene>/` | Dataset: `image/`, `intrinsic/`, `pose/`, `mesh/gt_mesh.ply` (symlink to the data drive) |
| `runs/real/mobilebrick_benchmark/<scene>/` | Prepared sequence: `frames/`, `masks/`, `sequence.json`, `calibration.json` |
| `runs/real/dtu_dataset/<scan>/` | DTU converted: `image/`, `intrinsic/`, `pose/` (metres), `reference/` (stl, ObsMask, Plane) |
| `runs/real/dtu_benchmark/<scan>/` | DTU prepared sequences |
| `runs/real/{mobilebrick,dtu}_sweep_support/` | Current default (2026-10-11): as `…_sweep` plus the support filter (`OBJECT_SUPPORT=.25`) |
| `runs/real/{mobilebrick,dtu}_sweep/` | Default before the support filter: fusion + refinement, DA3 at 756, 1% cross-view tolerance, weighted object TSDF with robust pass |
| `…_sweep_v3/`, `…_sweep_v2/`, `…_sweep_norefine/`, `…_sweep_nofusion/`, `…_sweep_loop/` | Earlier configurations kept for comparison. v3: DA3 at 756 + weighted TSDF with 3% tolerance and one pass. v2: DA3 at 504 and scene-TSDF objects. |
| `runs/real/birefnet_seq/{mobilebrick,dtu}/<scene>/` | Benchmark sequences with BiRefNet masks (`alone/`, linked as `masks`); frames and calibration link to the prepared sequences |
| `runs/real/{mobilebrick,dtu}_sweep_{support,birefnet,birefnet_support,vote06}/` | Mask experiments of 2026-10-11 (see Findings) |
| `runs/real/dtu_sweep_truek/`, `runs/real/dtu_sweep_truecams/<scan>/cams/` | DTU with true intrinsics (our poses), and true intrinsics + poses |
| `runs/real/dev/poses/<variant>/<scene>/` | Camera-only runs of the solver variants (base, pp, pin_pp, exh_pp, aliked, aliked_pin_pp) |
| `runs/real/academic/` | Queue and helper scripts for the academic evaluation (`queue.sh`, `rescore_official.sh`, `score_pose_auc.sh`, `download_tnt*.sh`, `gpu_sampler.py`); `queue.log` |
| `runs/real/{mobilebrick,dtu}_profile/` | Timing runs (one scene at a time; dev scenes / scan24, 37, 63, 83), GPU log `runs/real/profile_gpu.tsv` |
| `runs/real/{mobilebrick,dtu}_sweep_repeat{2,3}/` | Repeats of the default (run 1 = `…_sweep_support`) |
| `runs/real/{mobilebrick,dtu}_ablation_{nofusion,norefine,scenetsdf,nocheck}/` | Leave-one-out ablations of the default (`FUSION=`, `REFINE=`, `OBJECT_TSDF=scene`, `CONSISTENCY_TOLERANCE=100`) |
| `/media/megrad/Data_Storage1/datasets/tanks_and_temples/` | T&T training scenes: `<Scene>/images_zip/<Scene>/*.jpg` (from the official bucket, or Drive with resource keys for Truck and Caterpillar), evaluation files in `eval/<Scene>/` (`trainingdata_eval.zip`) |
| `runs/real/mobilebrick_stage_ranking/` | Stage ranking results for all 18 scenes (DTU stopped after building surfaces) |
| `runs/real/dev/<scene>/` | Development runs with intermediates for jeep, audi, convertible, big_ben. `ours` (no fusion), `default`, `oracle` (truth depth), `photo_*`, `refine_*`, `features/` |
| `runs/real/dtu_reference_surfaces/` | Poisson reference surfaces for DTU diagnostics |

Development discipline: jeep, audi, convertible and big_ben are the development scenes. Results are confirmed on all 18 MobileBrick scenes and on DTU, which is never used for tuning.

## Current results (default pipeline)

| | MobileBrick (18, median over scenes) | DTU (15) |
|---|---|---|
| F-score at 1 / 2 mm | 54.1 / 75.1% | — |
| Recall / precision at 1 mm | 47.1 / 63.8% | — |
| Shape median / p90 | 0.671 / 3.12 mm | — |
| Official surface median | 1.846 mm | — |
| Coverage ≤ 10 mm | 95.2% | — |
| Chamfer, mean / median | — | 0.886 / 0.680 mm |
| Shape chamfer, mean / median | — | 0.505 / 0.468 mm |

DTU with the dataset's cameras (2026-10-11, current defaults; `runs/real/dtu_sweep_truek/` true intrinsics with our poses, `runs/real/dtu_sweep_truecams/<scan>/cams/` true intrinsics and poses via the stage-ranking reference substitution):

| Cameras | Chamfer mean / median | Accuracy | Completeness | Shape chamfer |
|---|---|---|---|---|
| Estimated (default) | 0.886 / 0.680 | 0.848 | 0.924 | 0.505 |
| True intrinsics, our poses | 0.818 / 0.665 | 0.757 | 0.879 | 0.504 |
| True intrinsics, our poses, orientation alignment (`pose_aligned_chamfer_mm`) | 0.594 | | | |
| True intrinsics and poses | 0.556 / 0.504 | 0.542 | 0.569 | 0.510 |

Shape chamfer is unchanged across the three, so the whole gap is placement, not surface quality. Where the placement error comes from (2026-10-11):
- Camera-centre residuals are lateral (across the view axis), 0.4–1.3 mm, with 0.1–0.2° orientation error: the pivot ambiguity of a narrow lens (DTU about 31° field of view, cameras about 630 mm from the object).
- The official alignment fits a similarity to camera centres only. Its rotation is weakly fixed by centres on a cap, and the 600 mm lever arm turns 0.06–0.34° into 0.5–3 mm at the object even with true intrinsics.
- Taking the rotation from the camera orientations instead (`mesh.pose_similarity`, reported as `pose_aligned_*` by `metrics/dtu.py`) gives 0.594 mm with true intrinsics: our poses are nearly as good as DTU's (0.556).
- With estimated intrinsics it fails (2.28 mm): DTU's principal point is (823, 619), about 30 px from the centre we assume, which tilts every estimated camera by the same 0.6°. Fitting a shared camera-frame rotation in the alignment recovers that bias (0.58–0.70° on every scan) but improves placement only modestly (median 1.34 to 1.23 mm).
- Self-calibration doesn't fix it: estimating the principal point (`REFINE_PP=1`) gets within 5–15 px but leaves pose error at 0.90 against 0.94 mm and hurts MobileBrick badly (jeep 0.84 to 4.19 mm); separate x/y focal (`CAMERA_MODEL=pinhole`, DTU's pixels are 0.3% non-square) and exhaustive pairs change nothing; ALIKED keypoints (`FEATURES=aliked`) cut the focal error on scan37/63/83 (−1.5/−1.1/−1.4% to −0.8/−0.65/−0.4%) but not pose error. Runs in `runs/real/dev/poses/<variant>/<scene>/`. Published means, all with the dataset's cameras: 3DGS 1.96, NeRF 1.49, SuGaR 1.33, VolSDF 0.86, NeuS 0.84, 2DGS 0.80, GSDF 0.80, GOF 0.74, QGS 0.73, Neuralangelo 0.61, PGSR 0.53, DiGS 0.46. With COLMAP poses and true intrinsics, Voxurf scores 3.48 and 0.85 after PoRF pose refinement (Bian et al., ICLR 2024; PoRF assumes known intrinsics, so compare it with our true-intrinsics rows). Our scans take about 4–7 minutes each, two at a time on one GPU.

MobileBrick under its official protocol (2026-10-12; `metrics/object_official.json`, mean of the 18 test scenes; published rows from the MobileBrick paper, Table 2, and PoRF, Table 4):

| Method | Poses | F1 @ 2.5 mm | F1 @ 5 mm | Chamfer (mm, sum) |
|---|---|---|---|---|
| Ours, dataset masks (`…_sweep_support`) | estimated (RGB only) | 64.9 | 87.0 | 5.77 |
| Ours, BiRefNet masks (`…_sweep_birefnet_support`) | estimated (RGB only) | 65.5 | 87.1 | 5.65 |
| Ours, dataset masks, scale error removed (`scaled_*`) | estimated | 78.8 | 90.7 | 4.29 |
| COLMAP MVS | given | 71.1 | 88.7 | 5.26 |
| NeuS (>10 h per object) | given | 73.7 | 89.3 | 4.74 |
| Voxurf | given | 75.1 | 90.4 | 4.71 |
| Voxurf | ARKit | 69.2 | 88.4 | 5.30 |
| Voxurf + PoRF | ARKit, refined | 75.7 | 90.7 | 4.67 |

The official alignment is rigid (no scale), so our 1–2.5% focal over-estimate on these phone captures (object about 1% too large, ~2 mm on 20 cm) costs about 14 points of F1 @ 2.5 mm; the official ICP's fitness test passed on only 8 of 18 scenes. With that scale removed the surface beats every published row. As on DTU, the remaining gap is camera calibration.

The same official protocol over the kept MobileBrick sweeps (mean of 18; scaled = after our scaled ICP). These are the historical configurations, not leave-one-out ablations (those are queued):

| Sweep | F1 @ 2.5 / 5 mm | Chamfer mm | Scaled F1 @ 2.5 | Scaled chamfer |
|---|---|---|---|---|
| `…_sweep_nofusion` (VGGT hybrid, no fusion) | 58.7 / 83.7 | 6.62 | 72.5 | 5.21 |
| `…_sweep_norefine` (fusion, no refinement) | 60.4 / 85.0 | 6.36 | 75.7 | 4.87 |
| `…_sweep_v2` (+ refinement, DA3 504, scene-TSDF object) | 63.1 / 85.8 | 6.17 | 76.8 | 4.77 |
| `…_sweep_v3` (DA3 756, weighted object TSDF) | 65.0 / 87.0 | 5.76 | 78.6 | 4.29 |
| `…_sweep` (1% check, robust pass) | 64.9 / 87.0 | 5.77 | 78.8 | 4.28 |
| `…_sweep_support` (support filter, current) | 64.9 / 87.0 | 5.77 | 78.8 | 4.29 |

Fusion and refinement carry the gains under the official metric (+6.3 scaled F1 @ 2.5 from no fusion to v3); the last two changes are neutral on it.

Pose accuracy (`metrics/pose_auc.json`, mean over scenes): MobileBrick AUC@1 / @3 / @30 = 90.1 / 95.5 / 99.0, median relative rotation / translation error 0.37° / 0.35°. DTU 98.5 / 99.5 / 99.95, 0.39° / 0.52°; with true intrinsics 99.99 / 100 / 100, 0.11° / 0.12° (the principal-point bias again).

Run time (`benchmark/timing.py`; `runs/real/{mobilebrick,dtu}_profile`, one scene at a time on an RTX 4090; seconds from the first stage to scoring):

| Scene | Frames | Total | Texture | Scene mesh + clouds | Object mesh only |
|---|---|---|---|---|---|
| jeep | 43 | 162 | 25 | 15 | 122 |
| audi | 45 | 161 | 19 | 11 | 131 |
| convertible | 63 | 280 | 100 | 13 | 167 |
| big_ben | 127 | 520 | 137 | 20 | 363 |
| DTU scan24 | 49 | 283 | 133 | 7 | 143 |
| DTU scan37 | 49 | 259 | 127 | 6 | 126 |
| DTU scan63 | 49 | 146 | 29 | 8 | 109 |
| DTU scan83 | 49 | 166 | 23 | 10 | 133 |

The object mesh, the evaluated output, takes about 2–2.5 minutes for 43–49 frames and scales roughly linearly with frames (big_ben, 127 frames: 6 minutes). Texturing (xatlas unwrap plus projection) varies most, 19–137 s, with mesh complexity. Largest geometry stages: camera poses (35–40 s), DA3 (24 s), VGGT (18–21 s), MoGe (15 s), photometric refinement (11–14 s), weighted object TSDF (11–14 s). Peak GPU memory above idle: 20.1 GiB (DA3-Giant at 756 px); VGGT 13, scene TSDF 9–12, everything else under 6. The DTU profile overlapped with CPU-only rescoring, so its pose times may be slightly high.

### History of the main changes

| Change | MobileBrick shape median / p90 | DTU mean chamfer |
|---|---|---|
| VGGT hybrid (no fusion) | 0.862 / 3.26 | 0.965 |
| + photometric fusion of VGGT and DA3 | 0.772 / 3.05 | 0.916 |
| + plane-sweep refinement | 0.723 / 2.80 | 0.884 |
| + DA3 at 756 px, confidence-weighted object TSDF | 0.707 / 3.14 (F@1mm 50.4 to 53.5%) | 0.889 (shape chamfer 0.601 to 0.515) |
| + 1% cross-view tolerance, robust TSDF pass | 0.671 / 3.12 (F@1mm 54.1%; 15 of 18 better) | 0.886 (shape chamfer 0.505; 13 of 15 better) |

### Where the remaining error is

The stage ranking (all 18 MobileBrick scenes, `runs/real/mobilebrick_stage_ranking/`, scored with the old vertex metric; its poses were pruned, so it can't be re-scored) showed that truth poses change nothing and truth depth halves the error. Re-measured with the corrected metric on the development scenes (`runs/real/dev/<scene>/{oracle,default}`):

| Scene | Truth depth (meshing loss only) | Default pipeline |
|---|---|---|
| jeep | 0.085 / 0.361 | 0.340 / 1.252 |
| audi | 0.179 / 0.760 | 0.482 / 1.393 |
| convertible | 0.157 / 0.613 | 0.719 / 3.016 |
| big_ben | 0.262 / 0.950 | 0.848 / 2.655 |

Depth is the dominant error (0.3–0.6 mm of the median); meshing loses 0.1–0.26 mm median and up to about 0.95 mm p90 with perfect depth.

Meshing variants (object-only TSDF at 2× and 4× finer voxels, narrow or wide truncation, 0 or 5 smoothing passes; four scenes, perfect and real depth) found:

- The **truncation band width** decides it, not the voxel size. A narrow band (6 voxels) reproduces perfect depth precisely: jeep 0.085 / 0.361 to 0.038 / 0.117 mm.
- A wide band (0.12 map units, the current default) averages real depth noise. On real depth no variant beat the current settings reliably: jeep 0.340 / 1.250 current, 0.355–0.470 with a narrow band; big_ben improved but jeep, audi and convertible lost.
- Smoothing makes no measurable difference.
- The finest object-only volume raised coverage by 2–4 points (thin parts survive), at a cost in accuracy.

Defaults stay as they are. Revisit the narrow band once depth is more accurate. The harness used was a meshing-only rerun on kept dev runs (TSDF, then `surface.object`, then `surface.clean`, then `metrics.mesh`).

**DTU stage ranking with the current pipeline** (`runs/real/dtu_stage_ranking_v3/`; scan24, 37, 63, 83). True cameras (poses and intrinsics) roughly halve the official chamfer, while shape chamfer barely moves:

| Scan | Ours: chamfer / shape / size error | True cameras |
|---|---|---|
| scan24 | 0.628 / 0.364 / +0.28% | 0.443 / 0.356 / +0.14% |
| scan37 | 1.214 / 0.663 / +1.90% | 0.713 / 0.655 / +0.33% |
| scan63 | 1.562 / 0.695 / +1.42% | 0.765 / 0.737 / +0.35% |
| scan83 | 2.143 / 1.055 / +1.93% | 1.125 / 1.073 / +0.47% |

- The cause is the estimated focal: 1.0–1.5% short on the hard scans (−0.5% on scan24), which makes objects 1.4–1.9% too large relative to the camera rig.
- Tried without success:
  - exhaustive pairs (`PAIRS=exhaustive`): −1.51 to −1.43% on scan37, −1.36 to −1.03% on scan83;
  - estimating the principal point (`REFINE_PP=1`): slightly worse;
  - DA3's predicted focal: −2.1 to −4.0%, the same direction.
- DTU's arc of cameras leaves focal and object depth weakly separable from images. The truth-depth rungs of that run were invalid (fusion overwrote the injected depth); `stage_ranking.py` now runs them with `FUSION=`.

**The official metrics are now limited by placement, not shape.**
- Both official metrics place our mesh using a similarity fitted to our estimated camera centres. Published DTU methods use the true cameras, so they carry no placement error.
- On DTU the gap between official chamfer (0.889) and shape chamfer (0.515) is that placement error.
- The latest change improved shape chamfer on all 15 scans and the MobileBrick F-score on 16 of 18 scenes, but left both official numbers flat. Widening the TSDF band did not change that, so it's not a TSDF bias.
- To move the official numbers, improve camera placement and scale. Judge geometry by shape and F-score.

## Findings that shaped the design

- **Masks (2026-10-11):**
  - Lowering `VGGT_MASK_VOTE` to 0.6 only compensates for incomplete masks. With dataset masks: MobileBrick mean F@1 53.7 to 54.0 but shape p90 3.48 to 3.55; DTU chamfer 0.886 to 0.890. Kept at 0.8.
  - SAM-HQ refinement of tracker masks gains a little (jeep IoU 0.854 to 0.887); giving it more freedom (bigger box, wider band, looser overlap) made masks worse. CascadePSP changed nothing (0.853).
  - **BiRefNet with no prompt** scored IoU 0.86–0.97 on all 18 MobileBrick scenes and ≥0.985 on 11 of 15 DTU scans, beating a first-frame-prompted video tracker everywhere (the tracker failed outright on 6 MobileBrick scenes). Guided by tracker masks it gives the same masks where tracking holds.
  - Reconstructions from BiRefNet masks, both with the support filter (`runs/real/{mobilebrick,dtu}_sweep_birefnet_support/`; dataset-mask runs in `…_sweep_support/`): MobileBrick median F@1 / F@2 54.9 / 77.2 against 54.2 / 75.2, better on 12 scenes. DTU chamfer 0.868 against 0.863 mm on the 14 scans other than scan37.
  - Failures are about what "the object" is: DTU scan37 (tools masked, bricks not; chamfer 4.05 against 1.22) and big_ben close-ups (F@1 27.6 against 38.5).
  - The support filter is neutral with complete masks (≤0.2 F@1 per MobileBrick scene, ≤0.01 mm DTU chamfer) and removed big_ben's chair (shape p90 490 to 5.8 mm). Default since 2026-10-11.

- **Floaters in the scene mesh** come from surface seen by 1–2 frames. Hence `SCENE_MIN_VIEWS=3` and dropping pieces under 5%.
- **Focal:** the estimated focal is about 1.5% long on MobileBrick and within 0.5% on DTU (except scan37, 63 and 83, about 1.4% short). It changes overall size, not shape: shape after scaled alignment is identical with true and estimated focal. MobileBrick's reference poses and CAD meshes disagree slightly in scale, so the official metric includes a size term.
- **Depth error is mid-scale shape** (4–20 mm wavelength), shared across a model's views. Smooth correction fields recover at most about 40% even fitted to the truth (`bias.py`). Fine detail (high-pass) differs by only 0.1–0.3 mm between models.
- **VGGT and DA3** are equally accurate but err in different places. The best per-pixel pick of the six models would reach about 0.47 mm on jeep; the best per 16×16 block, about 0.52 (old vertex metric, against 0.65 for VGGT alone).
- **Agreement can't pick the right model,** because errors repeat across a model's views. Learned selectors on geometric and image cues didn't generalise. Pooled photometric evidence does.
- **Single-image models** (Depth Pro, Depth Anything V2, Marigold, and MoGe as a candidate) are worse than VGGT in every region type, despite their claims. The filter picks them for under 1% of pixels. Excluded from fusion.
- **Refinement needs wide baselines:** 10–35°, not the closest views. Slanted patches gave no measurable gain. Glossy surfaces (convertible) and 5 MobileBrick scenes lose some p90.
- **Neutral or negative:** loop-closure pairs, scale-only grounding, larger VGGT windows (16 → 48 frames), per-pixel photometric choice, and the learned selector.

## Open items

0. **Academic evaluation (started 2026-10-12; status kept here):**
   - Done: official MobileBrick protocol, pose AUC, stage timing (MobileBrick), comparison tool, T&T evaluation port (validated), DTU true-camera arms.
   - **Paused 2026-10-12 ~09:10 at the user's request** (GPU needed for pipeline work). Done by then: both timing profiles; 8+ of 18 scenes of `mobilebrick_sweep_repeat2`. Not started: the other repeats, all ablations, the baselines on every scene, T&T runs.
   - To resume (both scripts skip finished work; a half-finished scene reruns only its unfinished stages): `setsid nohup runs/real/academic/queue.sh > runs/real/academic/queue.log 2>&1 < /dev/null & disown`, then the same for `queue2.sh` (it waits for `queue finished` in `queue.log`). Mount the data drive first if the machine has rebooted. Remaining GPU time: about 8 h (queue) + 7–10 h (queue2).
   - Queue contents: `queue.sh` runs two repeats of the default and leave-one-out ablations (no fusion, no refinement, scene TSDF for the object, no cross-view check) on both datasets; `queue2.sh` runs the VGGT feed-forward and 2DGS baselines on all 33 scenes (wall time in each run's `latency.json`), then the T&T training scenes (Barn, Caterpillar, Courthouse, Ignatius, Meetingroom, Truck) in scene mode with `PER_FRAME_FOCAL=`, scored on `mesh/scene.ply`.
   - **Report:** `make report` (`recon/evaluation/benchmark/report.py`) writes `docs/REPORT.md` from the runs: MobileBrick and DTU against published results, T&T, latency, camera accuracy, ablations and spread. Published numbers are constants in the script with sources; missing runs show as pending. Regenerate after runs finish instead of editing the file.
   - Known corrections: published "NeuS with COLMAP 3.48 mm" on DTU was Voxurf with COLMAP poses and true intrinsics (PoRF's table); fixed here and in the README.

1. **Meshing:**
   - Settings explored (see "Where the remaining error is"); nothing beats the current wide-band TSDF on today's depth.
   - A confidence-weighted dense object TSDF is built (`surface/weighted_tsdf.py`). Per-pixel weights come from VGGT–DA3 agreement (σ 1%) times viewing angle; band 0.06; whole-image depth carves free space.
     - **Weights also gating surface:** accuracy up a lot (convertible shape 0.719 / 3.016 to 0.567 / 2.252) but coverage down (convertible 79.8% to 69.7%), and F-score down on 2 of 4 scenes.
     - **Weights only averaging, surface needs 3 views (`--min-views`):** F-score above the default on all four scenes at 1 and 2 mm (+0.3 to +2.6 points), but shape median and p90 worse because more hard surface is kept.
     - **Adopted as the default object surface** (2026-10-10): `OBJECT_TSDF=weighted`, band 0.06, σ 1%, angle weight, 3 views.
     - **Robust second pass** (default `--robust-passes 2 --robust-scale .5`): each observation is re-weighted by its agreement with the first pass (iteratively reweighted averaging). Development F@1mm went +0.3 / +0.5 / +0.5 / +1.1 (jeep, audi, convertible, big_ben); F@2mm stayed within ±0.2 except big_ben (+1.1).
   - Recall is the weak side: 26–68% at 1 mm against 56–88% precision.
   - `diagnostics/recall_loss.py` (removed 2026-10-12, archived) attributed missed reference surface to its first failing stage. On jeep (recall 70%), the 30% missed breaks down as:
     - truly absent, about 8%: unobserved 6.7%, outside mask 0.9%, holes 0.8%;
     - "almost there", 21%: depth present but more than 1 mm off 12.0%; accurate depth in at least 3 frames but the fused surface 1–3 mm off 9.4%, because averaging includes the off frames.
   - Recall at 1 mm is therefore mostly an accuracy problem near the threshold.
   - Current default on jeep (`runs/real/dev/jeep/base`, 2026-10-12): recall 70.5%; missed: inaccurate depth 11.4%, unobserved 6.8%, meshing 1–3 tolerances away 8.0%, removed by cross-view check 1.6%, outside mask 0.9%, no surface nearby 0.9%.
   - **Photometric mesh refinement (tried 2026-10-12, rejected; code archived as `refine_mesh.py`):** each vertex searched ±0.6% of depth along its normal for the best slanted-patch NCC between its best-facing view and 6 neighbours 10–35° away (best 3), displacement smoothed over the mesh. On jeep every variant lost: F@1 75.7 → 71.6 without pooling, 65–67 with the score curves pooled over the mesh neighbourhood; shape median 0.37 → 0.54–0.62 mm. Diagnosis: refining the reference CAD mesh itself (placed in our frame) moved it by a median 0.67 mm with no net bias, so the photometric optimum scatters by more than our surface error. At our camera accuracy (centres off by 0.5–1 mm) per-vertex photo-consistency cannot place the surface to better than about half a millimetre; it helps depth only as a choice between hypotheses pooled over image windows (fusion), not as a free refinement. Also found: the object meshes' triangle winding faces inward, so ray-cast visibility needs per-vertex normal orientation (flip towards the cameras that see the vertex).
2. **Remaining depth gap:** more independent multi-view models. Tested so far:
   - **MapAnything** (`multiview/mapanything.py`, removed 2026-10-12): 3.9 mm per-pixel median against about 1.3 for VGGT and DA3, with or without our cameras. Rejected.
   - **π³X** (`multiview/pi3.py`, removed 2026-10-12): 1.35 mm. As a third fusion candidate it gave −2 to −4% on three scenes and +3% on jeep, so it's not adopted yet.
   - **VGGT-Ω** (`multiview/omega.py`): adapter ready; weights still gated.

   DA3 at 756 px is adopted.
3. **Reflections:** MVInverse (feed-forward multi-view albedo, CVPR 2026; `fusion/intrinsics.py` (removed 2026-10-12), needed 8-frame windows on 24 GB) was tested and rejected.
   - Its albedo removes highlights and is view-consistent, but it is produced at about 1022 px and is smoother than the photos.
   - Perfect-depth discriminability on glossy convertible rose only from 25% to 27% (jeep, non-glossy: 33% with RGB).
   - End to end it was worse: convertible fusion+refinement 0.909 / 3.097 mm against 0.740 / 2.746 with RGB; jeep 0.410 / 1.438 against 0.338 / 1.214.
   - Its shading does not reproduce the image when multiplied by albedo (scale or colour-space mismatch), so the residual and de-lit outputs are unusable.
   - **Less lossy "ratio albedo"** (`ratio_albedo/`): the albedo/image ratio is computed at model resolution, upsampled, and applied to the full-resolution photo. It keeps the photo's texture and suppresses highlights.
     - With the current pipeline (weighted TSDF, DA3 at 756): convertible F@1/F@2 47.4/66.5 with RGB, 47.9/66.9 with ratio albedo; p90 2.537 to 2.462 mm.
     - On jeep (matte) it was slightly worse: F 75.5/89.5 to 74.5/89.2.
   - **Albedo only in troubled areas** (`material/` holds roughness and metallicity): blending to ratio albedo where roughness < about 0.3 switched only 0.8% of convertible's pixels (the windscreen), giving F 47.7/66.8.
   - Net: about ±1 F-point for 50–90 s of MVInverse per scene (4–8-frame windows while other GPU jobs run), so it's not adopted. A better trouble mask (metallicity marks the whole glossy car) bounds the gain at the ratio-albedo result above.
4. **DTU scan63 and scan83** get worse with fusion; 5 MobileBrick scenes lose p90 with refinement.
5. **`make construct`** now prepares the capture (working frames, `full/` copies, all-white masks, or `SCAN_MASKS`) and runs the standard `make reconstruct` with `NORMALIZE_DEPTH=6.5` (`recon/apps/construct.py`, `sfm.normalize_depth`).
   - In scene mode DA3 grounds fewer frames (16 of 29 on the London bus video): a whole-frame affine fit misses the 3% check more often.
   - The scene point cloud is sparse there, because point spacing follows the "object" (the whole scene) diagonal.
   - **Masks are user-supplied:** `SCAN_MASKS` (by source file stem) or `SCAN_MASK_COMMAND` (an external generator, given `{frames}` and `{output}`), optionally refined with SAM-HQ (`SCAN_REFINE_MASKS=1`). Without masks the scene is reconstructed. This project must not ship or reference any particular mask generator.
6. **Texture mapping** (done): `surface/texture.py` writes `mesh/object.glb`.
   - xatlas UV unwrap; Open3D's UVAtlas segfaults on non-manifold vertices.
   - Projects 24 spread frames from the full-resolution images at 4096 px (`project_images_to_albedo`), with meshes simplified to 300k faces.
   - About 25 s per scene; `TEXTURE=` skips it. Far sharper than vertex colours (LEGO studs and grilles readable).
7. **Housekeeping:** the scratch directory holds about 97 GB of old experiments. `nvidia-smi` reports a driver/library mismatch until reboot (CUDA works).
