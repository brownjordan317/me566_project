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
| 7b | PatchMatch depth (`PATCHMATCH=1`, fusion candidate, experimental) | `recon.depth.multiview.patchmatch` (CUDA pycolmap venv) | `depth/patchmatch/` |
| 9a–9d | DA3 scene adaptation (`ADAPT=1`, experimental) | `recon.depth.multiview.da3_adapt`, `.da3 --head-weights`, `recon.depth.combine`, `recon.depth.fusion.refine` | `depth/da3_head.pt`, `depth/da3_adapted/`, `depth/adapted_hybrid/`, `depth/adapted_refined/` (the object depth when adapting) |
| 9e | PatchMatch geometric refinement (`GEOMETRIC=1`; MobileBrick +1.07 F@1, DTU neutral) | `recon.depth.multiview.patchmatch --seed-dir` (CUDA pycolmap venv) | `depth/geometric/` (the object depth when on) |
| 10 | Cross-view depth check | `recon.depth.consistency` | `depth/consistent/` |
| 11 | TSDF fusion | `recon.surface.tsdf` | `work/scene.ply` |
| 11b | Weighted object TSDF | `recon.surface.weighted_tsdf` | `work/object_tsdf.ply` |
| 12 | Object extraction | `recon.surface.object` | `work/object.ply` (from `object_tsdf.ply`; `OBJECT_TSDF=scene` uses `scene.ply`) |
| 13 | Mesh cleanup | `recon.surface.clean` | `mesh/object.ply` |
| 14 | Scene mesh cleanup | `recon.surface.clean` | `mesh/scene.ply` |
| 15 | Point clouds | `recon.surface.points` | `points/object.ply`, `points/scene.ply` |
| 16 | Texture | `recon.surface.texture` | `mesh/object.glb` |
| — | Scoring | `recon.evaluation.metrics.*` | `metrics/*.json` |

Stages 7–8 run with `FUSION=1` (default); stage 9 runs with `REFINE=1` (default) and refines whichever object depth there is: the fused depth, or with `FUSION=` the single model's `depth/hybrid/` (the output folder keeps the name `fused_refined`). With `GEOMETRIC=1`, stage 9e takes that depth as its seed. `OBJECT_DEPTH_INPUT` in the Makefile names the depth that stages 10, 12 and 15 consume. Stages 7b and 9e need a CUDA build of COLMAP: `make setup-patchmatch` installs pycolmap-cuda12 4.2.1 in `~/.venvs/me566-colmap-cuda` (`PATCHMATCH_PYTHON`), because our COLMAP 3.9.1 binary and the main environment's pycolmap have no CUDA.

### Component slots

Every stage hands the next one files in a fixed format (COLMAP text models for cameras, `<frame>_grounded.npy` depth, PNG masks, PLY meshes), so any component that reads and writes them can fill a slot. Pretrained models are used as pinned libraries; techniques taken from other work are reimplemented in `recon/` and credited. Slots and their Makefile settings:

| Slot | Setting | Components |
|---|---|---|
| Masks | `SCAN_MASKS`, `SCAN_MASK_COMMAND`, `SCAN_COMPLETE_MASKS`, `SCAN_REFINE_MASKS` (construct) | BiRefNet (default for captures), your own, SAM-HQ refinement |
| Keypoints | `FEATURES` | SuperPoint (default), ALIKED, SIFT; matched by LightGlue |
| Cameras | `CAMERAS`, `CAMERA_ROUNDS` | `colmap` (incremental mapping, default), `omega` (VGGT-Ω's one-pass cameras, fixed), `omega_ba` (Ω's cameras + bundle-adjustment rounds) |
| Camera model | `CAMERA_MODEL`, `REFINE_PP`, `PER_FRAME_FOCAL`, `INTRINSICS=dataset` | one or two focals, principal point, per-frame focal |
| Depth candidates | `DEPTH_MODEL`, `OMEGA`, `FUSION` | VGGT (default), DA3 (fusion), VGGT-Ω (replacement or third candidate), MoGe (background) |
| Depth adaptation | `ADAPT` | DA3 head fine-tuned on the scene |
| Refinement | `REFINE`, `REFINE_PASSES` | plane sweep, 1 pass (3 being confirmed) |
| Object surface | `OBJECT_TSDF`, `OBJECT_SUPPORT`, `VGGT_MASK_VOTE` | weighted object TSDF (default), scene TSDF |

### Piece-by-piece selection

The pipeline is assembled from the best single piece for each job, whichever library it comes from, with no per-scene training. Each library's outputs are separate pieces (VGGT: cameras, depth, point maps, tracks; VGGT-Ω: cameras, depth and confidence; DA3: depth, rays and cameras, and it accepts known cameras; MoGe: point maps, normals, metric scale; COLMAP: triangulation and bundle adjustment; LightGlue: matching; BiRefNet: masks). Each piece is scored on its own against ground truth, not only through the final mesh:

| Job | Candidate pieces | Scored by | Best so far |
|---|---|---|---|
| Masks | dataset, BiRefNet, SAM-HQ refinement | final mesh | BiRefNet, level with the dataset's masks |
| Keypoints | SuperPoint, ALIKED, SIFT | pose error | SuperPoint |
| Matching | LightGlue, VGGT track head | pose error | LightGlue (track head untested) |
| Focal length | COLMAP self-calibration, VGGT, VGGT-Ω, DA3, MoGe | focal error against the calibration (`diagnostics/focal_sources.py`) | COLMAP (no learned estimate or average beats it) |
| Starting poses | COLMAP incremental mapping, VGGT-Ω, GLOMAP global mapping | pose error after bundle adjustment | COLMAP and VGGT-Ω equal (jeep 0.84 / 0.85 mm); GLOMAP fails on object orbits |
| Pose refinement | COLMAP bundle adjustment | | only candidate |
| Object depth | VGGT, DA3 (given our cameras), VGGT-Ω, MoGe | per-pixel error (`diagnostics/fusion_ceiling.py`) | DA3 alone; VGGT+DA3 photometric choice beats either |
| Depth refinement | plane sweep (ours), COLMAP PatchMatch geometric consistency (seeded with ours) | per-pixel error, mesh | plane sweep then seeded PatchMatch (per-pixel; mesh being tested) |
| Classical stereo candidate | COLMAP PatchMatch from scratch | per-pixel error | accurate but 600 s; seeding is better and 20× faster |
| Surface | weighted object TSDF, 2DGS | mesh | TSDF |

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
- **VGGT-Ω experiment** (`multiview/omega.py`): `DEPTH_MODEL=omega` replaces VGGT; `OMEGA=1` adds Omega as a third photometric fusion candidate. Use a fresh `RUN` for each comparison. The gated checkpoint is cached through Hugging Face. Omega runs up to 128 frames in one pass by default; overlapping windows are grounded separately before averaging. The default remains VGGT + DA3. `grounding.ground` (affine fit in depth or inverse-depth space) is shared by DA3 and Omega.
- **Removed 2026-10-12** (rejected experiments; findings below; code archived in `/media/megrad/Data_Storage1/me566_archive/2026-10-12_code_cleanup/project_before_cleanup.tar.gz`): single-image candidates (Depth Pro, Depth Anything V2, Marigold; `candidates.py`), smooth bias fields (`bias.py`), the agreement selector (`fusion/agreement.py`), MVInverse albedo (`fusion/intrinsics.py`), MapAnything and π³ adapters, the old MoGe patch refinement and propagation (`multiview/refine.py`, `propagate.py`), and the Makefile's `reconstruct-moge`, `reconstruct-dense`, `reconstruct-vggt-full` arms with their `evaluate-*` targets.

### 10–15. Surface (`recon/surface/`)

- **TSDF** (`tsdf.py`): Open3D VoxelBlockGrid on CUDA.
  - Voxel 0.02 map units, truncation 0.12.
  - `--min-views 3` (`SCENE_MIN_VIEWS`) keeps surface seen by at least 3 frames, which removes floaters.
  - The depth discontinuity filter `tau` is 0.03; depth range 0.5–30.
  - Hash capacity is 3× the estimated block count (minimum 100k) with a GPU memory check, because the CUDA hash map corrupts above about 45% load.
- **Weighted object TSDF** (`weighted_tsdf.py`): dense grid over the object's bounds, voxel 0.02, band 0.06 map units; per-pixel weight = VGGT–DA3 agreement (σ 1%) × viewing angle, plus `--confirmed-dir` (pixels PatchMatch's seeded geometric pass confirmed weigh 1, the rest `--unconfirmed-weight` 0.2; off by default). `OBJECT_TSDF_ARGS` passes extra options from make.
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
- **Benchmark** (`benchmark/run.py`, `make benchmark`): a rich dashboard, two scenes in parallel (`BENCH_JOBS`), a GPU lock file serialising GPU stages. `BENCH_SCORES_ONLY=1` (`--scores-only`) builds only what scoring needs, skipping texture, point clouds and the scene mesh (texturing castle's 900k-face mesh took over 17 minutes). Two parallel scenes gain little when both are GPU-bound: each GPU stage roughly doubles while waiting on the lock (aston: 466 s shared vs about 215 s alone). `summarize.py` writes `summary_<metrics>.md`/`.csv`.
- **Comparison** (`benchmark/compare.py`, added 2026-10-12): repeated default runs give each metric's mean ± sd over runs and the median per-scene sd (the noise floor); each ablation arm is paired with default run 1: mean, difference, scenes better/worse, Wilcoxon signed-rank p.
- **Timing** (`benchmark/timing.py`, added 2026-10-12): every Makefile stage appends `<unix time>\t<label>` to `RUN/timing.log` (the `stage` macro); stage time is until the next stage starts, reconstruction ends where scoring starts. With a GPU memory log (`<time>\t<GiB>`, sampled by `cudaMemGetInfo` because NVML can be unavailable) it reports peak memory per stage above idle. Profile runs use `BENCH_JOBS=1` so stages don't wait on the GPU lock.
- **Baselines** (`baselines/`, comparison only, added 2026-10-12):
  - `vggt_feedforward.py`: one VGGT-1B pass over the sequence (≤96 frames, evenly spread) for cameras, intrinsics and depth; written as a COLMAP text model + trajectory + depth (rescaled to median depth 6.5), then our scene TSDF, mask-vote extraction (vote 0.8, support 0.25) and cleanup. Placed by VGGT's camera centres for scoring.
  - `pgsr.py` (2026-10-13): PGSR from `third_party/PGSR` (commit de24f1a, research and non-profit use only) in the 2DGS venv, its `diff-plane-rasterization` built with the 2DGS flags; PGSR imports only `pytorch3d.transforms.quaternion_to_matrix`, so the venv has a one-function stand-in instead of pytorch3d. Reuses the 2DGS wrapper's workspace and mask culling. Recipe as PGSR's `scripts/run_dtu.py` (`--ncc_scale 0.5`, our frames already at its `-r 2` size, TSDF voxel 0.002 normalised), or a warm start (`--init-points` our dense clouds, `--train-args` to start its geometric losses early).
  - Both Gaussian baselines ignore the principal point (they assume the image centre): since 2026-10-13 the 2DGS workspace shifts each frame to centre it and crops (DTU's is about 12 px off at 800 px); without it, 2DGS with the dataset's cameras put scan24's surface about 3 mm off (chamfer 2.86 mm). Earlier 2DGS runs used our cameras, whose principal point is the centre, so they were unaffected.
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
- **GPU lock:** every GPU stage runs under `flock $(GPU_LOCK)`. The camera stage takes it inside `solve.py` (`--gpu-lock`) only around SuperPoint/LightGlue, so COLMAP's CPU mapping doesn't hold the GPU; before 2026-10-12 it took no lock and failed with out-of-memory when another scene's DA3 (about 20 GB) was running.
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
- Self-calibration doesn't fix it: estimating the principal point (`REFINE_PP=1`) gets within 5–15 px but leaves pose error at 0.90 against 0.94 mm and hurts MobileBrick badly (jeep 0.84 to 4.19 mm); separate x/y focal (`CAMERA_MODEL=pinhole`, DTU's pixels are 0.3% non-square) and exhaustive pairs change nothing; ALIKED keypoints (`FEATURES=aliked`) cut the focal error on scan37/63/83 (−1.5/−1.1/−1.4% to −0.8/−0.65/−0.4%) but not pose error. Runs in `runs/real/dev/poses/<variant>/<scene>/`. Published means, all with the dataset's cameras: 3DGS 1.96, NeRF 1.49, SuGaR 1.33, VolSDF 0.86, NeuS 0.84, 2DGS 0.80, GSDF 0.80, GOF 0.74, QGS 0.73, Neuralangelo 0.61, PGSR 0.53, DiGS 0.46. With COLMAP poses and true intrinsics, Voxurf scores 3.48 and 0.85 after PoRF pose refinement (Bian et al., ICLR 2024; PoRF assumes known intrinsics, so compare it with our true-intrinsics rows). Our scans take about 2–2.5 minutes each to the object mesh, one at a time on one GPU (`runs/real/dtu_profile`: 109–143 s; two at once roughly doubles each because GPU stages wait on each other).

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
- **Focal length by source (2026-10-12, `diagnostics/focal_sources.py`, `runs/real/components/focal/`):** median focal of 32 frames against the dataset calibration; mean absolute error / median signed error. MobileBrick (18): COLMAP 1.51% / +1.53%, VGGT 1.63 / +1.56, DA3 (504 px, no poses) 2.77 / +2.83, VGGT-Ω (512) 3.73 / +3.68, MoGe-3 8.15 / −7.23. DTU (15): COLMAP 0.56 / −0.28, DA3 0.90 / −0.62, VGGT-Ω 6.44 / +6.38, VGGT 15.2 / +18.4, MoGe 39.7 / −39.7 (the MoGe figures may reflect how its normalised intrinsics are read; not pursued). Averages of COLMAP with any model are worse than COLMAP alone (MobileBrick 1.54–2.14%, DTU 0.72%+). The focal slot stays with COLMAP. On MobileBrick every good source reads about 1.5% long, so part of the "object 1% too large" error may lie in the dataset calibration rather than our solve; untested.
- **GLOMAP (2026-10-12, pycolmap 4.2.1 `global_mapping`, prototype only):** on the dev scenes' own match databases it registers every frame (jeep 43 in 9–13 s against COLMAP's 20 s mapper) but the poses are wrong: median position error jeep 0.17–0.20 m, audi 0.15, convertible 0.10, big_ben 0.31 against 1–2 mm. Rotation averaging folds the orbit (blocks of frames 90° or 180° off). Not fixed by a good focal prior (VGGT's), re-verification with pycolmap 4, recasting or dropping planar pairs, keeping only nearby-frame pairs, or six rounds of bundle adjustment afterwards. pycolmap 4's incremental mapper on the same database gives 1.6 mm, so the database is not the cause. Mapping is about 20 s of the roughly 90 s camera stage anyway; features and matching are the rest. Rejected; no code kept.
- **VGGT-Ω in VGGT's place, fair test (2026-10-12, `runs/real/dev_omega{512,416}_swap/`):** fusion with DA3 and 3 refinement passes, against VGGT (`mobilebrick_sweep_passes3`). F@1 VGGT / Ω 512 / Ω 416 (`vggt_omega_1b_416_reproduce.pt`, the newest checkpoint, Sept 2026): jeep 76.67 / 75.96 / 76.60; audi 72.86 / 71.68 / 71.45; convertible 46.70 / 45.63 / 45.11; big_ben 39.54 / 38.33 / 38.24. F@2 is mixed (Ω 512 better on audi and convertible). VGGT is ahead at F@1 on all four by 0.1–1.6 points. Depth-stage time per scene is noisy under GPU sharing (VGGT 18–85 s, Ω 14–42 s), a small share of the 6–8 minute run either way. VGGT stays.
- **COLMAP PatchMatch stereo (2026-10-12, `multiview/patchmatch.py`, jeep, `diagnostics/fusion_ceiling.py --extra`):** per-pixel error on object pixels, median / p90 / within 1 mm / time. Our fused + refined depth 0.95 / 3.56 / 52%; the truth-chosen best of VGGT and DA3 per pixel 0.72 / 2.63 / 61%. PatchMatch from scratch with COLMAP defaults (20 sources, 5 iterations, geometric pass): 0.63 / 3.79 / 66% (90% of the pixels), 602 s. Photometric pass only: 0.94–1.14 / 8.6–9.6, 50–229 s. Geometric pass with 8 sources, 3 iterations, window step 2: 0.73 / 3.36 / 61%, 114 s. The accuracy comes from the geometric-consistency pass, not the random search. **Seeded** (our depth and its normals written as the photometric output, which COLMAP then skips; only the geometric pass runs): from our refined depth, 8 sources, 1 iteration, step 2: 0.65 / 2.35 / 67%, 26 s; 3 iterations 0.64 / 2.34, 66 s; 5 sources 0.68 / 2.39, 23 s; window step 1 0.65 / 2.35, 66 s; seeded from the unrefined fusion 0.75 / 2.63 / 60%. Our plane sweep and the seeded geometric pass work best together. Being measured on meshes (`runs/real/dev_geometric/`).
- **Seeded PatchMatch on meshes (2026-10-12, `runs/real/dev_geometric/`, 3 refinement passes, against `mobilebrick_sweep_passes3`):** F@1 / F@2 / median: jeep 76.67 → 77.11 / 89.17 → 88.20 / 0.497 → 0.470 mm; audi 72.86 → 74.55 / 87.38 → 87.44 / 0.673 → 0.657; convertible 46.70 → 48.43 / 66.17 → 66.27 / 1.865 → 1.786; big_ben 39.54 → 42.31 / 73.75 → 76.26 / 1.392 → 1.305 (p90 3.98 → 3.24). F@1 up on all four (+0.4 to +2.8). The 30% lower per-pixel depth error barely reaches the mesh, so the object TSDF (band 0.06, weights built for VGGT/DA3 noise) is now what limits it; voxel, band and confirmed-pixel weights are being swept (`runs/real/dev/<scene>/geometric_*`, `runs/real/academic/mesh_sweep.sh`). In the pipeline the stage takes about 54 s on jeep (with undistortion and loading; 26 s for PatchMatch itself).
- **VGGT-Ω pieces beyond cameras and depth (2026-10-12, jeep, prototypes in the session scratchpad):** the package exposes a per-frame DINO-style ViT-L (`aggregator.patch_embed`), multi-view tokens at layers 4/11/17/23 (frame and global halves, 2048-d), the dense head's 256-d quarter-resolution map, `depth_conf`, and a text-alignment head that reads one language embedding per sequence (not per pixel). As descriptors, nearest-neighbour matching against reference-mesh correspondences: the patch tokens and layers 4–11 put 74–76% of matches within 24 px (one 16-px patch ≈ 26 working px) and 17–19% within 8 px for neighbouring frames, 34–41% within 24 px six frames apart; later layers are worse (layer 23 global: 23%); the dense map is sharper for neighbours (29% within 8 px) but fails six frames apart (12% within 24 px). Coarse, so no use for millimetre matching where SuperPoint + LightGlue and NCC work. Ω depth on object crops (mask box + 15%, a removed `omega.py --object-crop` prototype): 1.53 → 2.29 mm median, worse (each frame gets its own zoom). Input resolution 512 / 640 / 768 (512 checkpoint): 1.53 / 1.58 / 1.68 mm; 512 is best. Seeding the PatchMatch geometric pass from Ω alone: 1.05 mm median, DA3 alone 0.92, the unrefined VGGT+DA3 fusion 0.75, the refined fusion 0.65; the seed's quality carries through, so the fusion and our plane sweep stay in front of it.
- **VGGT-Ω grounding, confidence and flip averaging (2026-10-12, jeep, 512 checkpoint, one pass over all 43 frames):** median / p90 per-pixel error: per-frame affine fit to the frame's sparse points (current) 1.53 / 5.52 mm; per-frame scale only 1.55 / 5.22; one scale for the whole sequence 3.02 / 7.87 (Ω's depth is not scale-consistent across frames in our map, so the per-frame fit stays); flip-averaged 1.53 / 5.34. `depth_conf` ranks Ω's own error (Spearman −0.32; top quarter 1.05 mm, the rest 1.77) but marks easy pixels, not pixels where Ω is better: on Ω's most confident quarter VGGT is 0.87, DA3 0.99, the refined fusion 0.76 and seeded PatchMatch 0.54 mm, and Ω beats both VGGT and DA3 on only 33% of those pixels. A confidence-gated Ω candidate would not help the fusion.
- **VGGT versus VGGT-Ω as DA3's partner, and all three (2026-10-12, jeep, per-pixel against the reference):** VGGT 1.28 / p90 4.16 mm, Ω (512) 1.53 / 5.52, DA3 1.19 / 3.65. Error correlation with DA3: VGGT 0.71, Ω 0.55 (Ω's mistakes are more independent). Perfect pick per pixel / per 17×17 block: VGGT+DA3 0.72 / 0.86, Ω+DA3 0.75 / 0.90, all three 0.53 / 0.75. Selectors without the truth do not reach it: our photometric choice 1.08 with two or three models; a geometric vote (each candidate's points checked against all candidates' depth in the 12 nearest frames, 0.5% tolerance) 1.11 / 1.09. After seeded PatchMatch the two-way seed gives 0.66 / p90 2.50, the three-way 0.65 / 2.35 (separate camera solves). On meshes (`runs/real/dev_tri_geometric/`, three-way fusion with Ω 512 + 3 passes + seeded PatchMatch, against the two-way equivalent): F@1 jeep 76.88 vs 77.42, audi 74.58 vs 74.46, convertible 48.98 vs 48.26, big_ben 41.63 vs 42.25; F@2 87.90 / 87.49 / 66.25 / 75.65 vs 88.55 / 87.48 / 66.39 / 76.22. Mixed, no gain: Ω stays out of the default.
- **Judging reconstructions from the images alone (2026-10-12, jeep, prototype):** score = patch NCC (9×9, best 3 of 6 neighbours in the 10–35° band, as the fusion) at the evaluated depth on mask pixels, every 4th pixel. Mesh variants of the meshing sweep (F@1 76.1–77.4): rendered-mesh scores 0.5689–0.5698 and silhouette IoU 0.939–0.941, no discrimination. Depth sets (median score / true median error): fused + refined 0.659 / 0.95 mm, seeded PatchMatch 0.649 / 0.65, seeded from the unrefined fusion 0.631 / 0.75, fused 0.611 / 1.08, seeded from DA3 0.602 / 0.92, VGGT 0.577 / 1.28, DA3 0.577 / 1.20, seeded from Ω 0.575 / 1.05, Ω 0.545 / 1.53. Mostly in order, but the plane sweep maximises this same score and is ranked first while fourth in truth: a judge must use frames (or a signal) the stages did not optimise. Hence `diagnostics/heldout.py`: `split` copies a finished camera solve without every 8th frame (from the 5th) and re-exports it, so a `make reconstruct RUN=…` on that folder builds the object from the other frames; `score` renders each such run's mesh into the held-out frames and averages the held-out image's patch NCC against its nearest training views at the rendered depth (mask pixels the mesh misses count −1), plus mask coverage and silhouette IoU. Tested on six variants × four dev scenes (`runs/real/dev/<scene>/heldout/`, `runs/real/academic/heldout_judge.sh`; 5 of 38–127 frames held out; variants: VGGT only, two-way fusion with 1 or 3 passes, + seeded PatchMatch, + scene TSDF, three-way + seeded PatchMatch; F@1 spans 4–11 points per scene). The mean NCC on the held-out pixels the mesh covers (`score_on_surface`) ranks the variants like the ground truth: Spearman with F@1 jeep 0.89, audi 0.94, convertible 1.00, big_ben 0.77 (F@2 0.14 / 0.94 / 0.94 / 0.94). It picks the three-way variant on all four; the truth's best is that one on audi and convertible and the two-way + PatchMatch one on jeep (0.07 better) and big_ben (0.59 better): 0.16 F@1 lost on average to a perfect choice. The score with misses counted as −1 is worse (Spearman 0.31–0.94), because mask coverage differences are mostly mask-edge noise. So the images, from frames the reconstruction did not use, can rank pipeline choices per scene without ground truth. Cost per candidate: one reconstruction from 7/8 of the frames (about 50 s per variant once model depth is shared).
- **Per-scene setting choice by the held-out judge, all scenes (2026-10-13, `runs/real/auto_{mobilebrick,dtu}/<scene>/`, `runs/real/academic/auto_scene.sh`):** per scene, the pipeline with 3 passes and seeded PatchMatch on all frames (`full`) and on 7/8 of them (`split`), each meshed four ways sharing its depth: band 0.06 (current), 0.04, 0.03, and PatchMatch-confirmed weights in place of the agreement weight (`conf`); the judge (`score_on_surface`) picks one per scene from the split meshes, scored on the full ones. MobileBrick (18, F@1): fixed 0.06 / 0.04 / 0.03 / conf 55.25 / 54.81 / 54.31 / 55.20; judge's picks 54.44 (−0.81); perfect per-scene 55.61; median rank correlation 0.30. DTU (15, chamfer): 0.887 / 0.862 / 0.850 / 0.891; judge 0.849; perfect 0.845; median rank correlation 0.80, but it picks 0.03 on 14 of 15 and fixed 0.03 gives 0.850. Rejected for mesh settings: both judge scores are monotone in the band (the score on covered pixels always favours the narrowest band, which keeps only the sharpest surface; the score counting holes always favours the widest), so they measure the sharpness/coverage trade-off, not accuracy (boat: 0.03 scores best and loses 6 F@1). The dev-scene success compared pipelines differing in depth quality, which held-out images do detect. Side results: the band that suits DTU (0.03: 0.887 → 0.850 mm) costs MobileBrick 0.9 F@1; `conf` ties the agreement weight on both datasets.
- **Seeded PatchMatch on all scenes (2026-10-13, the `full_b06` meshes above against `…_sweep_passes3`, separate camera solves):** MobileBrick F@1 54.18 → 55.25 (+1.07, better on 17 of 18, Wilcoxon p = 0.0003), F@2 74.99 → 75.74 (14 of 18), median error 1.964 → 1.936 mm. DTU chamfer 0.886 → 0.887 (7 of 15 better, p = 0.60), shape chamfer 0.503 → 0.504. Confirmed on MobileBrick, neutral on DTU. Kept opt-in (decision 2026-10-13): the gain is not yet large enough to justify the separate CUDA pycolmap environment (`make setup-patchmatch`).
- **Fixes (2026-10-13):** seeded PatchMatch crashed when an earlier stage had rejected a frame's depth (castle, camera, 6 DTU scans); such frames now get PatchMatch's own photometric pass. The judge's scoring did not take the GPU lock.
- **Worst MobileBrick scenes (2026-10-13, `runs/real/diag/<scene>`, 3 passes, intermediates kept):** F@1 / precision / recall at 1 mm: colosseum 18.3 / 51.2 / 11.1, castle 20.4 / 33.7 / 14.6, big_ben 39.5 / 60.0 / 29.5 (jeep 76.7 / 83.6 / 70.8); recall trails precision by about 10 points on almost every scene. `recall_loss` (shape-aligned, so camera placement is factored out), share of reference surface missed at 1 mm by cause, colosseum / castle / big_ben: inaccurate depth 38 / 39 / 26%, meshing (surface 1–3 mm away) 15 / 14 / 19%, unobserved 13 / 9 / 5%, cross-view check 1 / 3 / 1%, mask and missing depth ≤ 2.5%. Refined depth median 2.14 / 2.73 / 2.11 mm against 0.89 on jeep: fine repetitive LEGO detail (arches, battlements, tower). Seeded PatchMatch: 1.68 / 2.24 / 1.59 mm (coverage 84 / 49 / 77%).
- **castle's reference poses have a jump (2026-10-13):** our camera error is 14.2 mm median, but frames 0–90 and 91–107 each match the reference to 1.4 / 0.7 mm when fitted alone; together they differ by a rigid 0.68°. Our solve is consistent across that seam (105 verified pairs with about 1,440 inliers each; 6,772 3D points seen on both sides reproject at 1.60 px against 1.39 px inside a block, where a 0.68° block error would be about 8.6 px), so the jump is in the dataset's poses (probably ARKit relocalisation when frames 91+ revisit frames 27–43). The scoring places our mesh by those poses: aligned on frames 0–90 only, castle goes from F@1 20.2 / F@2 48.7 / median 7.30 mm / 55.5% within 10 mm to 26.2 / 60.0 / 2.89 / 90.4%. A benchmark artefact of about 6 F@1 on castle; the shape-aligned metrics (`shape_*`) are unaffected.
- **MobileBrick with the dataset's intrinsics (2026-10-13, `runs/real/mobilebrick_sweep_truek/`, `INTRINSICS=dataset`, 3 passes, against `…_sweep_passes3`):** official F1@2.5 64.9 → 70.7 (better on 17 of 18, Wilcoxon p = 0.0001; ours with scale removed 79.7), F1@5 87.2 → 87.9, chamfer 5.75 → 5.32 mm, size error −1.27% → −0.02%; our camera-placed F@1 54.2 → 53.7. The scale error is in our focal estimate, not the dataset's calibration (which reproduces the right size): an exact focal is worth about +6 official F1@2.5; the remaining 9 points to the scale-removed score are other placement error.
- **Where the MobileBrick focal bias comes from (2026-10-13, COLMAP 3.9.1 mapper rerun on each scene's kept match database, `runs/real/auto_mobilebrick/<scene>/full/poses/database.db`):** shared focal against the dataset's median focal, 18 scenes: SIMPLE_PINHOLE (as the pipeline) +1.65% mean (+0.37 to +2.13; castle lowest); RADIAL (k1, k2 refined) +1.27%, with consistent distortion on every scene (k1 +0.027 to +0.049, k2 −0.049 to −0.069: one phone model's lens); principal point fixed at the dataset's value (diagnostic only) +1.62% / +1.25%, so the principal point is not the cause. Starting bundle adjustment from VGGT-Ω's cameras ends at the same focal as from COLMAP's (jeep +1.2% vs +1.0%; Ω's own cameras +5.6%), so camera-pose agreement with Ω cannot remove it. Unmodelled distortion explains about a quarter; the remaining ≈ +1.3% is systematic and consistent with the focal/distance trade-off of object-centric captures with a narrow field of view. Not acted on yet; modelling distortion would need the depth stages to undistort images.
- **Tiling frames for DA3, and full-resolution input (2026-10-13, big_ben / colosseum / jeep, per-pixel against the reference):** DA3 on 2×2 overlapping tiles of the 1920-px photos, each tile with its own intrinsics (principal point shifted) and grounded to the sparse points inside it before feathered stitching: big_ben 5.65 mm median with 10% overlap, 4.52 with 40%, against 2.96 for whole working frames (stitching before grounding: 10.0 mm, 48 of 127 frames rejected). Tiles lose the context DA3 needs; rejected. The whole-photo control found the gain: feeding DA3 the full-resolution photos instead of the 960-px working frames (it still resizes to 756) gives jeep 1.21 → 1.19, big_ben 2.96 → 2.77, colosseum 3.91 → 3.22 mm (p90 13.4 → 11.3); grounding each window before averaging changes nothing (colosseum 3.91 with working frames). The same input for VGGT and VGGT-Ω (`--images`) changes nothing (jeep / big_ben / colosseum: VGGT 1.29 / 3.89 / 3.27 vs 1.29 / 3.90 / 3.25; Ω 512 1.53 / 4.74 / 3.55 vs 1.55 / 4.72 / 3.53): they resize with antialiasing, so they get the same pixels either way.
- **Default since 2026-10-13: models read the original photos** (user decision: depth and camera estimators always start from untouched images). `MODEL_IMAGES` (default `FUSION_IMAGES`, the dataset's or capture's full-resolution photos) is passed as `--images` to MoGe, VGGT, DA3, VGGT-Ω depth and `cameras/feedforward.py`; each still writes its output on the working grid (960 px for MobileBrick), and `MODEL_IMAGES=` restores the working frames. Keypoints (`FEATURE_FRAMES`), fusion, refinement and texture already read the originals. PatchMatch still runs on the working frames (full resolution would change its output grid; untested). All-scene confirmation running (`runs/real/{mobilebrick,dtu}_sweep_fullres/`).
- **Default with original photos, partial check (2026-10-13, `runs/real/mobilebrick_sweep_fullres/`, stopped after 6 scenes):** F@1 against 3-pass defaults: aston 52.09 → 52.38, audi 72.86 → 73.08, beetles 54.14 → 53.96, big_ben 39.54 → 39.78, boat 43.93 → 44.05, bridge 64.79 → 65.02 (+0.15, 5 of 6). DA3's better depth is diluted: the fusion uses it only where it wins and the refinement then reworks it.
- **Choosing between models inside the refinement (2026-10-13, rejected, code removed):** the first refinement pass also swept ±0.6% around each model's own depth (VGGT, DA3; or VGGT, full-resolution DA3, Ω 512) with the penalty relative to each, every pixel keeping the best aggregated score. Per pixel, median / p90 / within 1 mm, today → two models → three: jeep 0.89 / 3.24 / 54% → 0.86 / 3.15 / 55% → 0.87 / 3.21 / 55%; big_ben 2.11 / 10.46 / 28% → 2.06 / 10.89 / 29% → 2.04 / 10.98 / 29%; colosseum 2.14 / 10.82 / 27% → 2.08 / 11.18 / 28% → 2.02 / 10.89 / 29%. Meshes (F@1): jeep 76.64 / 76.95 / 76.25, big_ben 39.44 / 39.27 / 38.72, colosseum 18.30 / 18.29 / 18.60. Medians improve but the photometric test also accepts wrong depths on repetitive detail, raising the tail; no mesh gain. With the geometric vote, per-pixel choice and learned selectors this makes four ways of choosing that fail to realise the three-model ceiling.
- **Averaging instead of selecting (2026-10-13, per pixel, median / p90 mm):** geometric mean of VGGT and DA3 against our photometric selection (fused) and its refinement: jeep 1.02 / 3.42 vs 1.06 / 3.34 (refined 0.89 / 3.23); big_ben 3.01 / 12.00 vs 2.77 / 10.67; colosseum 3.14 / 12.16 vs 2.74 / 11.04. Three models (VGGT, full-resolution DA3, Ω): 1.03 / 3.69, 3.03 / 12.76, 2.85 / 11.75; per-pixel median of three no better. Selection beats averaging on the hard scenes.
- **Seeded PatchMatch on the original photos (2026-10-13, `patchmatch.py --images`, `--window-radius`):** cameras rescaled to the photos, seeds upsampled, output averaged back onto the working grid. jeep: radius 5 (as on working frames) 0.71 / 2.77 mm, 42% of pixels confirmed; radius 10 (same footprint) 0.65 / 2.61, 49%, 128 s; working frames 0.66 / 2.51, 67%, 26 s. colosseum, radius 10: 1.66 / 9.69 mm in 275 s against 1.68 / 9.45 in 65 s on working frames. A tie at four to five times the cost: fine-detail error is not a resolution limit (repetitive structure fools the matching). The option stays; `GEOMETRIC` runs on the working frames.
- **Warm-started Gaussian optimisation on our reconstruction (2026-10-13, DTU with the dataset's cameras, `runs/real/warm_{2dgs,pgsr}/<scan>/`, `runs/real/academic/warm_{2dgs,pgsr}.sh`):** the Gaussians start from our dense coloured point clouds (object + scene, 300k points) instead of SfM points, opacity resets off; PGSR's single- and multi-view geometric losses from iteration 500 (its default is 7000). scan24 chamfer / accuracy / completeness: ours 0.445 / 0.385 / 0.506 (about 2.5 min); ours + PGSR 3k iterations 0.393 / 0.376 / 0.409 (train 205 s, mesh 89 s); 7k 0.387 / 0.368 / 0.406 (539 s, 120 s); PGSR from scratch (its DTU recipe, 30k) 0.520 / 0.595 / 0.445 (1308 s, 103 s); 2DGS from scratch 0.862; 2DGS warm 3k / 7k 1.334 / 1.620 (without a geometric loss the warm start drifts). The PGSR pass mostly adds completeness. In this setup PGSR from scratch reaches 0.520 against its published 0.34, so our setup (800-px frames, centred crop, our culling and scorer) is harsher than the paper's: compare within it. scan37 / scan63 running.
- **Scene-first versus object-first meshing (2026-10-12, seeded-PatchMatch depth, 3 passes, `runs/real/dev/<scene>/geometric{,_scene}`):** F@1 / F@2 / p90 object TSDF vs scene TSDF: jeep 77.42 / 88.55 / 1.27 vs 75.12 / 88.11 / 1.21; audi 74.46 / 87.48 / 1.71 vs 68.90 / 84.68 / 1.81; convertible 48.26 / 66.39 / 3.55 vs 46.50 / 65.19 / 3.43; big_ben 42.25 / 76.22 / 3.30 vs 40.27 / 72.71 / 2.96. The object TSDF wins F@1 by 1.8–5.6 and F@2 on all four; the scene TSDF has the smaller p90 on three. Object-first stays.
- **Object TSDF on seeded-PatchMatch depth (2026-10-12, `runs/real/dev/<scene>/geometric_*`, `mesh_sweep.sh`):** F@1 jeep / audi / convertible / big_ben: current (voxel 0.02, band 0.06) 77.42 / 74.46 / 48.26 / 42.25; band 0.04: 76.69 / 73.54 / 49.65 / 41.82; band 0.03: 76.95 / 73.09 / 49.74 / 41.50; voxel 0.01 with band 0.04 / 0.03 / 0.02: jeep 76.76 / 76.87 / 76.13, audi 74.08 / 73.44 / 73.02, convertible 48.30 / 48.71 / 47.59 (big_ben's 0.01 grid did not finish). PatchMatch-confirmed weights (`--confirmed-dir`, unconfirmed 0.2) on top of the agreement weight: 77.41 / 74.40 / 48.46 / 42.35; in place of it: 77.43 / 74.47 / 49.17 / 42.29 (F@2 +0.07 / −0.04 / +0.46 / +0.27). Current settings kept; replacing the agreement weight by the confirmed weight is neutral to slightly positive.
- **Fix (2026-10-12):** `photometric.Images` keyed frames by frame number while `ncc_scores` and its callers (fusion, refinement) look them up by position in the model's frame list; the two agree only when no frame is missing, so a capture where COLMAP leaves a frame unregistered would crash those stages. Now keyed by position (`tests/depth/test_fusion.py`). Found through the held-out runs.
- **Fusion ceiling (2026-10-12, `diagnostics/fusion_ceiling.py`):** per-pixel depth error on object pixels against the reference rendered into our cameras (median mm / share within 1 mm). jeep: VGGT 1.27/41%, DA3 1.18/44%, our selection 1.04/48%, + refinement 0.88/55%, truth-chosen best per 17×17 tile 0.85/56%, best per pixel 0.71/62%. audi 1.47, 1.32, 1.18, 1.00, 0.98, 0.81; convertible 2.22, 2.10, 1.81, 1.62, 1.42, 1.23; big_ben 3.88, 2.95, 2.77, 2.16, 2.32, 1.93. Selection plus refinement already matches the best tile choice (beats it on big_ben; 0.2 mm short on convertible), and a perfect per-pixel choice would add only 0.15–0.4 mm. With two candidates, a better selector has little left: gains must come from better candidates (a third independent model, or per-scene adaptation).
- **Refinement passes (2026-10-12, `refine.py --passes`, Makefile `REFINE_PASSES`):** each pass is another plane sweep re-centred on the previous result (frames are independent, so repeating per frame equals chaining whole runs; `--passes 1` reproduces the old output exactly). F@1 / F@2 / recall@1 change against one pass on the dev scenes, 3 passes: jeep +1.39 / +0.60 / +1.40, audi +1.31 / +0.69 / +1.56, convertible +1.53 / +1.64 / +1.86, big_ben +0.87 / +1.62 / +0.66; shape medians equal or 0.01–0.03 mm better. 2 passes gave about two thirds of that, 4 almost nothing more; a finer second pass (±0.3%) gained less than repeating ±0.6%. Each pass costs about 12 s per scene. Being confirmed on all scenes (`…_sweep_passes3`) before becoming the default.
- **Layer 2: scene adaptation of DA3 (2026-10-12, `multiview/da3_adapt.py`, experimental):** the multi-model evidence supervises DA3 on the scene itself. DA3's backbone (1.1B, where frames and cameras interact) is frozen and its features cached once per window; only the 50M DPT depth head is trained (Adam 1e-5, 300 steps of 4 frames). Labels are the refined fused depth on object pixels whose fusion confidence (photometric winner-over-runner-up margin) is in the top half of the frame, mapped into DA3's output scale by a per-frame affine fit to the unadapted prediction; elsewhere the head is held to its unadapted output (log-L1, weight 0.1). The DA3 stage then predicts again with `--head-weights`, and fusion, 3 refinement passes, the cross-view check and the weighted TSDF run as usual. jeep (vs 3 refinement passes, `runs/real/dev/jeep/adapt_v1`): DA3 per-pixel median error 1.18 → 0.83 mm, F@1 +0.85, recall@1 +0.80, precision@1 +0.91, F@2 +0.07; adaptation 136 s. All four dev scenes improve (vs 3 refinement passes; F@1 / F@2 / precision@1 / recall@1, shape median, DA3 per-pixel median error before → after): jeep +0.85 / +0.07 / +0.91 / +0.80, 0.34 mm, 1.18 → 0.83; audi +1.27 / +0.04 / +1.55 / +1.08, 0.38 mm (−0.02), 1.32 → 0.95; convertible +1.91 / +0.11 / +3.75 / +1.04, 0.66 mm (−0.08), 2.10 → 1.36; big_ben +1.16 / +1.07 / +2.45 / +0.70, 0.73 mm (−0.04), 2.95 → 2.45. With 3 refinement passes that is F@1 +2.0 to +3.4 over the 2026-10-11 default. Adaptation takes 126–152 s per scene. On every scene the adapted DA3 alone (0.83–2.45 mm) now beats the fused depth (0.89–2.52), so the fusion's choice of VGGT sometimes loses; tested against adapt_v1 (F@1 / precision@1): a +0.02 fusion prior for the adapted DA3 gave jeep +0.21 / +0.53, audi +0.34 / +0.37, big_ben +0.61 / +0.89, convertible −0.32 / −0.79; adapted DA3 alone (VGGT kept in the filter with prior −10, since the filter needs two candidates) gave jeep +0.25 / +0.43, audi +0.27 / +0.52, big_ben +1.84 / +2.82 (F@2 +2.48), convertible −0.47 / −0.68. After adaptation VGGT still helps only on the glossy convertible. Being tested: a second adaptation round trained on the first round's refined output (self-training).
- **Weighted object TSDF settings (2026-10-12, dev scenes):** 3 robust passes at scale 0.25 or 0.15 truncation units, and a 0.015 voxel, moved F@1 by −0.9 to +0.5 (mostly negative on convertible and big_ben). Defaults kept.
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
   - **VGGT-Ω** (`multiview/omega.py`, 2026-10-12): the gated, retrained 416 px checkpoint is available as an opt-in candidate (`OMEGA=1`) or VGGT replacement (`DEPTH_MODEL=omega`). On the four development scenes, VGGT+DA3 / Ω+DA3 / all three gave F@1mm: jeep 75.68 / 76.21 / 75.95; audi 71.60 / 70.01 / 72.46; convertible 45.22 / 42.79 / 45.55; big_ben 38.46 / 37.18 / 38.81. The three-model filter selected Ω for 16–25% of object pixels and improved F@1mm on all four by 0.27–0.86 points. The 18-scene holdout has not been run, so the default remains VGGT+DA3. Results: `runs/real/omega_416_sweep/`, `runs/real/omega_fusion_416_sweep/`.
     - **Depth accuracy** (later 2026-10-12, `diagnostics/fusion_ceiling.py --extra omega`), per-pixel median on object pixels: jeep 1.87 mm (1.54 with the 512 checkpoint) against VGGT 1.27 and DA3 1.18; audi 1.72 vs 1.47 / 1.32; convertible 2.75 vs 2.22 / 2.10; big_ben 6.18 vs 3.89 / 2.96. Worse than both on its own; it helps the fusion only where it is the one that is right. 17–34 s per scene. Package now pinned (vggt-omega 48b23c8, `make setup-omega`) instead of the `third_party/` clone.
     - **Cameras** (`cameras/feedforward.py`: one pass, the package's `encoding_to_camera`, principal point at the centre), against our COLMAP: centre error 2.1–3.2 mm on DTU scan24/37/63 (COLMAP 0.86–1.22) and 5.4–10.3 mm on the MobileBrick dev scenes (COLMAP ~0.8); median relative rotation 0.59–0.90° on DTU, 1.4–1.8° on MobileBrick (COLMAP 0.27–0.45°); focal +2.6 to +6.3% (COLMAP within ±1.5%). Not a replacement for our cameras; plain VGGT's rotations beat Ω's on scan24 (0.28° vs 0.59°). On jeep VGGT and Ω are alike (centre 5.30 / 5.43 mm, AUC@3 46 / 43, AUC@30 93 / 93), so the decoding is sound: their accuracy shows at AUC@30, not at our millimetre scale.
     - **Camera slot** (2026-10-12): `CAMERAS=omega` runs `cameras/feedforward.py` and keeps Ω's cameras fixed while COLMAP triangulates our matches (`solve.py --init-cameras … --init-rounds 0`); `CAMERAS=omega_ba` adds `CAMERA_ROUNDS` (6) rounds of bundle adjustment and re-triangulation (`sfm.initialised_model`) in place of incremental mapping. jeep through `make poses`: colmap 0.84 mm / AUC@3 98.45 / 0.27°; omega 5.43 / 42.67 / 1.62°; omega_ba 0.85 / 98.34 / 0.285°.
     - **Ω as COLMAP's starting point** (experiment: our database's SuperPoint/LightGlue matches triangulated with Ω's cameras fixed, then bundle adjustment with focal refined; repeated rounds re-triangulate with the refined cameras). jeep, against COLMAP's own shared-focal model: 1 round 2.56 mm / AUC@3 79.6 / 0.74°; 3 rounds 1.71 / 97.0 / 0.30°; 6 rounds 1.58 / 97.3 / 0.30° (COLMAP 1.58 / 97.4 / 0.28°). DTU scan37: 1 round 2.13 mm / 86.1 / 0.62° / focal +2.0%; 6 rounds 1.33 / 96.7 / 0.47° / −1.25% (COLMAP 1.22 / 98.2 / 0.45° / −1.5%). Both starts converge to the same bundle-adjustment optimum: camera accuracy is limited by the matches and the camera model (fixed principal point), not by initialisation, so Ω cannot improve it. What Ω could offer is speed and robustness on long captures: one ~3 s pass plus ~15 s of triangulation and adjustment for 49 frames could replace incremental mapping (the whole COLMAP stage took 62 s on scan37, features and matching included), and every frame gets a camera even where incremental mapping would split the sequence. Not built.
     - **Register-only attention** (the paper's 6%-FLOP variant, trained that way; the release keeps global attention in all inter-frame blocks except 2, 6, 9, 14, 20): switching the released weights to register-only at inference (`feedforward.py --register-only`) breaks them (jeep rotations ~90° off, focal +47%) while halving the forward time (2.9 → 1.4 s for 43 frames). Usable only with the paper's register-only checkpoint, which is not released; for us it would give single-pass cameras for captures longer than the 128-frame global pass.
     - **Three-way fusion with 3 refinement passes** (`runs/real/dev/<scene>/tri_omega`), F@1 against two-model fusion with 3 passes: jeep −0.91 (precision −2.40), audi +0.44, convertible −0.30, big_ben +0.30. The one-pass gain above disappears once refinement runs three passes. The scene-adapted DA3 alone (`adapt_da3only`) beats both everywhere: F@1 +1.10 / +1.54 / +1.44 / +2.99, refined depth median 0.80 / 0.90 / 1.43 / 1.91 mm against 0.95 / 1.03 / 1.73 / 2.17 with Ω. Ω is not adopted for depth or cameras.

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
