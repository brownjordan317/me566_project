# Results report

Generated 2026-10-12 by `make report` (`recon/evaluation/benchmark/report.py`) from the runs listed in [PIPELINE.md](PIPELINE.md); regenerate it after new runs instead of editing it. All numbers are ours unless marked published. Method details, conventions and the history behind each result are in PIPELINE.md.

**Setting.** Our pipeline takes RGB images and object masks only: no calibration, no poses, no depth sensor. Published DTU and MobileBrick results are given the dataset's calibrated cameras (or ARKit poses); we show ours with estimated cameras and, on DTU, with the true cameras for a like-for-like comparison. On Tanks and Temples every method estimates its cameras.

## MobileBrick (18 test scenes, official protocol)

Official protocol: the mesh is placed in the reference frame (ours by a similarity fitted to our estimated camera centres), rigid ICP, visibility crop, 100k samples each side; precision/recall/F1 at 2.5 and 5 mm, chamfer as the sum of both directions. Ours estimates cameras and lens from the images; "scale removed" repeats the score after a scaled ICP, separating our RGB-only scale error from surface quality.

| Method | Cameras | Masks | F1 @ 2.5 mm | F1 @ 5 mm | Chamfer mm | Time per object |
|---|---|---|---|---|---|---|
| TSDF-Fusion (depth sensor) | given | dataset | 28.8 | 53.4 | 13.78 | — |
| COLMAP MVS | given | dataset | 71.1 | 88.7 | 5.26 | — |
| NeRF | given | dataset | 43.5 | 73.5 | 7.98 | >10 h |
| NeuS | given | dataset | 73.7 | 89.3 | 4.74 | >10 h |
| Voxurf | given | dataset | 75.1 | 90.4 | 4.71 | — |
| Voxurf | ARKit | dataset | 69.2 | 88.4 | 5.30 | — |
| Voxurf + PoRF | ARKit, refined | dataset | 75.7 | 90.7 | 4.67 | +2.5 h |
| **Ours** | estimated | dataset | **64.9** | **87.0** | **5.77** | 2.5 min |
| **Ours, scale removed** | estimated | dataset | **78.8** | **90.7** | **4.29** | 2.5 min |
| **Ours** | estimated | BiRefNet (automatic) | **65.5** | **87.1** | **5.65** | 2.5 min |
| **Ours, scale removed** | estimated | BiRefNet (automatic) | **79.3** | **91.2** | **4.15** | 2.5 min |

## DTU (15 scans, official protocol)

Mean chamfer of the DTU evaluation (DTUeval-python port). Published methods use DTU's calibrated cameras; ours is shown with estimated cameras, with true intrinsics only, and with the true cameras.

| Method | Cameras | Chamfer mm | Shape chamfer mm | Time per scan |
|---|---|---|---|---|
| 3DGS | given | 1.96 | — | 11 min |
| NeRF | given | 1.49 | — | >12 h |
| SuGaR | given | 1.33 | — | ~1 h |
| VolSDF | given | 0.86 | — | >12 h |
| NeuS | given | 0.84 | — | >12 h |
| 2DGS | given | 0.80 | — | 19 min |
| GOF | given | 0.74 | — | 1–2 h |
| QGS | given | 0.73 | — | 50 min |
| Neuralangelo | given | 0.61 | — | >128 h |
| PGSR | given | 0.53 | — | 1 h |
| DiGS | given | 0.46 | — | 1.2 h |
| Voxurf | COLMAP poses, true intrinsics | 3.48 | — | — |
| Voxurf + PoRF | COLMAP poses refined, true intrinsics | 0.85 | — | +2.5 h |
| **Ours** | estimated (no calibration) | **0.886** | 0.505 | 2.2 min |
| **Ours** | true intrinsics, our poses | **0.818** | 0.504 | 2.2 min |
| **Ours** | true intrinsics, our poses, orientation alignment | **0.594** | 0.504 | 2.2 min |
| **Ours** | given (true intrinsics and poses) | **0.556** | 0.510 | 2.2 min |

Shape chamfer (after scaled ICP) is the same in all our camera settings, so the gap to the given-camera result is camera placement. DTU's principal point lies about 30 px from the image centre, which self-calibration from its narrow lens cannot recover; see PIPELINE.md, "Current results".

## Tanks and Temples (6 training scenes, official protocol)

F-score at each scene's threshold after the official trajectory alignment and ICP refinement. Every method here, ours included, uses cameras estimated from the images. Ours runs in scene mode (no masks, one shared focal) and is scored on its scene mesh.

| Method | Barn | Caterpillar | Courthouse | Ignatius | Meetingroom | Truck | Mean | Time per scene |
|---|---|---|---|---|---|---|---|---|
| 3DGS | 0.13 | 0.08 | 0.09 | 0.04 | 0.01 | 0.19 | 0.09 | 14 min |
| SuGaR | 0.14 | 0.16 | 0.08 | 0.33 | 0.15 | 0.26 | 0.19 | >1 h |
| 2DGS | 0.36 | 0.23 | 0.13 | 0.44 | 0.16 | 0.26 | 0.30 | 34 min |
| Geo-NeuS | 0.33 | 0.26 | 0.12 | 0.72 | 0.20 | 0.45 | 0.35 | >24 h |
| NeuS | 0.29 | 0.29 | 0.17 | 0.83 | 0.24 | 0.45 | 0.38 | >24 h |
| GOF | 0.44 | 0.41 | 0.28 | 0.68 | 0.28 | 0.58 | 0.46 | — |
| PGSR | 0.66 | 0.41 | 0.21 | 0.80 | 0.29 | 0.60 | 0.50 | — |
| Neuralangelo | 0.70 | 0.36 | 0.28 | 0.89 | 0.32 | 0.48 | 0.50 | >24 h |
| DiGS | 0.68 | 0.42 | 0.25 | 0.83 | 0.28 | 0.61 | 0.51 | — |
| **Ours** | pending | pending | pending | pending | pending | pending | pending | pending |

## Latency

Wall time per scene on one RTX 4090, one scene at a time, from the first stage to the finished object mesh (texture, scene mesh and point clouds excluded; listed separately). Measured with `recon/evaluation/benchmark/timing.py` from each run's `timing.log`.

| Run | Scene | Frames | Object mesh s | Texture s | Total s |
|---|---|---|---|---|---|
| mobilebrick | audi | 45 | 131 | 19 | 161 |
| mobilebrick | big_ben | 127 | 363 | 137 | 520 |
| mobilebrick | convertible | 63 | 167 | 100 | 280 |
| mobilebrick | jeep | 43 | 122 | 25 | 162 |
| dtu | scan24 | 49 | 143 | 133 | 283 |
| dtu | scan37 | 49 | 126 | 127 | 259 |
| dtu | scan63 | 49 | 109 | 29 | 146 |
| dtu | scan83 | 64 | 133 | 23 | 166 |

| Stage | Median s per scene |
|---|---|
| Texture | 64.4 |
| Camera poses | 38.7 |
| DA3 depth | 24.6 |
| VGGT depth | 18.6 |
| MoGe depth | 15.0 |
| Photometric refinement | 13.1 |
| Weighted object TSDF | 12.0 |
| Point clouds | 6.4 |
| Photometric fusion | 5.0 |
| TSDF fusion | 2.3 |
| Cross-view depth check | 1.7 |
| Scene mesh cleanup | 1.6 |
| MoGe grounding (scene) | 1.4 |

Baselines run here, same GPU (median wall time per scene, including their meshing):

| Method | MobileBrick | DTU |
|---|---|---|
| VGGT feed-forward | — | — |
| 2DGS (30k iterations) | — | — |
| Ours (object mesh) | 2.5 min | 2.2 min |

## Camera accuracy

Relative-pose accuracy over all frame pairs (VGGT / PoseDiffusion protocol): AUC of the larger of rotation and translation-direction error up to 1°, 3° and 30°, and median errors.

| Run | Cameras | AUC@1 | AUC@3 | AUC@30 | Rotation ° | Translation ° |
|---|---|---|---|---|---|---|
| MobileBrick, ours | estimated | 90.08 | 95.50 | 99.04 | 0.37 | 0.35 |
| MobileBrick, VGGT | feed-forward | pending | | | | |
| DTU, ours | estimated | 98.46 | 99.49 | 99.95 | 0.39 | 0.52 |
| DTU, ours | true intrinsics | 99.99 | 100.00 | 100.00 | 0.11 | 0.12 |
| DTU, VGGT | feed-forward | pending | | | | |

## Ablations and run-to-run spread

### MobileBrick

*Pending: 2 more repeat run(s) of the default.*

*Pending: 4 of 4 ablations.*

### DTU

*Pending: 2 more repeat run(s) of the default.*

*Pending: 4 of 4 ablations.*

Sources: MobileBrick (Li et al., CVPR 2023, Table 2); PoRF (Bian et al., ICLR 2024, Tables 2 and 4);
2DGS (Huang et al., SIGGRAPH 2024, Tables 1 and 2); QGS (Zhang et al., 2024); DiGS ("Accurate and Complete
Surface Reconstruction from 3D Gaussians via Direct SDF Learning", 2025, Tables I and II). Times are as reported
by each paper, on their hardware.
