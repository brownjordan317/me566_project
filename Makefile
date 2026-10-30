.DEFAULT_GOAL := help

PYTHON ?= python3
# COLMAP 3.9.1: a local, user-space extraction of the Ubuntu package if present, else colmap on PATH.
COLMAP_ROOT ?= $(firstword $(wildcard $(HOME)/.local/colmap-3.9.1 /tmp/colmap-local))
COLMAP ?= $(if $(COLMAP_ROOT),$(COLMAP_ROOT)/usr/bin/colmap,colmap)
COLMAP_LIB := $(if $(COLMAP_ROOT),$(COLMAP_ROOT)/usr/lib/x86_64-linux-gnu,)
export LD_LIBRARY_PATH := $(if $(COLMAP_LIB),$(COLMAP_LIB):,)$(LD_LIBRARY_PATH)

SCENE ?= london_bus
# DTU: DATA=runs/real/dtu_dataset SEQ_ROOT=runs/real/dtu_benchmark SCENE=scan24
DATA ?= runs/real/mobilebrick_dataset/test
SEQ_ROOT ?= runs/real/mobilebrick_benchmark
SEQ := $(SEQ_ROOT)/$(SCENE)
# Inputs are RGB frames plus object masks; everything else is estimated.
# Poses are solved on the full-resolution originals with keypoints from two
# scales (fine detail where it is distinctive, coarse where it repeats, e.g.
# patterned rugs), then the model is rescaled to the working frames.
FEATURE_FRAMES ?= $(DATA)/$(SCENE)/image
FEATURE_SCALES ?= 1,0.5
# Keypoints per scale: 2048 x 2 scales halves mapper time against 4096 x 2 with
# pose accuracy unchanged within noise (jeep, london_bus, porsche, DTU scan24).
MAX_KEYPOINTS ?= 2048
# Keypoints for pose estimation: superpoint, aliked or sift (all matched with LightGlue).
FEATURES ?= superpoint
# Estimated camera: simple_pinhole (one focal) or pinhole (separate x and y focal lengths).
CAMERA_MODEL ?= simple_pinhole
# Where cameras come from: colmap (incremental mapping, default); omega (VGGT-Omega's one-pass cameras, kept
# fixed while COLMAP triangulates our matches); omega_ba (Omega's cameras, then CAMERA_ROUNDS rounds of bundle
# adjustment and re-triangulation, replacing incremental mapping: same optimum as colmap, quicker on long
# captures, every frame registered). The omega modes need `make setup-omega`.
CAMERAS ?= colmap
CAMERA_ROUNDS ?= 6
FEEDFORWARD_CAMERAS = $(POSE)/feedforward
CAMERAS_ARG = $(if $(filter omega,$(CAMERAS)),--init-cameras $(FEEDFORWARD_CAMERAS)/sparse_txt --init-rounds 0,$(if $(filter omega_ba,$(CAMERAS)),--init-cameras $(FEEDFORWARD_CAMERAS)/sparse_txt --init-rounds $(CAMERA_ROUNDS),))
# estimate: one shared focal length from the images. dataset: the dataset's
# per-frame intrinsics, for comparison only.
INTRINSICS ?= estimate
INTRINSICS_ARG := $(if $(filter dataset,$(INTRINSICS)),--calibration $(SEQ)/calibration.json --per-frame-intrinsics $(DATA)/$(SCENE)/intrinsic,)
# Each frame's focal length is estimated from the images (phone lenses refocus
# during capture). PER_FRAME_FOCAL= (empty) keeps one shared focal instead,
# which suits fixed-focus cameras such as DTU's.
PER_FRAME_FOCAL ?= 1
# Which frame pairs are matched: sequential (video order, plus jumps) or exhaustive (every pair).
PAIRS ?= sequential
# NORMALIZE_DEPTH=D rescales the camera model so the median scene depth is D map units, so map-unit
# settings fit captures of unknown scale (used by `make construct`; benchmarks already sit near 6-7).
NORMALIZE_DEPTH ?=
# REFINE_PP=1 also estimates the principal point instead of fixing it at the image centre.
REFINE_PP ?=
# LOOP_PAIRS=1 also matches the end of the sequence back to its start, closing
# orbit captures that finish where they began (experimental).
LOOP_PAIRS ?=
# Outputs go to runs/real/<dataset>_runs/<scene>/ (RUN=dir writes elsewhere).
# Kept: mesh/<arm>.ply, metrics/<name>.json, poses/sparse_txt + export*/ (cameras,
# trajectory). Intermediates (depth/, work/, COLMAP database and logs) are
# deleted after the last requested target; KEEP_INTERMEDIATE=1 keeps them.
IS_DTU := $(findstring dtu,$(DATA))
RUN ?= runs/real/$(if $(IS_DTU),dtu,mobilebrick)_runs/$(SCENE)$(if $(filter omega,$(DEPTH_MODEL)),_omega_replacement,$(if $(OMEGA),_omega_fusion,))
KEEP_INTERMEDIATE ?=
# GPU-heavy stages take turns through one lock, so parallel scenes
# (BENCH_JOBS) overlap one scene's CPU work with another's GPU work.
GPU_LOCK ?= /tmp/me566_gpu.lock
GPU := flock $(GPU_LOCK)
# Quiet recipes: one coloured stage label per command; VERBOSE=1 shows commands.
Q := $(if $(VERBOSE),,@)
stage = printf '\033[1;36m▶ %s\033[0m  %s\n' '$(1)' '$(notdir $@)' && mkdir -p '$(RUN)' && printf '%s\t%s\n' "$$(date +%s.%N)" '$(1)' >> '$(RUN)/timing.log'
DEPTH_ROOT := $(RUN)/depth
WORK := $(RUN)/work
MESHES := $(RUN)/mesh
METRICS := $(RUN)/metrics
POSE := $(RUN)/poses
MASKS ?= $(if $(wildcard $(SEQ)/sam_hq/masks/00000000.png),$(SEQ)/sam_hq/masks,$(SEQ)/masks)
MOGE_CHECKPOINT ?=
VGGT_CHECKPOINT_DIR ?=
VGGT_WINDOW ?= 16
VGGT_STEP ?= 8
VGGT_MASK_VOTE ?= .8
# Drop object pieces masked in fewer views than this fraction of the main piece (0: off). Removes
# things a mask briefly covered (a chair in a few frames); neutral with complete masks.
OBJECT_SUPPORT ?= .25
# Pixels either side of the mask edge where a face abstains from the extraction vote (recon/surface/object.py
# --edge-band; 0 = every visible view votes). Edge-on surface projects onto the silhouette and is voted out there.
OBJECT_EDGE_BAND ?= 0
TSDF_DEVICE ?= cuda
# Scene surface must be seen by this many frames; fewer leaves floaters and shards.
SCENE_MIN_VIEWS ?= 3
TSDF_ARGS = --device $(TSDF_DEVICE) --min-views $(SCENE_MIN_VIEWS)
# Accept `make construct path` (relative or absolute folder or video) and SCAN=path.
SCAN_GOAL := $(if $(filter construct,$(MAKECMDGOALS)),$(firstword $(filter-out construct,$(MAKECMDGOALS))),)
SCAN ?= $(SCAN_GOAL)
SCAN_MAX_EDGE ?= 1200
SCAN_FPS ?= 6
# Object masks for a capture: by default BiRefNet masks the salient object in every frame. To
# define the object yourself, supply SCAN_MASKS=dir (one per source image, same file stem) or
# SCAN_MASK_COMMAND='program {frames} {output}' (any external generator writing one NNNNNNNN.png
# per frame); SCAN_COMPLETE_MASKS=1 fills in parts they miss with BiRefNet, SCAN_REFINE_MASKS=1
# refines their edges with SAM-HQ. SCAN_SCENE=1 skips masks and reconstructs the whole scene.
# Automatic masks are grown slightly: a mask 2 px too tight costs 12.6 points of MobileBrick F1 while one
# 10 px too loose costs 0.8 (docs/PIPELINE.md). Scaled to the frame width; 0 turns it off.
SCAN_MASK_DILATE ?= 2
SCAN_MASKS ?=
SCAN_MASK_COMMAND ?=
SCAN_COMPLETE_MASKS ?=
SCAN_REFINE_MASKS ?=
SCAN_SCENE ?=
SCAN_NAME := $(notdir $(patsubst %/,%,$(SCAN)))
SCAN_OUTPUT ?= runs/scans/$(if $(wildcard $(SCAN)/.),$(SCAN_NAME),$(basename $(SCAN_NAME)))
CHECKPOINT_ARG := $(if $(MOGE_CHECKPOINT),--checkpoint $(MOGE_CHECKPOINT),)
RAW := $(DEPTH_ROOT)/moge_raw
SCENE_DEPTH := $(DEPTH_ROOT)/moge_scene
VGGT_DEPTH := $(DEPTH_ROOT)/vggt
DEPTH_MODEL ?= vggt
PRIMARY_DEPTH = $(if $(filter omega,$(DEPTH_MODEL)),$(OMEGA_DEPTH),$(VGGT_DEPTH))
PRIMARY_SUMMARY = $(if $(filter omega,$(DEPTH_MODEL)),$(OMEGA_DEPTH)/candidate.json,$(VGGT_DEPTH)/vggt_inference.json)
# Default (hybrid): VGGT depth on the object, MoGe depth for the background.
HYBRID_DEPTH := $(DEPTH_ROOT)/hybrid
# Default: add DA3 depth and pick VGGT or DA3 region by region with full-resolution photometric
# evidence (recon.depth.fusion.filter); everything downstream uses the fused depth.
# FUSION_ARGS passes extra options, e.g. FUSION_ARGS='--combine mean' to average the candidates instead
# of choosing between them by patch score.
# FUSION= (empty) keeps VGGT object depth alone.
FUSION_ARGS ?=
FUSION ?= 1
DA3_CHECKPOINT ?= depth-anything/DA3-GIANT
# DA3 input resolution: 756 beat 504 on every development scene (p90 5-11% lower).
DA3_RES ?= 756
# Extra recon.depth.multiview.da3 options. Memory is window x resolution squared, and the stage now
# sizes its own window to the free memory, so a higher resolution needs no hand-chosen window:
# DA3_RES=1008 lowers the median depth error on every development scene (jeep -16%, audi -9%,
# convertible -7%, big_ben -2%) for about 1.5x the stage's time, and picks a window of 9 where that
# measurement used 8 (docs/PIPELINE.md).
# Regions for the depth grounding's affine fit (1 = one global fit per frame, the long-standing
# behaviour). A frame holding both a near and a far surface cannot be served by a single affine, which is
# what costs an unmasked object run its thin surface (docs/PIPELINE.md); scenes have the widest range of
# all. Passed to every depth stage.
GROUNDING_REGIONS ?= 1
GROUNDING_ARG = --grounding-regions $(GROUNDING_REGIONS)
DA3_ARGS ?=
FUSION_IMAGES ?= $(DATA)/$(SCENE)/image
DA3_DEPTH := $(DEPTH_ROOT)/da3
# Models (MoGe, VGGT, DA3, VGGT-Omega) read the original photos, not the downsized working frames; they
# resize internally, and their depth comes back on the working grid. MODEL_IMAGES= uses the working frames.
MODEL_IMAGES ?= $(FUSION_IMAGES)
MODEL_IMAGES_ARG = $(if $(MODEL_IMAGES),--images $(MODEL_IMAGES),)
OMEGA ?=
OMEGA_DEPTH := $(DEPTH_ROOT)/omega
# Authors recommend the retrained 416-pixel checkpoint for benchmark comparisons.
OMEGA_CHECKPOINT ?= vggt_omega_1b_416_reproduce.pt
OMEGA_RES ?= 416
OMEGA_WINDOW ?= 128
OMEGA_STEP ?= 64
OMEGA_RESIZE_MODE ?= balanced
OMEGA_CANDIDATE := $(if $(and $(OMEGA),$(filter vggt,$(DEPTH_MODEL))),--candidate omega $(OMEGA_DEPTH) 0,)
OMEGA_DEP := $(if $(and $(OMEGA),$(filter vggt,$(DEPTH_MODEL))),$(OMEGA_DEPTH)/candidate.json,)
# PATCHMATCH=1 adds COLMAP PatchMatch stereo as a fusion candidate (needs a CUDA pycolmap: make setup-patchmatch).
PATCHMATCH ?=
PATCHMATCH_PYTHON ?= $(HOME)/.venvs/me566-colmap-cuda/bin/python
PATCHMATCH_ARGS ?=
PATCHMATCH_DEPTH := $(DEPTH_ROOT)/patchmatch
PATCHMATCH_CANDIDATE := $(if $(PATCHMATCH),--candidate patchmatch $(PATCHMATCH_DEPTH) 0,)
PATCHMATCH_DEP := $(if $(PATCHMATCH),$(PATCHMATCH_DEPTH)/candidate.json,)
FUSED_DEPTH := $(DEPTH_ROOT)/fused
# Then a photometric plane sweep fine-tunes the fused depth; REFINE= (empty) skips it.
REFINE ?= 1
# Plane-sweep passes per frame, each re-centred on the last (dev scenes: 3 passes +1.3 to +1.5 F@1 over 1).
REFINE_PASSES ?= 1
# Cross-view depth check: relative depth difference that counts as agreement. 1% beat 3% on the
# development scenes (F@1mm +0.8/-0.2/+1.1/+0.1, shape p90 lower on all four); 0.5% and 0.25% were mixed.
CONSISTENCY_TOLERANCE ?= .01
REFINED_FUSED_DEPTH := $(DEPTH_ROOT)/fused_refined
# Scene adaptation (ADAPT=1; needs FUSION and REFINE): DA3's depth head is fine-tuned on the scene, labelled
# by the refined fused depth where the photometric fusion is confident (recon.depth.multiview.da3_adapt);
# DA3 then predicts again and its depth, refined REFINE_PASSES times, is the object depth.
ADAPT ?=
DA3_HEAD := $(DEPTH_ROOT)/da3_head.pt
ADAPTED_DA3_DEPTH := $(DEPTH_ROOT)/da3_adapted
ADAPTED_HYBRID_DEPTH := $(DEPTH_ROOT)/adapted_hybrid
ADAPTED_REFINED_DEPTH := $(DEPTH_ROOT)/adapted_refined
ADAPTING := $(and $(ADAPT),$(FUSION),$(REFINE))
# The refinement sweeps whichever object depth there is: the fused depth, or the single model's without FUSION.
REFINE_INPUT := $(if $(FUSION),$(FUSED_DEPTH),$(HYBRID_DEPTH))
PRE_GEOMETRIC_DEPTH := $(if $(ADAPTING),$(ADAPTED_REFINED_DEPTH),$(if $(REFINE),$(REFINED_FUSED_DEPTH),$(REFINE_INPUT)))
# GEOMETRIC=1: COLMAP PatchMatch's geometric-consistency pass, seeded with that depth (make setup-patchmatch).
GEOMETRIC ?=
GEOMETRIC_DEPTH := $(DEPTH_ROOT)/geometric
GEOMETRIC_ARGS ?=
OBJECT_DEPTH_INPUT := $(if $(GEOMETRIC),$(GEOMETRIC_DEPTH),$(PRE_GEOMETRIC_DEPTH))
# GAUSSIAN=1 (make refine-gaussian): our planar-Gaussian refinement, a per-scene optimiser that fits Gaussians to
# every photo and re-exports depth (recon/depth/gaussian_refine.py, needs the gsplat venv: make setup-gaussian).
# It seeds from the first pass's checked depth and point clouds, so it runs after them and the object mesh is
# rebuilt from its depth. Measured better than the PGSR baseline on 17 of 18 MobileBrick scenes (docs/PIPELINE.md).
GAUSSIAN_PYTHON ?= $(HOME)/.venvs/me566-pgsr/bin/python
GAUSSIAN_DEPTH := $(DEPTH_ROOT)/gaussian
GAUSSIAN_ARGS ?=
GAUSSIAN_PLY := $(GAUSSIAN_DEPTH)/gaussians.ply
# Watertight closure and splat-filled texture, the asset-facing finish (make asset). The closure is the
# finished mesh's generalised winding number bounded by the mask hull, and it refuses rather than guess on
# a mesh too open for that to separate inside from outside; the texture takes texels the photos leave
# unpainted from renders of the fitted Gaussians. Both need the gsplat/libigl environment where noted.
SPLAT_VIEWS := $(WORK)/splat_views
WATERTIGHT_MESH := $(MESHES)/object_watertight.ply
WATERTIGHT_ARGS ?=
SPLAT_VIEWS_ARGS ?=
ASSET_MESH ?= $(WATERTIGHT_MESH)
# The closed mesh carries several times the open one's faces (jeep 307,872 against 54,835), so at a fixed
# atlas each face gets a fraction of the texels and the texture goes soft. Decimating first trades a
# little coverage for detail: at 4096 px, 300k faces gave 1.36% of the surface unpainted at Laplacian
# sharpness 12.3, 100k gave 2.04% at 14.0 and 60k 2.13% at 16.0 (docs/PIPELINE.md).
ASSET_MAX_FACES ?= 100000
ASSET_GLB := $(MESHES)/asset.glb
AGREEMENT_DEPTH := $(if $(ADAPTING),$(ADAPTED_DA3_DEPTH),$(DA3_DEPTH))
CONSISTENT_DEPTH := $(DEPTH_ROOT)/consistent
SCENE_MESH := $(WORK)/scene.ply
OBJECT_MESH := $(WORK)/object.ply
# The object surface comes from a confidence-weighted object TSDF (recon.surface.weighted_tsdf):
# pixels count by VGGT-DA3 agreement and viewing angle; surface still needs 3 observing frames.
# OBJECT_TSDF=scene extracts the object from the scene TSDF instead (the earlier default).
OBJECT_TSDF ?= weighted
# Extra recon.surface.tsdf options for the scene mesh, e.g. TSDF_ARGS='--voxel .04'. The default voxel suits an
# object capture; a whole outdoor scene (Tanks and Temples) needs a coarser one to fit the GPU, as the sparse block
# grid's memory grows as 1/voxel^2. Pick the finest that fits, which the stage reports when it does not.
TSDF_ARGS ?=
# Extra recon.surface.weighted_tsdf options, e.g. OBJECT_TSDF_ARGS='--voxel .01 --truncation .03'.
OBJECT_TSDF_ARGS ?=
# The weighted TSDF is a dense grid over the box of these landmarks. Scene-first runs on white masks, where the
# sequence's own landmarks span the whole table (jeep 1.4 M -> 514 M voxels, big_ben 24 M -> 943 M, which does
# not fit the card); pointing this at landmarks exported through the real masks keeps the box object-sized
# while every pixel of depth inside it is still fused, so the masks bound the grid and delete nothing.
WEIGHTED_LANDMARKS ?= $(LANDMARKS)
WEIGHTED_OBJECT_MESH := $(WORK)/object_tsdf.ply
OBJECT_SOURCE_MESH := $(if $(filter weighted,$(OBJECT_TSDF)),$(WEIGHTED_OBJECT_MESH),$(SCENE_MESH))
FINAL_MESH := $(MESHES)/object.ply
# Feature-preserving denoising (recon.surface.denoise) after cleanup: flat panels lose the millimetre bumps of
# view-to-view disagreement while creases and studs stay. Chosen by eye (official and shape within +-0.15 and
# +-0.01 on dev, docs/PIPELINE.md); DENOISE= keeps the cleaned mesh as it is.
DENOISE ?= 1
DENOISE_ARGS ?=
CLEAN_MESH := $(WORK)/object_clean.ply
# Textured copy of the object mesh (photos projected onto a UV atlas) for viewing; TEXTURE= skips it.
TEXTURE ?= 1
TEXTURED_MESH := $(MESHES)/object.glb
TEXTURE_IMAGES ?= $(FUSION_IMAGES)
VGGT_CHECKPOINT_ARG := $(if $(VGGT_CHECKPOINT_DIR),--checkpoint-dir $(VGGT_CHECKPOINT_DIR),)
# Full scenes (background included): pieces under 5% of the main surface are
# dropped and tears up to 4 TSDF voxels filled, but not smoothed.
# About 30-100 MB each; SCENE_MESHES= (empty) skips them.
SCENE_MESHES ?= 1
FINAL_SCENE := $(MESHES)/scene.ply
scene_mesh = $(if $(SCENE_MESHES),$(1),)
# Coloured point clouds of the default pipeline; POINT_CLOUDS= (empty) skips them.
POINT_CLOUDS ?= 1
POINTS := $(RUN)/points
SCENE_POINTS := $(POINTS)/scene.ply
OBJECT_POINTS := $(POINTS)/object.ply
TRAJECTORY := $(POSE)/export/trajectory.csv
LANDMARKS := $(POSE)/export_object/landmarks.ply
BENCH_ROOT ?= runs/real/mobilebrick_sweep$(if $(filter omega,$(DEPTH_MODEL)),_omega_replacement,$(if $(OMEGA),_omega_fusion,))
RANK_ROOT ?= runs/real/$(if $(IS_DTU),dtu,mobilebrick)_stage_ranking
BENCH_SCENES ?= $(sort $(notdir $(patsubst %/image,%,$(wildcard $(DATA)/*/image))))
MODEL := $(POSE)/sparse_txt

# Intermediate stages: if pruned, make treats them as done rather than rebuilding.
.SECONDARY: $(RAW)/moge_inference.json $(SCENE_DEPTH)/moge_inference.json \
  $(VGGT_DEPTH)/vggt_inference.json $(OMEGA_DEPTH)/candidate.json $(HYBRID_DEPTH)/hybrid_depth.json \
  $(DA3_DEPTH)/candidate.json $(OMEGA_DEPTH)/candidate.json $(PATCHMATCH_DEPTH)/candidate.json $(GEOMETRIC_DEPTH)/hybrid_depth.json $(FUSED_DEPTH)/hybrid_depth.json $(REFINED_FUSED_DEPTH)/hybrid_depth.json \
  $(DA3_HEAD) $(ADAPTED_DA3_DEPTH)/candidate.json $(ADAPTED_HYBRID_DEPTH)/hybrid_depth.json $(ADAPTED_REFINED_DEPTH)/hybrid_depth.json \
  $(CONSISTENT_DEPTH)/consistency.json $(SCENE_MESH) $(OBJECT_MESH) $(WEIGHTED_OBJECT_MESH)

# Runs only as part of the last goal on the command line, so a later goal in the
# same command (e.g. `make reconstruct evaluate`) still finds shared depth.
PRUNE_COMMAND = test -n "$(SCENE)" && test -d "$(RUN)/poses" && rm -rf "$(DEPTH_ROOT)" "$(WORK)" \
	"$(POSE)/database.db" "$(POSE)/lightglue_raw_matches.txt" "$(POSE)/sparse" \
	"$(POSE)/sparse_per_frame_focal" "$(POSE)/shared_focal_txt" "$(POSE)/verify_chunks" \
	&& rm -f "$(POSE)"/*.log "$(MESHES)"/*.json "$(POINTS)"/*.json && echo 'Pruned intermediates in $(RUN)'
PRUNE = $(if $(KEEP_INTERMEDIATE),,$(if $(filter $@,$(lastword $(MAKECMDGOALS))),@$(PRUNE_COMMAND)))

.PHONY: watertight asset segment report view_ply stage-ranking prepare-dtu evaluate-dtu setup-vggt benchmark help setup setup-masks download prepare reconstruct rebuild-object construct poses depth scene object evaluate test status refine-masks setup-omega setup-patchmatch

help:
	@echo 'Inputs are RGB frames plus object masks; cameras are estimated from the images.'
	@echo 'make setup                 Install dependencies and download model weights (network required once)'
	@echo 'make download              Download the MobileBrick test split (network required once)'
	@echo 'make prepare               Prepare half-resolution frames, masks and calibration for SCENE'
	@echo 'make reconstruct           Object and full-scene meshes -> RUN/mesh/object.ply, scene.ply'
	@echo '                           DEPTH_MODEL=omega replaces VGGT with VGGT-Omega'
	@echo '                           OMEGA=1 adds VGGT-Omega as a third fusion candidate'
	@echo '                           PATCHMATCH=1 adds COLMAP PatchMatch stereo as a fusion candidate'
	@echo '                           GEOMETRIC=1 runs PatchMatch geometric consistency seeded with the refined depth'
	@echo '  watertight               close the finished object mesh (winding number bounded by the mask hull)'
	@echo '  asset                    closed mesh textured from the photos, gaps filled from splat renders'
	@echo '  refine-gaussian          second pass: planar-Gaussian refinement, then rebuild the mesh from its depth'
	@echo '                           (make setup-gaussian first; GAUSSIAN_ARGS passes options)'
	@echo '                           and point clouds -> RUN/points/object.ply, scene.ply'
	@echo 'make evaluate              Score the object mesh and poses against the MobileBrick reference'
	@echo 'make evaluate-dtu          Chamfer of the object mesh (or EVAL_MESH) against the DTU laser scan'
	@echo 'make benchmark             Reconstruct and score every scene, with a live dashboard and summary'
	@echo 'make stage-ranking         Rerun every scene with truth poses and/or truth depth to rank stages'
	@echo 'make construct path        Full pipeline on your own image folder or video; BiRefNet masks the'
	@echo '                           object unless you bring SCAN_MASKS=dir or SCAN_MASK_COMMAND=...'
	@echo '                           (SCAN_SCENE=1: whole scene, no masks)'
	@echo 'make report                Write docs/REPORT.md from the finished runs'
	@echo 'make view_ply PATH=f.ply    Open a mesh or point cloud in a coloured, interactive viewer'
	@echo 'make test                  Run first-party tests'
	@echo 'Options: SCENE=london_bus  DATA=runs/real/dtu_dataset SEQ_ROOT=runs/real/dtu_benchmark (DTU)'
	@echo '         RUN=dir  BENCH_ROOT=dir  KEEP_INTERMEDIATE=1  PER_FRAME_FOCAL= (one shared focal)'
	@echo '         SCENE_MESHES= (skip full-scene meshes)  POINT_CLOUDS= (skip point clouds)  VERBOSE=1'
	@echo '         SCENE_MIN_VIEWS=3 (frames that must see scene surface; higher = cleaner, smaller)'
	@echo '         TSDF_DEVICE=cpu|cuda  COLMAP=/path/to/colmap  MASKS=/path/to/masks'
	@echo '         FUSION= (skip DA3 and the photometric choice between VGGT and DA3; on by default)'
	@echo '         REFINE= (skip the photometric plane-sweep refinement of the object depth)'
	@echo '         ADAPT=1 (fine-tune DA3 on the scene from the fused evidence; experimental)'

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -c "from huggingface_hub import hf_hub_download; hf_hub_download('Ruicheng/moge-3-vitl', 'model.pt')"
	$(PYTHON) -c "from lightglue import SuperPoint, LightGlue; SuperPoint(max_num_keypoints=4096); LightGlue(features='superpoint')"
	$(PYTHON) -c "from recon.masks.birefnet import load_model; load_model('cpu')"
	# DA3 pins numpy<2; install it (pinned commit) without dependencies and add only what it imports.
	$(PYTHON) -m pip install --no-deps "depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
	$(PYTHON) -m pip install addict plyfile evo "moviepy==1.0.3"
	$(PYTHON) -c "from huggingface_hub import snapshot_download; snapshot_download('$(DA3_CHECKPOINT)')"

setup-vggt:
	$(PYTHON) -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/VGGT-1B')"

# VGGT-Omega (FAIR Noncommercial Research License; gated weights: log in with an approved Hugging Face account).
setup-omega:
	$(PYTHON) -m pip install --no-deps "vggt_omega @ git+https://github.com/facebookresearch/vggt-omega.git@48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6"
	$(PYTHON) -c "from huggingface_hub import hf_hub_download; [hf_hub_download('facebook/VGGT-Omega', f) for f in ('vggt_omega_1b_416_reproduce.pt', 'vggt_omega_1b_512.pt')]"

setup-patchmatch:
	python3 -m venv $(HOME)/.venvs/me566-colmap-cuda
	$(HOME)/.venvs/me566-colmap-cuda/bin/pip install pycolmap-cuda12==4.2.1 numpy==2.5.4 opencv-python-headless==5.0.0.93

setup-masks:
	$(PYTHON) -c "from huggingface_hub import hf_hub_download; hf_hub_download('lkeab/hq-sam', 'sam_hq_vit_b.pth', local_dir='runs/models/sam_hq')"

# Downloading is separate so all reconstruction and evaluation stages can stay offline.
download:
	@test -d $(DATA)/$(SCENE)/image || $(PYTHON) -m recon.inputs.mobilebrick.download

$(DATA)/$(SCENE)/image: download

prepare: $(SEQ)/sequence.json
$(SEQ)/sequence.json: $(DATA)/$(SCENE)/image
	$(if $(IS_DTU),@echo 'DTU is prepared once with: make prepare-dtu' && false,$(PYTHON) -m recon.inputs.mobilebrick.prepare --source $(DATA) --output $(SEQ_ROOT))

# DTU (NeuS/IDR release, reference scans and evaluation masks) under DTU_SOURCE.
DTU_SOURCE ?= /media/megrad/Data_Storage1/datasets/dtu
prepare-dtu:
	@$(call stage,Preparing DTU)
	$(Q)$(PYTHON) -m recon.inputs.dtu.prepare --source $(DTU_SOURCE) --data $(DTU_SOURCE)/converted --output runs/real/dtu_benchmark
	ln -sfn $(abspath $(DTU_SOURCE)/converted) runs/real/dtu_dataset

poses: $(TRAJECTORY) $(LANDMARKS)
$(FEEDFORWARD_CAMERAS)/summary.json: $(SEQ)/sequence.json
	@$(call stage,Feed-forward cameras)
	$(Q)$(GPU) env HF_HUB_OFFLINE=1 $(PYTHON) -m recon.cameras.feedforward $(SEQ) $(MODEL_IMAGES_ARG) --output $(FEEDFORWARD_CAMERAS)

$(POSE)/summary.json: $(SEQ)/sequence.json $(if $(filter colmap,$(CAMERAS)),,$(FEEDFORWARD_CAMERAS)/summary.json)
	@$(call stage,Camera poses)
	$(Q)$(PYTHON) -m recon.cameras.solve --gpu-lock $(GPU_LOCK) $(CAMERAS_ARG) --sequence $(SEQ) $(INTRINSICS_ARG) --output $(POSE) --colmap $(COLMAP) --features $(FEATURES) --camera-model $(CAMERA_MODEL) --max-keypoints $(MAX_KEYPOINTS) $(if $(FEATURE_FRAMES),--feature-frames $(FEATURE_FRAMES) --scales $(FEATURE_SCALES),) $(if $(PER_FRAME_FOCAL),--per-frame-focal,) $(if $(LOOP_PAIRS),--loop,) --pairs $(PAIRS) $(if $(REFINE_PP),--refine-principal-point,) $(if $(NORMALIZE_DEPTH),--normalize-depth $(NORMALIZE_DEPTH),)


$(TRAJECTORY): $(POSE)/summary.json
	@$(call stage,Exporting poses)
	$(Q)$(PYTHON) -m recon.cameras.export $(MODEL) $(POSE)/export

$(LANDMARKS): $(POSE)/summary.json
	@$(call stage,Exporting poses)
	$(Q)$(PYTHON) -m recon.cameras.export $(MODEL) $(POSE)/export_object --sequence $(SEQ)

depth: $(CONSISTENT_DEPTH)/consistency.json

scene: $(SCENE_MESH)
object: $(FINAL_MESH)
$(FINAL_SCENE): $(MESHES)/%.ply: $(WORK)/%.ply
	@$(call stage,Scene mesh cleanup)
	$(Q)$(PYTHON) -m recon.surface.clean $< $@ --smooth-iterations 0 --min-fraction .05 --hole-size .08

# VGGT multiview depth inside the mask, anchored to the same COLMAP map.

$(HYBRID_DEPTH)/hybrid_depth.json: $(SCENE_DEPTH)/moge_inference.json $(PRIMARY_SUMMARY)
	@$(call stage,Combining depth)
	$(Q)$(PYTHON) -m recon.depth.combine --scene-depth-dir $(SCENE_DEPTH) --object-depth-dir $(PRIMARY_DEPTH) --object-depth-kind grounded --masks-dir $(MASKS) --output-dir $(HYBRID_DEPTH)

# The depth models: MoGe (raw and scene grounding), VGGT and DA3. With DEPTH_BATCH (the default) they run as
# jobs in one process (recon.depth.batch), each model loaded once and dropped when no queued job needs it, so
# start-up and weight loading are paid once rather than per stage; each stage's summary keeps its own load_s and
# inference_s. DEPTH_BATCH= runs them as separate processes, with identical arguments and results.
MOGE_RAW_JOB = $(SEQ) --model $(MODEL) --output-dir $(RAW) $(MODEL_IMAGES_ARG) $(CHECKPOINT_ARG) $(GROUNDING_ARG)
MOGE_SCENE_JOB = $(SEQ) --model $(MODEL) --output-dir $(SCENE_DEPTH) $(MODEL_IMAGES_ARG) --anchor-region all --reuse-raw-dir $(RAW) $(CHECKPOINT_ARG) $(GROUNDING_ARG)
VGGT_JOB = $(SEQ) --model $(MODEL) $(MODEL_IMAGES_ARG) --output-dir $(VGGT_DEPTH) --window $(VGGT_WINDOW) --step $(VGGT_STEP) $(VGGT_CHECKPOINT_ARG) $(GROUNDING_ARG)
DA3_JOB = $(SEQ) --colmap-model $(MODEL) --output-dir $(DA3_DEPTH) --checkpoint $(DA3_CHECKPOINT) --process-res $(DA3_RES) $(MODEL_IMAGES_ARG) $(GROUNDING_ARG) $(DA3_ARGS)
DEPTH_BATCH ?= 1
USES_VGGT = $(if $(filter omega,$(DEPTH_MODEL)),,1)
DEPTH_JOBS = --job moge '$(MOGE_RAW_JOB)' --job moge '$(MOGE_SCENE_JOB)' $(if $(USES_VGGT),--job vggt '$(VGGT_JOB)',) --job da3 '$(DA3_JOB)'
DEPTH_ENV = env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

ifneq ($(DEPTH_BATCH),)
$(RAW)/moge_inference.json $(SCENE_DEPTH)/moge_inference.json $(if $(USES_VGGT),$(VGGT_DEPTH)/vggt_inference.json,) $(DA3_DEPTH)/candidate.json &: $(POSE)/summary.json
	@$(call stage,Depth models (MoGe, VGGT, DA3 in one process))
	$(Q)$(GPU) $(DEPTH_ENV) $(PYTHON) -m recon.depth.batch $(DEPTH_JOBS) --report $(DEPTH_ROOT)/depth_batch.json
else
$(RAW)/moge_inference.json: $(POSE)/summary.json
	@$(call stage,MoGe depth)
	$(Q)$(GPU) $(DEPTH_ENV) $(PYTHON) -m recon.depth.monocular.moge $(MOGE_RAW_JOB)

$(SCENE_DEPTH)/moge_inference.json: $(POSE)/summary.json $(RAW)/moge_inference.json
	@$(call stage,MoGe grounding (scene))
	$(Q)$(GPU) $(DEPTH_ENV) $(PYTHON) -m recon.depth.monocular.moge $(MOGE_SCENE_JOB)

$(VGGT_DEPTH)/vggt_inference.json: $(POSE)/summary.json
	@$(call stage,VGGT depth)
	$(Q)$(GPU) $(DEPTH_ENV) $(PYTHON) -m recon.depth.multiview.vggt $(VGGT_JOB)

$(DA3_DEPTH)/candidate.json: $(POSE)/summary.json
	@$(call stage,DA3 depth)
	$(Q)$(GPU) $(DEPTH_ENV) $(PYTHON) -m recon.depth.multiview.da3 $(DA3_JOB)
endif

# This scan's depth jobs as JSON, for assembling one batch over many scans (recon.evaluation.benchmark.run).
depth-jobs:
	@$(PYTHON) -m recon.depth.batch --print $(DEPTH_JOBS)

$(OMEGA_DEPTH)/candidate.json: $(POSE)/summary.json recon/depth/multiview/omega.py
	@$(call stage,VGGT-Omega depth)
	$(Q)$(GPU) $(PYTHON) -m recon.depth.multiview.omega $(SEQ) --colmap-model $(MODEL) $(MODEL_IMAGES_ARG) --masks-dir $(MASKS) --output-dir $(OMEGA_DEPTH) --checkpoint $(OMEGA_CHECKPOINT) --resolution $(OMEGA_RES) --resize-mode $(OMEGA_RESIZE_MODE) --window $(OMEGA_WINDOW) --step $(OMEGA_STEP) $(GROUNDING_ARG)

$(PATCHMATCH_DEPTH)/candidate.json: $(POSE)/summary.json recon/depth/multiview/patchmatch.py
	@$(call stage,PatchMatch depth)
	$(Q)$(GPU) env -u LD_LIBRARY_PATH $(PATCHMATCH_PYTHON) -m recon.depth.multiview.patchmatch $(SEQ) --colmap-model $(MODEL) --output-dir $(PATCHMATCH_DEPTH) $(PATCHMATCH_ARGS)

$(FUSED_DEPTH)/hybrid_depth.json: $(HYBRID_DEPTH)/hybrid_depth.json $(DA3_DEPTH)/candidate.json $(OMEGA_DEP) $(PATCHMATCH_DEP) recon/depth/fusion/filter.py
	@$(call stage,Photometric fusion)
	$(Q)$(GPU) $(PYTHON) -m recon.depth.fusion.filter --model $(MODEL) --masks-dir $(MASKS) --base-depth $(HYBRID_DEPTH) --images $(FUSION_IMAGES) --candidate $(DEPTH_MODEL) $(PRIMARY_DEPTH) 0 --candidate da3 $(DA3_DEPTH) 0 $(OMEGA_CANDIDATE) $(PATCHMATCH_CANDIDATE) --output-dir $(FUSED_DEPTH) $(FUSION_ARGS)

$(REFINED_FUSED_DEPTH)/hybrid_depth.json: $(REFINE_INPUT)/hybrid_depth.json recon/depth/fusion/refine.py
	@$(call stage,Photometric refinement)
	$(Q)$(GPU) $(PYTHON) -m recon.depth.fusion.refine --model $(MODEL) --masks-dir $(MASKS) --depth-dir $(REFINE_INPUT) --images $(FUSION_IMAGES) --output-dir $(REFINED_FUSED_DEPTH) --passes $(REFINE_PASSES)

$(GEOMETRIC_DEPTH)/hybrid_depth.json: $(PRE_GEOMETRIC_DEPTH)/hybrid_depth.json recon/depth/multiview/patchmatch.py
	@$(call stage,PatchMatch geometric refinement)
	$(Q)$(GPU) env -u LD_LIBRARY_PATH $(PATCHMATCH_PYTHON) -m recon.depth.multiview.patchmatch $(SEQ) --colmap-model $(MODEL) --seed-dir $(PRE_GEOMETRIC_DEPTH) --output-dir $(GEOMETRIC_DEPTH) $(GEOMETRIC_ARGS)

$(DA3_HEAD): $(REFINED_FUSED_DEPTH)/hybrid_depth.json recon/depth/multiview/da3_adapt.py
	@$(call stage,DA3 scene adaptation)
	$(Q)$(GPU) env HF_HUB_OFFLINE=1 $(PYTHON) -m recon.depth.multiview.da3_adapt $(SEQ) --colmap-model $(MODEL) --target-dir $(REFINED_FUSED_DEPTH) --confidence-dir $(FUSED_DEPTH) --output $@ --checkpoint $(DA3_CHECKPOINT) --process-res $(DA3_RES)

$(ADAPTED_DA3_DEPTH)/candidate.json: $(DA3_HEAD)
	@$(call stage,DA3 depth (adapted))
	$(Q)$(GPU) env HF_HUB_OFFLINE=1 $(PYTHON) -m recon.depth.multiview.da3 $(SEQ) --colmap-model $(MODEL) --output-dir $(ADAPTED_DA3_DEPTH) --checkpoint $(DA3_CHECKPOINT) --process-res $(DA3_RES) $(MODEL_IMAGES_ARG) $(GROUNDING_ARG) --head-weights $(DA3_HEAD)

$(ADAPTED_HYBRID_DEPTH)/hybrid_depth.json: $(ADAPTED_DA3_DEPTH)/candidate.json
	@$(call stage,Combining depth (adapted))
	$(Q)$(PYTHON) -m recon.depth.combine --scene-depth-dir $(SCENE_DEPTH) --object-depth-dir $(ADAPTED_DA3_DEPTH) --object-depth-kind grounded --masks-dir $(MASKS) --output-dir $(ADAPTED_HYBRID_DEPTH)

$(ADAPTED_REFINED_DEPTH)/hybrid_depth.json: $(ADAPTED_HYBRID_DEPTH)/hybrid_depth.json recon/depth/fusion/refine.py
	@$(call stage,Photometric refinement (adapted))
	$(Q)$(GPU) $(PYTHON) -m recon.depth.fusion.refine --model $(MODEL) --masks-dir $(MASKS) --depth-dir $(ADAPTED_HYBRID_DEPTH) --images $(FUSION_IMAGES) --output-dir $(ADAPTED_REFINED_DEPTH) --passes $(REFINE_PASSES)

$(CONSISTENT_DEPTH)/consistency.json: $(OBJECT_DEPTH_INPUT)/hybrid_depth.json $(TRAJECTORY)
	@$(call stage,Cross-view depth check)
	$(Q)$(GPU) $(PYTHON) -m recon.depth.consistency --model $(MODEL) --trajectory $(TRAJECTORY) --depth-dir $(OBJECT_DEPTH_INPUT) --output-dir $(CONSISTENT_DEPTH) --tolerance $(CONSISTENCY_TOLERANCE)

$(SCENE_MESH): $(CONSISTENT_DEPTH)/consistency.json $(TRAJECTORY) recon/surface/tsdf.py
	@$(call stage,TSDF fusion)
	$(Q)$(GPU) $(PYTHON) -m recon.surface.tsdf $(SEQ) --model $(MODEL) --trajectory $(TRAJECTORY) --depth-dir $(CONSISTENT_DEPTH) --output $@ $(TSDF_ARGS)

$(WEIGHTED_OBJECT_MESH): $(CONSISTENT_DEPTH)/consistency.json $(WEIGHTED_LANDMARKS) recon/surface/weighted_tsdf.py
	@$(call stage,Weighted object TSDF)
	$(Q)$(GPU) $(PYTHON) -m recon.surface.weighted_tsdf $(SEQ) --model $(MODEL) --depth-dir $(CONSISTENT_DEPTH) --landmarks $(WEIGHTED_LANDMARKS) --output $@ --angle-weight $(if $(FUSION),--agreement $(PRIMARY_DEPTH) $(AGREEMENT_DEPTH),) $(OBJECT_TSDF_ARGS)

$(OBJECT_MESH): $(OBJECT_SOURCE_MESH) $(LANDMARKS)
	@$(call stage,Object extraction)
	$(Q)$(PYTHON) -m recon.surface.object $(OBJECT_SOURCE_MESH) --bounds-cloud $(LANDMARKS) --masks-dir $(MASKS) --depth-dir $(OBJECT_DEPTH_INPUT) --trajectory $(TRAJECTORY) --model $(MODEL) --output $@ --min-ratio $(VGGT_MASK_VOTE) --min-component-support $(OBJECT_SUPPORT) --edge-band $(OBJECT_EDGE_BAND)

$(CLEAN_MESH): $(OBJECT_MESH)
	@$(call stage,Mesh cleanup)
	$(Q)$(PYTHON) -m recon.surface.clean $< $@ --hole-size .08

$(FINAL_MESH): $(CLEAN_MESH) $(if $(DENOISE),recon/surface/denoise.py,)
ifneq ($(DENOISE),)
	@$(call stage,Mesh denoising)
	$(Q)$(PYTHON) -m recon.surface.denoise $< --output $@ $(DENOISE_ARGS)
else
	@mkdir -p $(@D)
	$(Q)cp $< $@ && cp $(<:.ply=.json) $(@:.ply=.json)
endif

$(SCENE_POINTS) $(OBJECT_POINTS) &: $(CONSISTENT_DEPTH)/consistency.json $(TRAJECTORY) $(LANDMARKS) recon/surface/points.py
	@$(call stage,Point clouds)
	$(Q)$(GPU) $(PYTHON) -m recon.surface.points $(SEQ) --model $(MODEL) --trajectory $(TRAJECTORY) --depth-dir $(CONSISTENT_DEPTH) --vote-depth-dir $(OBJECT_DEPTH_INPUT) --masks-dir $(MASKS) --bounds-cloud $(LANDMARKS) --scene-output $(SCENE_POINTS) --object-output $(OBJECT_POINTS) --min-ratio $(VGGT_MASK_VOTE)

$(TEXTURED_MESH): $(FINAL_MESH) recon/surface/texture.py
	@$(call stage,Texture)
	$(Q)$(PYTHON) -m recon.surface.texture $(FINAL_MESH) --model $(MODEL) --trajectory $(TRAJECTORY) --images $(TEXTURE_IMAGES) --sequence $(SEQ) --output $@

reconstruct: $(FINAL_MESH) $(if $(TEXTURE),$(TEXTURED_MESH),) $(call scene_mesh,$(FINAL_SCENE)) $(if $(POINT_CLOUDS),$(SCENE_POINTS) $(OBJECT_POINTS),)
	@echo 'Object mesh:  $(abspath $(FINAL_MESH))'
	$(if $(TEXTURE),@echo 'Textured:     $(abspath $(TEXTURED_MESH))',)
	$(if $(SCENE_MESHES),@echo 'Scene mesh:   $(abspath $(FINAL_SCENE))',)
	$(if $(POINT_CLOUDS),@echo 'Point clouds: $(abspath $(POINTS))/object.ply and scene.ply',)
	$(PRUNE)

# After changing MASKS: rebuild the mask-dependent depth and meshes (needs KEEP_INTERMEDIATE=1 runs).
rebuild-object:
	rm -rf $(HYBRID_DEPTH) $(FUSED_DEPTH) $(REFINED_FUSED_DEPTH) $(CONSISTENT_DEPTH) $(DA3_HEAD) $(ADAPTED_DA3_DEPTH) $(ADAPTED_HYBRID_DEPTH) $(ADAPTED_REFINED_DEPTH) $(GEOMETRIC_DEPTH) $(GAUSSIAN_DEPTH)
	rm -f $(SCENE_MESH) $(OBJECT_MESH) $(CLEAN_MESH) $(FINAL_MESH) $(FINAL_SCENE) $(SCENE_POINTS) $(OBJECT_POINTS) $(METRICS)/object.json
	$(MAKE) reconstruct SCENE=$(SCENE) RUN=$(RUN) MASKS=$(MASKS)

$(GAUSSIAN_DEPTH)/hybrid_depth.json: $(OBJECT_POINTS) $(CONSISTENT_DEPTH)/consistency.json recon/depth/gaussian_refine.py
	@$(call stage,Planar-Gaussian refinement)
	$(Q)$(GPU) $(GAUSSIAN_PYTHON) -m recon.depth.gaussian_refine $(SEQ) --model $(MODEL) \
	  --init-points $(OBJECT_POINTS) $(SCENE_POINTS) --init-depth $(CONSISTENT_DEPTH) --init-opacity 0.3 \
	  --output-dir $(GAUSSIAN_DEPTH) --export-gaussians $(GAUSSIAN_PLY) $(GAUSSIAN_ARGS)

# Second pass: optimise, then rebuild every depth-derived output from the optimised depth.
refine-gaussian: $(GAUSSIAN_DEPTH)/hybrid_depth.json
	rm -rf $(CONSISTENT_DEPTH)
	rm -f $(SCENE_MESH) $(OBJECT_MESH) $(CLEAN_MESH) $(FINAL_MESH) $(FINAL_SCENE) $(SCENE_POINTS) $(OBJECT_POINTS) $(METRICS)/object.json
	$(MAKE) reconstruct SCENE=$(SCENE) RUN=$(RUN) MASKS=$(MASKS) OBJECT_DEPTH_INPUT=$(abspath $(GAUSSIAN_DEPTH))

# The refinement stage writes the PLY beside its depth. It is deliberately *not* derived from that rule
# here: refine-gaussian rebuilds the depth its own stage depends on, so making this a derived target had
# `make asset` re-run the whole optimiser (a second 90 s pass) to refresh a file that was already correct.
$(GAUSSIAN_PLY):
	@test -f $@ || { echo 'No $@; run make refine-gaussian first (it writes the Gaussians)'; exit 2; }

# The fitted Gaussians rendered from cameras facing the mesh, for texels the photos never see squarely.
$(SPLAT_VIEWS)/trajectory.csv: $(GAUSSIAN_PLY) $(FINAL_MESH) recon/surface/splat_views.py
	@$(call stage,Splat views for texturing)
	$(Q)$(GPU) $(GAUSSIAN_PYTHON) -m recon.surface.splat_views $(GAUSSIAN_PLY) --mesh $(FINAL_MESH) \
	  --output $(SPLAT_VIEWS) --trajectory $(TRAJECTORY) $(SPLAT_VIEWS_ARGS)

$(WATERTIGHT_MESH): $(FINAL_MESH) recon/surface/watertight.py
	@$(call stage,Watertight closure)
	$(Q)$(GPU) $(PYTHON) -m recon.surface.watertight $(SEQ) --mesh $(FINAL_MESH) --model $(MODEL) \
	  --trajectory $(TRAJECTORY) --output $@ $(WATERTIGHT_ARGS)

watertight: $(WATERTIGHT_MESH)
	@echo 'Watertight:   $(abspath $(WATERTIGHT_MESH))'

# A closed, textured asset: ASSET_MESH=$(FINAL_MESH) textures the open mesh instead, for a scene whose
# closure is refused.
# Specular-free photographs: each frame's highlights replaced by the other frames' consensus colour, aligned by
# the cross-view-checked depth (recon.depth.fusion.highlights). HIGHLIGHT_FREE=1 textures the asset from them.
HIGHLIGHTS := $(WORK)/highlights
HIGHLIGHT_FREE ?=
$(HIGHLIGHTS)/highlights.json: $(CONSISTENT_DEPTH)/consistency.json recon/depth/fusion/highlights.py
	@$(call stage,Highlight removal)
	$(Q)$(GPU) $(PYTHON) -m recon.depth.fusion.highlights $(SEQ) --model $(MODEL) --depth-dir $(CONSISTENT_DEPTH) \
	  $(if $(TEXTURE_IMAGES),--images $(TEXTURE_IMAGES),) --output-dir $(HIGHLIGHTS)

highlights: $(HIGHLIGHTS)/highlights.json
	@echo 'Specular-free: $(abspath $(HIGHLIGHTS))'

ASSET_IMAGES = $(if $(HIGHLIGHT_FREE),$(HIGHLIGHTS),$(TEXTURE_IMAGES))
$(ASSET_GLB): $(ASSET_MESH) $(SPLAT_VIEWS)/trajectory.csv recon/surface/texture.py $(if $(HIGHLIGHT_FREE),$(HIGHLIGHTS)/highlights.json,)
	@$(call stage,Textured asset)
	$(Q)$(PYTHON) -m recon.surface.texture $(ASSET_MESH) --model $(MODEL) --trajectory $(TRAJECTORY) \
	  --images $(ASSET_IMAGES) --sequence $(SEQ) --fill-views $(SPLAT_VIEWS) \
	  --max-faces $(ASSET_MAX_FACES) --output $@

asset: $(ASSET_GLB)
	@echo 'Asset:        $(abspath $(ASSET_GLB))'

# Scene-first: reconstruct with no masks (white ones, OBJECT_TSDF=scene), then cut the object out of the
# scene mesh with the real masks as supervision for a closed-form per-Gaussian labelling
# (recon.surface.segment). SEGMENT_MASKS must be the real object masks, not the sequence's own.
SEGMENT_MASKS ?= $(MASKS)
SEGMENT_ARGS ?=
SEGMENTED_MESH := $(MESHES)/object_segmented.ply
$(SEGMENTED_MESH): $(FINAL_MESH) $(GAUSSIAN_PLY) recon/surface/segment.py
	@$(call stage,3D segmentation)
	$(Q)$(GPU) $(GAUSSIAN_PYTHON) -m recon.surface.segment $(SEQ) --model $(MODEL) --gaussians $(GAUSSIAN_PLY) \
	  --mesh $(FINAL_MESH) --masks $(SEGMENT_MASKS) --output $@ $(SEGMENT_ARGS)

segment: $(SEGMENTED_MESH)
	@echo 'Segmented:    $(abspath $(SEGMENTED_MESH))'

setup-gaussian:
	python3 -m venv $(HOME)/.venvs/me566-pgsr
	$(HOME)/.venvs/me566-pgsr/bin/pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
	$(HOME)/.venvs/me566-pgsr/bin/pip install numpy opencv-python-headless open3d plyfile
	CUDA_HOME=/usr/local/cuda-12.9 TORCH_CUDA_ARCH_LIST=8.9 \
	  NVCC_APPEND_FLAGS="-include cfloat -include cstdint" CXXFLAGS="-include cstdint" \
	  $(HOME)/.venvs/me566-pgsr/bin/pip install --no-build-isolation gsplat==1.5.3

evaluate: $(FINAL_MESH)
	@$(call stage,Scoring mesh)
	$(Q)$(PYTHON) -m recon.evaluation.metrics.mesh $(FINAL_MESH) --reference $(DATA)/$(SCENE)/mesh/gt_mesh.ply --trajectory $(TRAJECTORY) --pose-dir $(DATA)/$(SCENE)/pose --output $(METRICS)/object.json
	@$(call stage,MobileBrick official score)
	$(Q)$(PYTHON) -m recon.evaluation.metrics.mobilebrick $(FINAL_MESH) --scene $(DATA)/$(SCENE) --trajectory $(TRAJECTORY) --output $(METRICS)/object_official.json > /dev/null
	@$(call stage,Scoring poses)
	$(Q)$(PYTHON) -m recon.evaluation.metrics.poses $(TRAJECTORY) $(DATA)/$(SCENE)/pose --output $(METRICS)/poses.json
	$(Q)$(PYTHON) -m recon.evaluation.metrics.pose_auc $(TRAJECTORY) $(DATA)/$(SCENE)/pose --output $(METRICS)/pose_auc.json > /dev/null
	@$(call stage,Reprojection check)
	$(Q)$(PYTHON) -m recon.evaluation.metrics.reprojection $(MODEL)
	$(PRUNE)

refine-masks:
	@test -n "$(TRACKED_MASKS)" || { echo 'Set TRACKED_MASKS=/path/to/tracked/masks'; exit 1; }
	$(PYTHON) -m recon.masks.sam_hq $(SEQ) --input-masks $(TRACKED_MASKS) --output-masks $(SEQ)/sam_hq/masks --checkpoint runs/models/sam_hq/sam_hq_vit_b.pth

# The default pipeline on every scene with supplied masks; one log per scene under
# BENCH_ROOT. DTU (DATA=runs/real/dtu_dataset ...) is scored by chamfer.
BENCH_JOBS ?= 2
# BENCH_SCORES_ONLY=1 builds only what scoring needs (skips texture, point clouds and the scene mesh).
BENCH_SCORES_ONLY ?=
# Live rich dashboard; each scene's full output goes to BENCH_ROOT/<scene>.log.
# Reconstruct every scene four times (ours, truth poses, truth depth, both) to rank stages.
stage-ranking:
	$(Q)$(PYTHON) -m recon.evaluation.diagnostics.stage_ranking --dataset $(if $(IS_DTU),dtu,mobilebrick) --root $(RANK_ROOT) --jobs $(BENCH_JOBS) --colmap $(COLMAP) --scenes $(BENCH_SCENES)

benchmark:
	$(Q)$(PYTHON) -m recon.evaluation.benchmark.run DATA=$(DATA) DEPTH_MODEL=$(DEPTH_MODEL) $(if $(OMEGA),OMEGA=1,) OMEGA_CHECKPOINT=$(OMEGA_CHECKPOINT) OMEGA_RES=$(OMEGA_RES) OMEGA_WINDOW=$(OMEGA_WINDOW) OMEGA_STEP=$(OMEGA_STEP) OMEGA_RESIZE_MODE=$(OMEGA_RESIZE_MODE) --root $(BENCH_ROOT) \
	  --seq-root $(SEQ_ROOT) --jobs $(BENCH_JOBS) $(if $(IS_DTU),--dtu,) $(if $(BENCH_SCORES_ONLY),--scores-only,) --scenes $(BENCH_SCENES)

# DTU chamfer (mm) of EVAL_MESH against the scan's laser reference.
EVAL_MESH ?= $(FINAL_MESH)
evaluate-dtu: $(EVAL_MESH)
	@$(call stage,DTU chamfer)
	$(Q)$(PYTHON) -m recon.evaluation.metrics.dtu $(EVAL_MESH) --reference $(DATA)/$(SCENE)/reference --trajectory $(TRAJECTORY) --pose-dir $(DATA)/$(SCENE)/pose --output $(METRICS)/$(notdir $(EVAL_MESH:.ply=))_dtu.json
	$(Q)$(PYTHON) -m recon.evaluation.metrics.pose_auc $(TRAJECTORY) $(DATA)/$(SCENE)/pose --output $(METRICS)/pose_auc.json > /dev/null
	$(PRUNE)

# Results report from the finished runs (published comparisons, latency, ablations): docs/REPORT.md.
report:
	$(PYTHON) -m recon.evaluation.benchmark.report

# Open3D's native viewer for a mesh, point cloud or textured .glb: make view_ply PATH=file (or PLY=file).
# A PATH= on the command line replaces the shell's search path, so the recipe restores the one
# make was started with (captured while parsing, before recipes see the override).
HOST_PATH := $(shell printenv PATH)
VIEW_PLY = $(or $(PLY),$(if $(filter command line,$(origin PATH)),$(PATH),))
view_ply:
	@test -n "$(VIEW_PLY)" || { echo 'Use make view_ply PATH=path/to/file.ply (or PLY=...)'; exit 2; }
	@/usr/bin/env PATH='$(HOST_PATH)' $(PYTHON) -m recon.apps.view_ply '$(VIEW_PLY)'

test:
	$(PYTHON) -m pytest -q tests

status:
	@echo 'Scene: $(SEQ)'
	@echo 'Masks: $(MASKS)'
	@echo 'Mesh: $(abspath $(FINAL_MESH))'
	@test -f $(FINAL_MESH) && ls -lh $(FINAL_MESH) || true

ifneq ($(SCAN_GOAL),)
.PHONY: $(SCAN_GOAL)
$(SCAN_GOAL):
	@:
endif

construct:
	@test -e "$(SCAN)" || { echo 'Use make construct path/to/images_or_video.mp4 or make construct SCAN=path'; exit 2; }
	$(PYTHON) -m recon.apps.construct "$(SCAN)" --output "$(SCAN_OUTPUT)" --max-edge $(SCAN_MAX_EDGE) --sample-fps $(SCAN_FPS) $(if $(SCAN_MASKS),--masks "$(SCAN_MASKS)",) $(if $(SCAN_MASK_COMMAND),--mask-command '$(SCAN_MASK_COMMAND)',) $(if $(SCAN_COMPLETE_MASKS),--complete-masks,) $(if $(SCAN_REFINE_MASKS),--refine-masks,) --mask-dilate $(SCAN_MASK_DILATE) $(if $(SCAN_SCENE),--scene,)
