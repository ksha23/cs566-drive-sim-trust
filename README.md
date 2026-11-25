# How far off the recorded path can a Gaussian-splat driving simulator be trusted?

CS 566 (Computer Vision, UW–Madison, Fall 2026) course project of Kyle Sha and Allen Xia.
Project page: <https://ksha23.github.io/cs566-drive-sim/>.
Results: [`results/RESULTS.md`](results/RESULTS.md) (all sections) and [`results/final/`](results/final/) (final
figures, tables and CSVs). Implementation notes and paper-vs-ours decisions: [`docs/`](docs/).

A driving simulator built from a recorded drive reconstructs the road as a 3D Gaussian splat and renders what the car's
cameras would see if it drove *differently*. We measure how fast that rendering stops being trustworthy as the camera
moves sideways, turns or changes height, against **real** images: Para-Lane scans the same roads in three parallel
lanes (median lateral offset per lane pair 2.3–4.1 m to the neighboring lane, 5.3–7.8 m to the far one), so a splat built from one lane can be checked against the real images of the
other lanes. We compare generative "fixers" (released Difix, NVIDIA Fixer, our reimplementation of the Difix training
recipe, a fine-tune of the released Difix, a small U-Net), the Difix3D+ progressive 3D update and a LiDAR road-surface
prior, and we count hallucinations (invented or erased lane paint, fake vehicles), not only image similarity.

| | Paper | Role here |
|---|---|---|
| [1] | Difix3D+ (CVPR 2025) | the method we reimplement (fixer training recipe from SD-Turbo; progressive 3D update); released Difix / Difix-ref as baselines |
| [2] | FaithFusion (CVPR 2026) | comparison method: EIG-weighted progressive update; does its map predict hallucinated pixels? |
| [3] | HUGSIM (TPAMI 2026) | splat-based simulator for the closed-loop test (released nuScenes scenes, LTF policy, HD-Score) |
| [4] | NVIDIA Fixer (2025) | driving-trained fixer, frozen baseline (+ 24 GB fine-tune pilot) |
| [5] | ReconDreamer (CVPR 2025) | NTA-IoU / NTL-IoU metrics (re-implemented) |

## Quick check (no GPU, no datasets; a few minutes)
Trained checkpoints are not released (`THIRD_PARTY.md`); everything beyond this check needs the datasets and retraining.
The per-image results of every experiment ship in `results/`, so the final tables can be rebuilt on any machine, and
one bundled Para-Lane frame (`examples/scene_17_frame`) can be scored with the same metrics:
```bash
python3 -m venv .venv && .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements-cpu.txt
PYTHON=.venv/bin/python bash scripts/quick_check.sh
```
Step 1 regenerates `results/final/` (tables, robustness checks) and reports whether every table, CSV and the headline numbers
(`key_numbers.json`) match the shipped ones; step 2 prints PSNR, LPIPS, lane-paint precision and recall and the spurious-vehicle count for the raw
render and two repaired versions of the example frame (our re-implementation invents a marking there: paint
precision 0.13 against 0.95 for the raw render). Tested on a clean Ubuntu 24.04 machine with Python 3.12
(`requirements-cpu.txt` lists the versions). The installation takes a few minutes; the first run downloads the
LPIPS (AlexNet) and YOLO11m weights, about 0.3 GB, and the check itself then runs in under a minute.

## Hardware used
Everything except part of stage 22 ran on one workstation (called sbel in logs and notes): 1x NVIDIA RTX 3090 (24 GB, driver 595, CUDA 12.x runtime), AMD Ryzen 7 9700X
(8 cores / 16 threads), 32 GB RAM, Ubuntu 22.04, ~0.5 TB of disk for data and outputs. The GPU was shared with other
projects through a lock wrapper (`CS566D_GPUJOB`, default `~/cs566/gpujob.sh`, which waits until the GPU has no
compute processes and then runs the command); every GPU stage below is written as `$GJ <command>`. If that file does
not exist and `CS566D_GPUJOB` is unset, `$GJ` runs the command directly. In stage 22, the heading / height fixers on
three roads and the FaithFusion updates on four roads, and their metrics and lane detection, ran as preemptible Slurm
array tasks on a university cluster (one GPU per task: H100, RTX A4500 or RTX 4000 Ada; `scripts/euler/`, resumable
after requeue). Outputs and results were copied back to the workstation; the cluster's CLRNet build reproduces the
workstation's lane detections to within 0.1 px.

## Repository layout
```
src/common/config.py        all machine paths (env CS566D_DATA, CS566D_SMOKE)
src/splats/                 Para-Lane loader, gsplat street splat (LiDAR init/depth, sky env map), fitting, road-surface prior
src/pairs/                  training/eval pair curation: cross-lane, under-fit, cycle reconstruction, on-path identity
src/fixers/                 registry.py (one interface to all fixers), apply.py (run fixers on eval renders),
                            unet/ (our small fixer), difix_ft/ (fine-tune of the released Difix), difix_reimpl/ (our Difix
                            training reimplementation from SD-Turbo), nv_fixer/ (NVIDIA Fixer wrapper + TE shim)
src/progressive/            Difix3D+ progressive 3D update (+ faithfusion/: EIG-weighted variant)
src/eval/                   metrics, paint / YOLO / CLRNet-lane evaluation, bootstrap summaries, figures, heading/height axis,
                            final_report.py (results/final)
src/hugsim/                 HUGSIM offset renders, NTA-IoU, closed loop with a fixer between simulator and policy
src/euvs/ src/mv2/ src/dl3dv/   cross-dataset (EUVS, MV2) and Difix-reimplementation validation (DL3DV) pipelines
src/viz/                    demo videos, qualitative grids
src/tools/                  command-line entry points (below)
scripts/NN_*.sh             pipeline stages in order;  scripts/lib/  env + helpers
configs/                    Para-Lane scene -> archive-part index
envs/                       environment specs (pip freezes, pixi manifests)
results/                    small JSON/CSV + figures; every number in RESULTS.md comes from here
```

## Environment setup
Two roots live outside the repository (set them in your shell; defaults in `scripts/lib/env.sh`):
`CS566D_DATA` (data and all outputs, default `~/cs566/data/D`) and `CS566D_SMOKE` (external repos, envs, weights,
default `~/cs566/smoke/D`). Exact package lists: [`docs/ENVIRONMENTS.md`](docs/ENVIRONMENTS.md) and `envs/`.

```bash
export CS566D_DATA=~/cs566/data/D CS566D_SMOKE=~/cs566/smoke/D; mkdir -p $CS566D_DATA $CS566D_SMOKE/{envs,weights}

# 1. project env (Python 3.10, torch 2.4.1 + CUDA 12.1, gsplat 1.4, diffusers 0.25.1, ...); scripts/lib/py.sh uses it
conda create -y -p $CS566D_SMOKE/envs/difix python=3.10
$CS566D_SMOKE/envs/difix/bin/pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
$CS566D_SMOKE/envs/difix/bin/pip install -r envs/difix_env.txt   # exact freeze (gsplat wheel: pt24cu121)
git clone https://github.com/nv-tlabs/Difix3D $CS566D_SMOKE/Difix3D && git -C $CS566D_SMOKE/Difix3D checkout c76edc5
$CS566D_SMOKE/envs/difix/bin/python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')" && mv yolo11m.pt $CS566D_SMOKE/weights/

# 2. CLRNet lane detector: venv over the project env (envs/lanes_env.txt), CULane DLA34 weights
git clone https://github.com/Turoad/CLRNet $CS566D_DATA/lanes/CLRNet && git -C $CS566D_DATA/lanes/CLRNet checkout 7269e9d
$CS566D_SMOKE/envs/difix/bin/python -m venv --system-site-packages $CS566D_DATA/lanes/env
$CS566D_DATA/lanes/env/bin/pip install mmcv==1.7.2 addict yapf==0.32.0   # then build CLRNet's NMS op for your GPU (nvcc)
#    weights: CLRNet release "culane_dla34.pth" -> $CS566D_DATA/lanes/weights/

# 3. our Difix reimplementation: venv over the project env + tensorboard / torch-fidelity
$CS566D_SMOKE/envs/difix/bin/python -m venv --system-site-packages $CS566D_DATA/difix_reimpl/venv
$CS566D_DATA/difix_reimpl/venv/bin/pip install -r envs/difix_reimpl_venv_local.txt

# 4. NVIDIA Fixer env (Python 3.12, torch 2.6 cu126, cosmos-predict2; no Docker needed; downloads nvidia/Fixer)
bash src/fixers/nv_fixer/setup_env.sh

# 5. HUGSIM simulator + LTF policy client (pixi; only for stages 10-13)
git clone https://github.com/hyzhou404/HUGSIM $CS566D_SMOKE/HUGSIM && git -C $CS566D_SMOKE/HUGSIM checkout 62c690d
cp envs/hugsim_pixi.toml $CS566D_SMOKE/HUGSIM/pixi.toml && (cd $CS566D_SMOKE/HUGSIM && pixi install)
git clone https://github.com/hyzhou404/NAVSIM $CS566D_DATA/hugsim/navsim_client && git -C $CS566D_DATA/hugsim/navsim_client checkout ca0ca7e
cp envs/navsim_ltf_pixi.toml $CS566D_DATA/hugsim/navsim_client/pixi.toml && (cd $CS566D_DATA/hugsim/navsim_client && pixi install)

# 6. FaithFusion rasterizer (only for stage 14)
bash scripts/14_faithfusion.sh setup
```
Hugging Face weights (`nvidia/difix`, `nvidia/difix_ref`, `stabilityai/sd-turbo`, `nvidia/Fixer`) are fetched on first
use into `$CS566D_SMOKE/hf`; accept their licenses on the model pages first.

## Data (not included; download it yourself)
| Dataset | Where | License | Download |
|---|---|---|---|
| Para-Lane 0315 (main benchmark) | ModelScope [`nizqleo/Paralane`](https://www.modelscope.cn/datasets/nizqleo/Paralane) | CC BY-NC 4.0 | `scripts/01_download_paralane.sh <scenes>` (fetches only the archive parts of those scenes) |
| HUGSIM released scenes, scenarios, map cache, 3DRealCar | HF [`XDimLab/HUGSIM`](https://huggingface.co/datasets/XDimLab/HUGSIM) | MIT; nuScenes-derived parts under the nuScenes terms (CC BY-NC-SA 4.0) | see `src/hugsim/README.md` |
| LTF weights | HF [`autonomousvision/navsim_baselines`](https://huggingface.co/autonomousvision/navsim_baselines) | Apache-2.0 | `ltf/ltf_seed_{0,1,2}.ckpt` |
| EUVS benchmark (MARS scenes) | HF [`ai4ce/EUVS-Benchmark`](https://huggingface.co/datasets/ai4ce/EUVS-Benchmark) (gated) | CC BY-NC-SA 4.0 | `scripts/16_euvs.sh download` |
| MV2 | HF [`sanjay810/mv2-dataset`](https://huggingface.co/datasets/sanjay810/mv2-dataset) | academic research use only | `scripts/17_mv2.sh download` |
| DL3DV-10K sample | HF [`DL3DV/DL3DV-10K-Sample`](https://huggingface.co/datasets/DL3DV/DL3DV-10K-Sample) (gated; request access) | DL3DV terms of use | `scripts/19_dl3dv.sh download` |

Gated downloads read your own Hugging Face login (`huggingface-cli login`); no token is stored in this repository.
Nothing from these datasets is redistributed here except figures/videos derived from Para-Lane, nuScenes (HUGSIM)
and MARS, with attribution; see [`THIRD_PARTY.md`](THIRD_PARTY.md) and [`docs/DATA_AND_LICENSES.md`](docs/DATA_AND_LICENSES.md).

## How to reproduce (stage by stage)
`GJ=${CS566D_GPUJOB:-~/cs566/gpujob.sh}` (or `env`, i.e. run directly, when that file does not exist; `scripts/lib/env.sh`). Times are wall-clock on the RTX 3090. Every stage is resumable (skips
finished outputs). Data products go to `$CS566D_DATA`; the small result files go to `results/`.

| Stage | Command | Time | Output |
|---|---|---|---|
| 1 data | `scripts/01_download_paralane.sh scene_16 scene_17 scene_19 scene_20 scene_21 scene_22 scene_23 scene_24`; for stage 18 also `scene_3 scene_4 scene_10 scene_11 scene_14 scene_15` (`docs/more_scenes.md`) | network-bound (~3–5 GB/scene) | `$DATA/paralane/extracted/0315/` |
| 2 splats + pairs | `for s in scene_20 scene_21 scene_22 scene_23 scene_24; do $GJ scripts/02_scene.sh $s train; done`; `for s in scene_19 scene_16 scene_17; do $GJ scripts/02_scene.sh $s eval; done` | 3 lane splats (4–6 min each) + pairs: ~30 min per eval scene, ~60 min per train scene | `$DATA/splats`, `$DATA/pairs` |
| 3 splat quality | `scripts/03_splat_quality.sh` | minutes (CPU) | `results/onpath`, `results/pairs` |
| 4 our U-Net | `$GJ scripts/04_train_unet.sh`; `$GJ scripts/04b_train_unet_gan.sh` | ~20–30 min per run (6 runs) | `$DATA/fixers/<run>`, `results/training` |
| 5 Difix-FT | `$GJ scripts/05_finetune_difix.sh` | ~45 min | `$DATA/difix_ft`, `results/training/difix_ft.json` |
| 5b Difix reimplementation | `bash src/fixers/difix_reimpl/jobs.sh ours` (also `official`, `ours448`) | ~25–50 min per run | `$DATA/difix_reimpl`, `results/difix_reimpl` |
| 6 fixers + metrics | `$GJ scripts/06_eval_fixers.sh` (`06b` extra U-Nets) | ~1–2 h per test scene for all fixers | `$DATA/fixed`, `results/eval`, `results/lanes` |
| 7 progressive update | `$GJ scripts/07_progressive.sh` (`07b` paper-like schedule, `07c` NVIDIA Fixer, `07d` our reimplementation) | ~20 min per scene and restorer | `$DATA/splats/<scene>/<clip>/prog_*` |
| 8 road prior | `$GJ scripts/08_road_prior.sh` | ~30 min per scene | `splat_road.pt`, `results/eval` |
| 9 tables / figures | `scripts/09_summarize_figures.sh video` | minutes (CPU) | `results/eval/tables.md`, `results/figures` |
| 10–13 HUGSIM | `$GJ scripts/10_hugsim_render.sh`; `$GJ scripts/11_hugsim_nta.sh ...`; `SEEDS="0 1 2" $GJ scripts/12_hugsim_closed_loop.sh <fixer> "<scenarios>"`; `scripts/13_hugsim_reproduction.sh` (details: `src/hugsim/README.md`) | closed loop: ~2 min per scenario without a fixer, ~2–3 min with one; 25 scenarios per seed | `$DATA/hugsim`, `results/hugsim` |
| 14 FaithFusion | `scripts/14_faithfusion.sh setup|maps|auroc|prog|eval` | ~1 h | `results/faithfusion` |
| 15 heading / height | `scripts/15_heading_height.sh render`; `... fixall`; `... lanes|eval <scene>`; `... summary` | ~3 h | `results/heading` |
| 16 EUVS | `scripts/16_euvs.sh download`; `... fitall`; `... fixall`; `... lanes|eval L3|rot60`; `... summary` | ~3 h | `results/euvs` |
| 17 MV2 | `scripts/17_mv2.sh download`; `... masks|geom <seqs>`; `... fitall`; `... fixall`; `... lanes|eval <dir>`; `... summary` | ~4 h | `results/mv2` |
| 18 more test roads | `scripts/18_more_scenes.sh screen <scenes>`; for each new scene `... fit|fix|eval|prog|road <scene>`; `... summary` | ~1.5–2.5 h per scene | `results/eval_more` |
| 19 DL3DV validation | `scripts/19_dl3dv.sh download`; `... fit 30|60`; `... json`; `... ours448|official448|oursref`; `... eval released|trained|metrics` | ~4 h | `results/dl3dv` |
| 20 videos | `$GJ scripts/20_videos.sh sweeps`; `scripts/20_videos.sh make` | ~1 h GPU + minutes | `results/videos` (index.json) |
| 21 our own renderer | `$GJ scripts/21_own_renderer.sh check|shortfit|refit|render|eval` (hybrid renderer vs gsplat, refit of scene_16/clip_0; `docs/own_renderer.md`) | ~1.5 h GPU | `results/own_renderer` |
| 22 final runs | `bash scripts/22_final_runs.sh ref heading ff closed` (released Difix-ref on four more roads; heading / height and FaithFusion on six more roads; our re-implementation in the closed loop); `... collect` after cluster steps (`scripts/euler/`; `setup_lanes.sbatch` and `score.sbatch` run the collect scoring there instead) | ~25 h GPU in total | `$DATA/fixed`, `$DATA/heading`, `results/eval_more`, `results/heading`, `results/faithfusion`, `results/hugsim` |
| 23 regenerate | `bash scripts/23_regenerate.sh` (every summary, table and figure, and `results/final/key_numbers.json`) | ~30 min (CPU) | `results/final/` and the summaries above |
| final | `CUDA_VISIBLE_DEVICES= scripts/lib/py.sh src/viz/qualitative.py` (qualitative grids) | minutes (CPU) | `results/final/` |

Split: train = Para-Lane 0315 scene_20–24, val = scene_19 (checkpoint and config selection only), test = scene_16
and scene_17, plus held-out test roads added later with the same pipeline (stage 18; `docs/more_scenes.md`). No test road was used for training. One design choice was compared on a test road: the two-round progressive schedule,
chosen for GPU time, also gave higher lane F1 on scene_16 than a four-step schedule closer to the paper's (0.90 vs.
0.77; RESULTS §6), so scene_16's progressive result is not strictly held out.

## Executables (command-line entry points)
All run in the project env through `scripts/lib/py.sh` (which puts `src/` on `PYTHONPATH`).

```bash
# Render a fitted splat off the recorded path: every 4th pose of the lane, 3.3 m to the right (+ = left), 5 deg yaw, +0.5 m
scripts/lib/py.sh src/tools/render_offpath.py scene_16 clip_0 $CS566D_DATA/splats/scene_16/clip_0/splat.pt /tmp/off \
    --lateral -3.3 --yaw 5 --dz 0.5 --every 4

# Apply a fixer to every image of a folder (difix | difix_ref | difix_ft | difix_reimpl | nv_fixer | ours)
scripts/lib/py.sh src/tools/fix_folder.py /tmp/off /tmp/off_difix --method difix
scripts/lib/py.sh src/tools/fix_folder.py /tmp/off /tmp/off_ft --method difix_ft --ckpt $CS566D_DATA/difix_ft/best_unet.pt

# Trust metrics of one render against the real image at the same pose (PSNR/SSIM/LPIPS, road LPIPS,
# lane-paint precision/recall/F1, texture ratio, YOLO11 fake vehicles if a movable-object mask is given)
scripts/lib/py.sh src/tools/pair_metrics.py render.png real.png --fg movable_mask.png [--cpu]

# Fit one lane of a Para-Lane scene as a splat (12k iterations, LiDAR init + depth, sky map)
scripts/lib/py.sh src/splats/fit_splat.py scene_16 clip_0 $CS566D_DATA/splats/scene_16/clip_0 --iters 12000

# Batch versions used for the results: fixers on a scene's eval renders; metrics; CLRNet lanes
scripts/lib/py.sh src/fixers/apply.py $CS566D_DATA/pairs scene_16 $CS566D_DATA/fixed --methods difix,nv_fixer
scripts/lib/py.sh src/eval/evaluate.py $CS566D_DATA/pairs scene_16 $CS566D_DATA/fixed out.json --methods raw,difix
scripts/lib/lanes_scene.sh scene_16 $CS566D_DATA/fixed out_lanes.json raw difix
```
CLRNet lane F1 needs the separate lanes env (setup step 2) and is computed by `scripts/lib/lanes_scene.sh`.

## Licenses
Our code: MIT ([`LICENSE`](LICENSE)). Third-party code, weights and data keep their own licenses
([`THIRD_PARTY.md`](THIRD_PARTY.md)); none of the weights or datasets are redistributed here.
