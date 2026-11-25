# Validating our Difix re-implementation

Our trainer: `src/fixers/difix_reimpl/` (see `docs/IMPLEMENTATION_NOTES_difix.md`). Official trainer:
`nv-tlabs/Difix3D` `src/train_difix.py`, run from a copy with the minimal patches listed below.
Released weights: `nvidia/difix`, `nvidia/difix_ref`. Everything was trained from SD-Turbo on the same data.

## 1. DL3DV (the paper's data): sample subset, run Sep 28, 2026 — `results/dl3dv/`, `scripts/19_dl3dv.sh`
**Access.** The student's DL3DV request was approved for **`DL3DV/DL3DV-10K-Sample` only**. `DL3DV-Benchmark` (the
paper's 140 evaluation scenes) and `DL3DV-ALL-960P` still return HTTP 403 "not in the authorized list" for the same
login (probe of one file per repo, `results/dl3dv/access_check.json`). History: until Sep 28 every DL3DV file returned
401 (invalid token on sbel), then 403 for all three repos (valid token, request pending). So this validation uses the
11 sample scenes (indoor/outdoor handheld videos, 142–399 frames, `images_4` = 960x540, COLMAP OPENCV model).

**Pairs, as Difix3D+ curates DL3DV** (`src/dl3dv/fit_pairs.py`; the paper's "sparse reconstruction" and "model
underfitting"): the images are undistorted to a pinhole camera; a splat (same gsplat model as our EUVS splats,
SfM-point init, 6k iterations) is fitted on every S-th frame only and rendered at all other frames ("sparse" pairs,
render vs. real); its 25 %-of-schedule snapshot is rendered at every frame ("underfit" pairs). Two sparsity levels:
S = 30 (held-out PSNR 14.6–25.9 dB per scene) and S = 60 (12.8–22.1 dB) (`results/dl3dv/split.json`). Reference view =
nearest training frame. S = 10 was tried first and discarded: 30.2 dB held-out, too little to fix.

**Split** (by scene, `src/dl3dv/build_json.py`): TRAIN 7 scenes (9,248 pairs: 4,562 sparse + 4,686 underfit, both
levels), VAL 1 scene (dafa9c7c; 339 pairs; monitoring and our ref model's checkpoint choice only), TEST 3 scenes
(5c3af581, 9c8c0e0f, cc08c0bd; every 3rd held-out frame: 333 pairs at S = 30, 339 at S = 60). A first version of the
split also listed the VAL scene under TEST; that was fixed before any evaluation (training pairs unchanged).

**Runs** (from SD-Turbo, same pairs, same settings as section 2: lr 2e-5, 100-step warm-up, batch 1, L2 + LPIPS +
0.5 Gram, τ = 200, 1,801 steps):
* `official448`: official trainer (patched copy), no reference view, 448x768 (it cannot fit the reference view on
  24 GB, section 2); stopped after its step-1801 checkpoint (18 min 47 s).
* `ours448`: our trainer, same setting (25 min, 12.8 GiB); last checkpoint, like the official run.
* `oursref`: our trainer with the reference view at 576x1024 (50 min, 15.3 GiB); checkpoint chosen by VAL LPIPS
  (step 1,200 of 1,801; VAL LPIPS 0.295 vs 0.298 at the end).
* released `nvidia/difix` and `nvidia/difix_ref` (trained by NVIDIA on the full DL3DV at a far larger budget).

**Results on the held-out TEST scenes** (`results/dl3dv/summary.md`; mean [95 % bootstrap CI over images]; FID over
the set of ~335 images, so noisy):

| S | model | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ |
|---|---|---|---|---|---|
| 30 | raw sparse-view render | 21.87 | 0.698 | 0.195 [0.182, 0.208] | 37.7 |
| 30 | released Difix | 21.18 | 0.648 | 0.177 [0.168, 0.187] | 24.4 |
| 30 | released Difix-ref | 21.77 | 0.676 | **0.163** [0.153, 0.173] | **18.9** |
| 30 | official trainer, matched | 19.91 | 0.583 | 0.237 [0.227, 0.247] | 36.3 |
| 30 | our trainer, matched | 21.20 | 0.653 | 0.218 [0.208, 0.229] | 26.1 |
| 30 | our trainer, ref view | **22.08** | 0.675 | 0.167 [0.157, 0.177] | 21.2 |
| 60 | raw sparse-view render | 19.25 | 0.602 | 0.267 [0.251, 0.283] | 64.4 |
| 60 | released Difix | 19.18 | 0.571 | 0.230 [0.219, 0.243] | 34.6 |
| 60 | released Difix-ref | 19.56 | 0.595 | **0.215** [0.204, 0.227] | **26.4** |
| 60 | official trainer, matched | 18.45 | 0.531 | 0.282 [0.269, 0.294] | 46.6 |
| 60 | our trainer, matched | 19.32 | 0.584 | 0.263 [0.250, 0.277] | 33.0 |
| 60 | our trainer, ref view | **19.99** | 0.604 | **0.215** [0.202, 0.227] | 28.0 |

Paired on the same images (`results/dl3dv/paired.csv`):
* **ours vs. official, matched**: ΔLPIPS −0.019 [−0.020, −0.017] at both levels, ΔPSNR +1.29 / +0.87 dB, FID 26.1 vs
  36.3 / 33.0 vs 46.6; ours is better on 94–96 % of images and on all 3 test scenes. This reproduces the driving-data
  result of section 2 on the paper's own kind of data.
* **our ref-view model vs. NVIDIA's released Difix-ref**: LPIPS +0.005 [+0.003, +0.006] (S = 30) and −0.000
  [−0.002, +0.001] (S = 60), i.e. a tie within 0.005; PSNR +0.32 / +0.43 dB for ours; FID worse (21.2 vs 18.9,
  28.0 vs 26.4). With 7 training scenes and 50 GPU-minutes it reaches the released model's perceptual quality on
  held-out DL3DV scenes, but not its distribution match (FID).
* **Caveat for the matched (448x768) runs**: both are *worse than the raw render* on LPIPS at S = 30, and ours is only
  on par with it at S = 60. Resizing alone costs this: the raw render resized 960x540 -> 768x448 -> 960x540 goes from
  LPIPS 0.195 to 0.243 (S = 30) and 0.267 to 0.311 (S = 60) (`results/dl3dv/resample_floor.json`). Against that floor
  both matched runs improve the image, ours by more. The comparison between the two trainers is fair (same size);
  the absolute numbers of the 448 runs are not a statement about Difix at full resolution.
* **What this still does not show**: the paper's own benchmark numbers (DL3DV-Benchmark, 28 held-out scenes) — that
  repo is not authorized for the account; and 3 test scenes are few (the per-scene LPIPS in `summary.md` shows one
  hard scene, 9c8c0e0f, dominating the mean).

**Earlier substitute (kept):** the same three-way comparison on our held-out driving data, below. VAL = 163 triplets
from scene_19 (all on-path held-out frames plus every 6th cross-lane frame), used for model selection. TEST = scenes
16 and 17, never used for selection.

## 2. Matched comparison: our trainer vs. the official trainer
Same `driving_pairs.json` (11,900 training triplets from scenes 20–24: cross-lane, under-fit, cycle and identity
pairs). Same settings:
* SD-Turbo init, τ = 200 (index 199);
* lr 2e-5 with 100-step warm-up then constant, batch 1;
* loss L2 + LPIPS + 0.5 Gram, no Gram warm-up;
* 1801 steps; last checkpoint used for both.

The official trainer with the reference view (`--mv_unet`, V = 2) **does not fit the 24 GB RTX 3090**:
* at its data loader's size of 576×1024 it runs out of memory at step 0;
* at 448×768 it runs out at step 2.

The reason is that it decodes both views without activation checkpointing and keeps bf16 AdamW states. The log
(`official_ref_oom.log`, not in the repo) has the CUDA OOM traces. The matched pair therefore runs **without the
reference view (V = 1) at 448×768**; 448 px is the smallest height that still fits the official 400×400 Gram crop.

| Run | Trainer | View | Resolution | Precision / optimizer | Speed | Peak memory |
|---|---|---|---|---|---|---|
| ours448 | ours | V = 1 | 448×768 | fp32 master, 8-bit AdamW, bf16 autocast | 0.82 s/step | 12.8 GiB |
| official448 | official (patched copy) | V = 1 | 448×768 | all weights bf16, AdamW | 0.62 s/step | ~17 GB (nvidia-smi) |
| ours_ref (our driving model) | ours | V = 2 | 576×1024 | as ours448 | 1.58 s/step | 15.3 GiB |

### VAL (scene_19, 163 triplets)
Metrics are measured at 960×540 against the real target image, with movable objects pasted from the target.
FID and KID come from torch-fidelity over the 163 images, so they are noisy at this set size.

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | KID ↓ |
|---|---|---|---|---|---|
| raw render | 18.17 | 0.483 | 0.550 | 103.3 | 0.091 |
| released Difix (`nvidia/difix`) | 17.77 | 0.414 | 0.482 | 68.5 | 0.052 |
| released Difix-ref (`nvidia/difix_ref`) | 17.94 | 0.449 | 0.325 | 46.0 | 0.022 |
| **official trainer, matched (V = 1, 448×768)** | 17.01 | 0.366 | 0.388 | 59.9 | 0.035 |
| **our trainer, matched (V = 1, 448×768)** | 17.82 | 0.418 | 0.299 | 52.8 | 0.027 |
| our trainer, ref view, 576×1024 (val-selected, step 1500) | 17.95 | 0.441 | **0.237** | 46.1 | 0.023 |

Source: `results/difix_reimpl/val_scene19.json`.

### TEST (scenes 16 + 17), mean over frames
Full table with 95% CIs from a cluster bootstrap over lane-pair sequences: `results/difix_reimpl/summary_test_compact.md`.

| Offset | Model | LPIPS ↓ | Lane-paint F1 ↑ | CLRNet NTL-IoU ↑ | Fake vehicles / image ↓ |
|---|---|---|---|---|---|
| ~3.6 m | raw | 0.524 | 0.521 | 0.070 | 0.043 |
| | official448 (matched) | 0.420 | 0.579 | 0.453 | 0.101 |
| | ours448 (matched) | 0.354 | 0.569 | 0.471 | 0.047 |
| | released Difix-ref (zero-shot) | 0.340 | 0.531 | 0.428 | 0.123 |
| | ours, ref, 576×1024 | **0.306** | 0.534 | **0.587** | 0.179 |
| ~7.1 m | raw | 0.664 | 0.156 | 0.016 | 0.223 |
| | official448 (matched) | 0.507 | 0.187 | 0.209 | 0.316 |
| | ours448 (matched) | 0.427 | 0.281 | 0.350 | 0.055 |
| | released Difix-ref (zero-shot) | 0.486 | 0.138 | 0.186 | 0.238 |
| | ours, ref, 576×1024 | **0.414** | **0.311** | **0.490** | 0.160 |

**Reading.**
* **Matched comparison.** On identical data, steps and settings, our trainer ends better than the official
  trainer on every VAL metric (LPIPS 0.299 vs 0.388, FID 52.8 vs 59.9) and on the TEST image and lane metrics.
  So our code at least reproduces the official trainer's behavior; it does not under-perform it.
* **Why the official run trails.** The gap is not a bug on our side. Two things in the official recipe explain it:
  - its Gram term is about 100× larger than L2 + LPIPS at this scale (0.5 × 150–1000 on raw renders vs. about 0.5
    for ours), so it dominates the loss;
  - it trains in pure bf16.
* **Our driving model (reference view, 576×1024)** is the strongest Difix variant on held-out driving scenes:
  - best LPIPS, best CLRNet lane agreement and best lane-paint F1 off-path;
  - it matches the released Difix-ref's FID on VAL;
  - but it invents more vehicles than the raw render at ~3.6 m (0.18 vs 0.04 per image).
* **DL3DV** (section 1, added later): the same matched ordering holds on DL3DV, and our ref-view model ties the
  released Difix-ref on LPIPS there (not on FID).

## 3. Differences between our trainer and the official trainer
| | Ours | Official |
|---|---|---|
| Code | own model builder (diffusers/peft), own attention processor for view mixing, own loop / data / loss | `Difix` class + copied `mv_unet.py` (2 views hard-coded), accelerate loop |
| Views | any V (V set per call) | `num_views = 2` hard-coded when `--mv_unet` |
| Decoded views | only the view being fixed; loss only on it | both views decoded; the reference view is supervised against itself |
| Memory savers | U-Net + VAE-decoder activation checkpointing, encoder without autograd, 8-bit AdamW with fp32 master weights | U-Net checkpointing (flag), xformers; everything else in bf16 |
| Gram loss | Johnson-style: Gram / (C·H·W), squared Frobenius norm, equal layer weights, full image | un-normalized Gram per layer, MSE / (C·H·W), per-layer weights up to 6.7, random 400×400 crop, optional warm-up |
| Stopping | stops at `--steps` or `--minutes`; resumable | `max_train_steps` only sets the progress bar and scheduler; the loop keeps running for `num_training_epochs` (we stopped it after the step-1801 checkpoint) |
| Checkpoints | trained parts only (best by VAL LPIPS, last with optimizer) | full U-Net + LoRA + optimizer (~5.2 GB) at `step % N == 1` |
| Validation | full-frame VAL LPIPS / PSNR every N steps, best kept | val L2 / LPIPS logged to the tracker only |

### Patches needed to run the official trainer
The patched copy is `$DATA/difix_reimpl/official_src/`; `*.orig` files are kept next to the patched ones. The same
changes are in `src/fixers/difix_reimpl/official_trainer.patch` (upstream commit c76edc5).
1. `dataset.py` uses undefined names (`img_t = F.to_tensor(img_t)`, likewise `output_t` and `ref_t`), so it crashes
   on the first sample. Fixed to `F.to_tensor(input_img)` etc.
2. `train_difix.py` shuffles `dataset_val.img_names`, which does not exist; renamed to `img_ids`.
3. `Accelerator(...)` has no `project_dir`, which offline TensorBoard logging needs; added.
4. Training-time `wandb.Image` visualization is guarded by `report_to == "wandb"`.
5. New arguments `--data_height/--data_width` for the loader size, which was hard-coded to 576×1024 and does not fit
   24 GB with the reference view.

## Re-run
```
# DL3DV (section 1): download, pairs at S = 30 / 60, JSONs, three training runs, eval (each a GPU job via $GJ)
scripts/19_dl3dv.sh download; $GJ scripts/19_dl3dv.sh fit 30; $GJ scripts/19_dl3dv.sh fit 60; scripts/19_dl3dv.sh json
$GJ scripts/19_dl3dv.sh ours448; $GJ scripts/19_dl3dv.sh official448; $GJ scripts/19_dl3dv.sh oursref
$GJ scripts/19_dl3dv.sh eval released; $GJ scripts/19_dl3dv.sh eval trained; $GJ scripts/19_dl3dv.sh eval metrics
scripts/lib/py.sh src/dl3dv/summarize.py; scripts/lib/py.sh src/dl3dv/resample_floor.py
# driving data (section 2)
H=src/fixers/difix_reimpl
bash $H/jobs.sh ours        # our driving model (ref view, 576x1024), ~50 min GPU job
bash $H/jobs.sh official    # official trainer, matched (V=1, 448x768); stop it after model_1801.pkl appears
bash $H/jobs.sh ours448     # our trainer, matched
$GJ bash $H/eval_job.sh base       # val outputs (ours, released) + test outputs/metrics for ours
$GJ bash $H/eval_job.sh official   # matched pair on val + test, val table
python3 $H/compact_table.py results/difix_reimpl/summary_test.csv results/difix_reimpl/summary_test_compact.md
```
The pairs JSON is built by `data.build_pairs_json` (see `data.py`). The official trainer's `max_train_steps` does
not stop training, so its job ends at the 3500 s timeout unless it is stopped after the step-1801 checkpoint.
