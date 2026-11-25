# Implementation notes and paper-vs-ours decisions

Component-level notes for everything except the Difix reimplementation (see `IMPLEMENTATION_NOTES_difix.md`),
NVIDIA Fixer (`nv_fixer.md`) and FaithFusion (`faithfusion.md`).

## Street splats (`src/splats/`)
* **Data.** Para-Lane 0315 scenes, front camera resized to 960x540 (native 1910x1075). COLMAP poses; the three clips of a
  scene share one world frame (x forward, y left, z up). Movable objects come from Para-Lane's foreground masks
  (dilated 9 px); they are excluded from all photometric losses and pasted from the real image before every metric,
  because cars differ between passes.
* **Fit** (gsplat 1.4, 12k iterations, ~4–6 min/lane): initialized from the clip's merged LiDAR cloud (60 m corridor,
  8 cm voxels, colored from the nearest un-occluded view; points ever landing on a movable-object mask dropped);
  LiDAR depth supervision (min-z per 4x4 cell, relative L1, weight 0.2 → 0.05); a learnable equirectangular sky
  environment map composited behind the splat plus an alpha→0 penalty on a LiDAR-skyline ∧ sky-color mask;
  0.8·L1 + 0.2·(1−SSIM); scale-anisotropy regularizer; every 8th frame held out.
* **Road-surface prior** (stretch, simplified ViSE): height field z = h(x, y) on a 0.5 m grid from low LiDAR returns
  (per-cell 10th percentile, hole-filled, Gaussian-smoothed → height and slope); Gaussians within 0.3 m are pulled onto
  it and flattened along its normal. ViSE has no public code; this is our own formulation.

## Pairs (`src/pairs/render_pairs.py`) — Difix3D+ curation adapted to parallel passes
| Difix3D+ strategy | Ours |
|---|---|
| cross reference (train on one camera, render the others) | **cross-lane**: lane A's splat at lane B's real poses, both directions |
| model underfitting (25–75% of the schedule) | splat saved at 2k of 12k iterations (17%), rendered at all poses |
| cycle reconstruction (shift 1–6 m, refit on renders, render back) | shift ±3 m (alternating), 8k-iter refit on the renders only (+ LiDAR depth), rendered back at A's poses |
| sparse reconstruction | not used (Para-Lane lanes are dense along-track) |
| — | **on-path identity pairs** (A's splat at A's training poses) so the fixer learns to leave good renders alone |
Train counts: 11,900 pairs (full 3,320, identity 1,600, under-fit 5,150, cycle 1,830); sampling weights per type
1.0 / 0.6 / 0.5 / 1.0 (`results/pairs/pair_counts.json`).

## Our fixer, smaller version from scratch (`src/fixers/unet/`)
Residual U-Net (base 48, 4 levels, GroupNorm+SiLU ResBlocks, zero-initialized output conv → identity at init), 15.2M
params. L1 + 0.2·LPIPS-VGG, optional Sobel high-frequency loss (w 0.5) and PatchGAN hinge term (implemented, not
trained for lack of GPU time). 512² crops, batch 6, AdamW 2e-4, cosine schedule, 1,800 steps; checkpoint selected by
val LPIPS on scene_19; config selected by mean val LPIPS over seeds (base, `results/training/selected_ours.txt`).

## Difix fine-tune from released weights (`src/fixers/difix_ft/`)
Not the reimplementation: starts from `nvidia/difix`, full U-Net fine-tune (VAE and its skip-LoRA frozen) with fp32
master weights + 8-bit AdamW, bf16 autocast, gradient checkpointing, 512² crops, batch 1, lr 1e-5 cosine, 4,000 steps,
MSE + LPIPS (no Gram), peak 18.1 GB. Val LPIPS 0.493 → 0.289 (`results/training/difix_ft.json`).
Bug found and fixed on the way: a differentiable Difix forward must copy the encoder's skip activations to the decoder
(`vae.decoder.incoming_skip_acts = vae.encoder.current_down_blocks`) after every encode.

## Progressive 3D update (`src/progressive/progressive.py`) — Difix3D+ Alg. 1 with ReconDreamer-style masking
Difix3D+: "after every 1.5k iterations, slightly perturb the ground-truth poses toward the target views, render, refine
with the diffusion model, add to the training set". Ours: target = lateral offsets (unknown to the procedure which side
the other lanes are), steps of **3 m then 6 m** (every 3rd training pose, alternating left/right), each followed by a
**2k-iteration** refinement on real views + all pseudo-views so far (pseudo-view probability 0.4, lower learning
rates). The paper gives no step size; our steps are coarser than its "slightly" because of GPU time (2 rounds). Pseudo
view loss masked as in ReconDreamer's DriveRestorer: no sky (sky-colored or empty pixels in the upper half), no far
region (rendered depth > 40 m); empty pixels in the lower half (road holes) are kept, since that is where the fixer's
restoration should be distilled. A first version that also masked all alpha<0.5 pixels was discarded (it removed
exactly the restored road). Control: same loop with the raw render as pseudo view (`prog_none`).

## Evaluation (`src/eval/`)
* Image metrics vs. the REAL target-lane image (movable objects pasted from real): PSNR, SSIM, LPIPS-Alex; road LPIPS on
  the lower 40% of the frame.
* Hallucination: white/yellow paint masks (row-wise asphalt 25th percentile + margin, saturation/hue) in the road
  region, 3-px tolerant precision (low = invented paint), recall (low = erased paint), F1 — always all three; road
  texture ratio (mean |Laplacian| vs. real).
* CLRNet (Turoad, CULane DLA34 weights): NTL-IoU = IoU of lanes rasterised 30 px thick at CULane scale; CULane-style
  lane F1 at IoU 0.5 (`src/eval/lanes/`).
* YOLO11m: static-object F1 and fake-vehicle count (vehicle/person boxes where the real frame has no movable object).
* 95% bootstrap CIs over frames and over lane-pair sequences (the latter reported; frames within a sequence are
  correlated).
* Difix-ref reference image: the nearest REAL frame of the SOURCE lane (what a simulator has), never the target; on-path
  that is the adjacent frame (~0.1 s). Variant `difix_ref_far`: 20 frames (2 s) away.

## HUGSIM (`src/hugsim/`)
NTA-IoU re-implemented from ReconDreamer's description (no released scorer); LTF (NAVSIM) policy client with a fixer
server between simulator and policy. Details and what did not run: `results/hugsim/CLOSED_LOOP.md`.
