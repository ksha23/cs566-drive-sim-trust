# NVIDIA Fixer [4] as a frozen baseline, plus a 24 GB fine-tune pilot

Fixer (github.com/nv-tlabs/Fixer, Apache-2.0 code; weights huggingface.co/nvidia/Fixer, not gated) applies the Difix3D+
recipe to Cosmos-Predict2 0.6B and is trained on NVIDIA's AV data. It is 1-step (timestep 250), with a 674M-parameter DiT
and a 77M-parameter VAE. Code: `src/fixers/nv_fixer/`.

## Setup (no Docker on sbel, so we built our own environment)
The official route is NVIDIA's `cosmos-predict2-container:1.2` Docker image. We use a pixi env instead:
Python 3.12, torch 2.6.0+cu126, `cosmos-predict2==1.0.9 --no-deps` plus its pure-Python deps, and flash-attn 2.7.4
(prebuilt cxx11abiTRUE wheel; only needed by the unused Qwen text-encoder import).
- Everything lives at `$CS566D_DATA/nv_fixer/{repo,env,weights,cache}`. Run scripts with `src/fixers/nv_fixer/run.sh`.
- **Transformer Engine shim** (`src/fixers/nv_fixer/te_shim/`):
  - Why: TE's PyTorch extension ships only as an sdist.
  - What it replaces: the 3 TE symbols the DiT uses (RMSNorm, DotProductAttention, apply_rotary_pos_emb), in pure PyTorch following TE 1.13 semantics.
  - `megatron.core` is imported while TE is hidden (needed only for `parallel_state`). Otherwise it wraps TE classes the shim doesn't provide.
- **Validation.** The official examples (`repo/examples/*.png`) give visually clean outputs.

  Differences from TE's fused kernels are expected only at bf16 rounding level. **We did not compare numerically against real TE, because we could not build it.**

## Inference protocol (`infer.py`)
- **Resolution:** our 960x540 renders are resized to Fixer's 960-wide setting (960x544, bilinear), and the output is resized back (bilinear), as in the official script.
- **Settings:** bf16, batch 1, no `torch.compile`, timestep 250 (the README command).
- **Items:** the same as every other fixer. Para-Lane test renders use `full` pairs, stride 2 plus the on-path held-out frames, for scene_16 and scene_17. The HUGSIM offset renders cover 5 scenes × 30 frames × 7 offsets.
- **Speed (RTX 3090):** 149 ms/img median, 1.8 GiB peak. For comparison, Difix takes 701 ms, Difix-ref 1422 ms and our U-Net 84 ms.

## Results
**Para-Lane test scenes, frozen** (scene_16 + scene_17; mean, 95% CI from resampling whole lane-pair sequences). Full table: `results/nv_fixer/summary_test_nvfixer_vs_baselines.{csv,md}`.

| offset | method | PSNR | SSIM | LPIPS | road LPIPS | paint P / R / F1 | NTL-IoU | CULane F1 | fake veh./img |
|---|---|---|---|---|---|---|---|---|---|
| 0 m | raw | 24.68 | 0.782 | 0.302 | 0.478 | 0.94 / 0.95 / 0.94 | 0.898 | 0.976 | 0.04 |
| 0 m | NVIDIA Fixer | 23.76 | 0.746 | 0.331 | 0.457 | 0.93 / 0.91 / 0.91 | 0.895 | 0.967 | 0.02 |
| ~3.6 m | raw | 17.35 | 0.554 | 0.524 | 0.750 | 0.53 / 0.54 / 0.52 | 0.070 | 0.074 | 0.04 |
| ~3.6 m | Difix (zero-shot) | 17.26 | 0.515 | 0.439 | 0.579 | 0.48 / 0.52 / 0.49 | 0.275 | 0.307 | 0.04 |
| ~3.6 m | Difix-FT (ours) | 17.40 | 0.508 | 0.333 | 0.380 | 0.68 / 0.61 / 0.62 | 0.580 | 0.682 | 0.06 |
| ~3.6 m | NVIDIA Fixer | **17.86** | **0.577** | 0.465 | 0.597 | 0.60 / 0.58 / 0.58 | 0.407 | 0.468 | 0.03 |
| ~7.1 m | raw | 15.03 | 0.481 | 0.664 | 0.904 | 0.32 / 0.15 / 0.16 | 0.016 | 0.021 | 0.22 |
| ~7.1 m | Difix (zero-shot) | 14.98 | 0.437 | 0.558 | 0.720 | 0.32 / 0.14 / 0.15 | 0.121 | 0.145 | 0.11 |
| ~7.1 m | Difix-FT (ours) | 15.85 | 0.463 | 0.427 | 0.480 | 0.45 / 0.18 / 0.22 | 0.434 | 0.475 | 0.06 |
| ~7.1 m | NVIDIA Fixer | 15.47 | **0.511** | 0.584 | 0.744 | 0.42 / 0.09 / 0.12 | 0.194 | 0.233 | 0.07 |

- **Frozen Fixer is conservative:**
  - It improves off-path PSNR and SSIM the most, and it does not invent brick textures or fake vehicles.
  - It gains less LPIPS than Difix (road LPIPS barely moves).
  - It recovers some lanes (NTL-IoU 0.07 → 0.41 at ~3.6 m). Difix-FT, trained on our Para-Lane pairs, recovers more.
- **On-path it slightly degrades renders** (PSNR −0.9 dB, LPIPS +0.03).

**HUGSIM NTA-IoU** (`results/hugsim/nv_fixer/nta_iou_summary.csv`; same scenes and protocol as `results/hugsim`):

| \|offset\| | raw | NVIDIA Fixer | Difix (from results/hugsim) |
|---|---|---|---|
| 0 m | 0.400 [0.359, 0.441] | **0.454** [0.421, 0.488] | 0.429 |
| 1 m | 0.392 | **0.438** | 0.417 |
| 2 m | 0.371 | **0.418** | 0.396 |
| 3 m | 0.336 [0.310, 0.364] | **0.392** [0.369, 0.416] | 0.369 |

Fixer, trained on AV data, gives the largest NTA-IoU gain of any fixer on nuScenes (+0.05 to +0.06). Its CIs do not overlap raw's at 0 m or 3 m.

## 24 GB fine-tune pilot (`finetune_pilot.py`, `results/nv_fixer/finetune_pilot.json`)
**Setup.** The pilot mirrors the official trainer's core:
- Trainable parts: the full DiT plus the VAE except its time_conv layers, 751M trainable parameters.
- Input 1024x576, timestep 250.
- Loss: L2 + 0.3·LPIPS-VGG on a random 128–512 crop; AdamW, lr 2e-5.
- The official `tokenizer.patch` is applied at runtime.
- **Omitted:** the CLIP/vision-aided GAN discriminator and the CLIP-similarity loss (extra dependencies), and the Gram loss (it has a 2000-step warm-up anyway).

**Memory probes** (6 steps each, batch 1):

| config | peak alloc | peak reserved | s/step |
|---|---|---|---|
| bf16 weights + AdamW (like the official `--mixed_precision bf16`, net cast to bf16) | 15.1 GiB | 15.4 GiB | 0.60 |
| bf16 weights + 8-bit AdamW | 13.9 GiB | 14.4 GiB | 0.60 |
| fp32 master weights + 8-bit AdamW + bf16 autocast | 16.9 GiB | 17.5 GiB | 0.61 |

**All three fit in 24 GB.** A GAN discriminator would add memory. We did not test it.

**Short fine-tune.** We ran fp32-master + 8-bit AdamW for 400 steps (0.68 s/step, 4.5 min) on the train pairs of scenes 20–24. Validation was on 60 frames of scene_19 (full frame, LPIPS-Alex vs. real, movable objects pasted from the real image):
- **LPIPS went from 0.529 to 0.411.** So fine-tuning Fixer on 24 GB is feasible and moves quickly toward our domain.
- **Caveat:** this validation subset is not the 279-image set used to select the U-Nets and Difix-FT, so the numbers are not directly comparable to those logs.

## Re-run
```
src/fixers/nv_fixer/run.sh src/fixers/nv_fixer/infer.py --paralane scene_16,scene_17 --timing $CS566D_DATA/fixed/nv_fixer_timing.json   # GPU job
src/fixers/nv_fixer/run.sh src/fixers/nv_fixer/infer.py --hugsim scene-0064,scene-0167,scene-0661,scene-0071,scene-0383          # GPU job
scripts/lib/py.sh src/eval/evaluate.py $CS566D_DATA/pairs <scene> $CS566D_DATA/fixed results/eval/eval_<scene>_nvfixer.json --methods nv_fixer
scripts/lib/lanes_scene.sh <scene> $CS566D_DATA/fixed results/lanes/ntl_<scene>_nvfixer.json nv_fixer
scripts/lib/py.sh src/hugsim/nta_iou.py --scenes scene-0064,scene-0167,scene-0661,scene-0071 --out results/hugsim/nv_fixer --methods raw,nv_fixer
src/fixers/nv_fixer/run.sh src/fixers/nv_fixer/finetune_pilot.py --probe                   # memory probe (GPU job)
src/fixers/nv_fixer/run.sh src/fixers/nv_fixer/finetune_pilot.py --config fp32m_adam8 --steps 400
```
Every GPU step goes through `~/cs566/gpujob.sh`. The env-setup commands are in `src/fixers/nv_fixer/setup_env.sh`.
