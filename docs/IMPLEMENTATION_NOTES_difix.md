# Difix re-implementation: implementation notes

Code: `src/fixers/difix_reimpl/` (`model.py`, `losses.py`, `data.py`, `train.py`, `infer.py`, `val_metrics.py`).
Written from the Difix3D+ paper (Wu et al., CVPR 2025, arXiv 2503.01774, Sec. 4.1 and 4.2). We read the released code
(`nv-tlabs/Difix3D`: `src/model.py`, `mv_unet.py`, `loss.py`, `train_difix.py`) only to check details; nothing from it
is imported by our trainer. The architecture is the same as the released one, rebuilt from diffusers/peft parts.

## What the paper specifies, and how we implement it

| Paper | Ours |
|---|---|
| Base: pretrained SD-Turbo | `stabilityai/sd-turbo` tokenizer, CLIP text encoder (frozen), VAE, U-Net, noise schedule |
| Single-step denoising at noise level τ = 200, degraded render as input | VAE latent of the render is used as z_t at timestep index 199 (τ = 200 of 1000); U-Net predicts ε; output latent z0 = (z_t − √(1−ᾱ_t) ε) / √ᾱ_t; no noise is added |
| Reference-view mixing layer: reshape latents `b c v (hw) → b c (v h w)`, self-attention, reshape back, inheriting 2D self-attention weights | `MultiViewAttnProcessor` on every `attn1` (self-attention) of the U-Net transformer blocks: tokens `(b v) n c → b (v n) c` before attention, back afterwards; V is set per call (V=2 with reference, V=1 without) |
| Frozen VAE encoder, LoRA fine-tuned decoder (following Pix2pix-Turbo) | encoder frozen (and run without autograd); LoRA rank 4 on all decoder convs and attention projections (39 modules); encoder→decoder skip connections as in Pix2pix-Turbo: four 1×1 convs from the encoder's down-block inputs to the decoder's up-block inputs, initialized to 1e-5 |
| U-Net fine-tuned | all 866M U-Net parameters trained |
| Loss L = L2 + LPIPS + 0.5 L_Gram (VGG-16 features) | `losses.py`: MSE on [−1,1] images + LPIPS(VGG) + 0.5 × Gram loss |

## Where the paper is ambiguous, and our choice

| Ambiguity | Our choice | What the official code does (for reference) |
|---|---|---|
| LoRA rank of the decoder | 4 (Pix2pix-Turbo default) | 4 |
| Which decoder layers get LoRA | convs (`conv1/conv2/conv_in/conv_shortcut/conv/conv_out`) + attention `to_q/k/v/out` | same set, plus LoRA on the skip convs (we train the skip convs fully instead) |
| Skip-connection design / init | 1×1 convs, init 1e-5, fully trained | same |
| Gram layers and weights | relu1_2, relu2_2, relu3_3, relu4_3, relu5_3; Gram normalized by C·H·W; MSE; equal weights; full image | same layers with per-layer weights (1/2.6, 1/4.8, 1/3.7, 1/5.6, 10/1.5), un-normalized Gram per layer divided by C·H·W, random 400×400 crop, 2000-step warm-up; default λ_Gram = 1.0 (README) vs 0.5 in the paper |
| Loss on the reference view | loss only on the fixed (first) view; only the first view is decoded | decodes both views and supervises the reference view against itself |
| Latent of the render | posterior sample of the VAE encoder | same |
| Resolution / cropping | full frame resized to 1024×576 (the size the official data loader uses); optional h-flip (off in the matched run) | `PairedDataset` resizes to 576×1024 (the `--resolution 512` flag is unused) |
| Optimizer, LR, schedule | 8-bit AdamW (bitsandbytes), lr 2e-5, wd 1e-2, 100-step linear warm-up then constant, grad clip 1.0 | AdamW, lr 2e-5 (README), constant (we pass `constant_with_warmup`, 100 steps, to match) |
| Precision | fp32 master weights, bf16 autocast, U-Net and decoder activation checkpointing | whole model cast to bf16 (`--mixed_precision bf16` casts weights), bf16 optimizer states |
| Batch size / iterations | 1 / time-boxed (see results) | README: 1 / 10k |
| Reference-view sampling (driving pairs) | nearest REAL frame of the source lane to the target pose (never the target frame itself) | not specified (dataset JSON gives it) |
| Prompt | fixed "remove degradation" | same |
| Movable objects (driving) | optional loss mask (`--fg_mask`); off in the matched comparison | none |
| Checkpoint selection | best validation LPIPS on the val scene (scene_19) | last checkpoint |

## Known issues found in the official trainer
See `docs/difix_reimpl_validation.md` (section 3) for the full list and the patches needed to run it:
* the data loader crashes on the first sample (undefined names);
* the val set is shuffled via a non-existent attribute;
* TensorBoard logging has no logging dir;
* `max_train_steps` does not stop the loop;
* at the loader's fixed 576×1024 size, the reference-view model does not fit 24 GB.

## Results in one line
Matched run (same data, steps and settings, V = 1, 448×768): our trainer reaches VAL LPIPS 0.299 / FID 52.8, against
0.388 / 59.9 for the official trainer. Our reference-view model at 576×1024 is the best Difix variant on the
held-out driving test scenes. Details: `docs/difix_reimpl_validation.md`.
