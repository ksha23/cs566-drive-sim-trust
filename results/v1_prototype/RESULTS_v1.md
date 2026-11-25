# Results (prototype, 2026-09-26 night)

Setup: Para-Lane, front camera at 960x540. Train scenes for the U-Net: scene_20, scene_22 (3,012 pairs).
Held-out test scene: scene_16 (never seen by the U-Net). Eval on scene_16 uses full splats, all 6 ordered lane pairs
(every 2nd frame) plus the on-path held-out frames: 38 on-path, 187 at ~3.1 m, 100 at ~6.2 m.
Pixels on movable objects (Para-Lane foreground mask of the real target image) are pasted from the real image before scoring.

## 1. On-path splat quality (held-out every 8th frame, 12k iters, ~4.5 min per lane)
| scene | clip_0 PSNR / LPIPS | clip_1 | clip_2 |
|---|---|---|---|
| scene_16 | 25.85 / 0.311 | 25.81 / 0.315 | 25.26 / 0.330 |
| scene_20 | 21.40 / 0.394 | 24.30 / 0.287 | 23.93 / 0.311 |
| scene_22 | 23.15 / 0.446 | 23.53 / 0.435 | 23.79 / 0.424 |

## 2. Trust vs. lateral offset on held-out scene_16 (mean over frames)
| offset | method | PSNR↑ | LPIPS↓ | road LPIPS↓ | paint F1↑ | paint prec↑ | YOLO static F1↑ | halluc. vehicles/img↓ | ms/img |
|---|---|---|---|---|---|---|---|---|---|
| 0 m | raw | **25.63** | 0.317 | 0.573 | 0.968 | 0.960 | 0.58 | 0.026 | – |
| 0 m | Difix | 23.64 | 0.249 | 0.381 | 0.729 | 0.719 | 0.44 | 0.079 | 688 (GPU) |
| 0 m | Difix-ref | 24.63 | **0.133** | **0.188** | 0.981 | 0.998 | 0.52 | 0.026 | 1426 (GPU) |
| 0 m | ours | 22.60 | 0.219 | 0.286 | 0.832 | 0.948 | 0.51 | 0.026 | **89 (GPU)** |
| ~3.1 m | raw | 19.63 | 0.493 | 0.869 | 0.473 | 0.454 | 0.45 | 0.032 | – |
| ~3.1 m | Difix | 19.38 | 0.380 | 0.610 | 0.409 | 0.392 | 0.48 | 0.064 | 688 |
| ~3.1 m | Difix-ref | 19.64 | **0.230** | **0.294** | 0.689 | 0.736 | 0.48 | 0.043 | 1426 |
| ~3.1 m | ours | **20.10** | 0.283 | 0.375 | **0.705** | **0.762** | **0.51** | **0.011** | |
| ~6.2 m | raw | 17.61 | 0.554 | 0.914 | 0.417 | 0.442 | **0.52** | 0.040 | – |
| ~6.2 m | Difix | 17.56 | 0.433 | 0.648 | 0.310 | 0.339 | 0.45 | 0.120 | 688 |
| ~6.2 m | Difix-ref | 17.59 | **0.307** | **0.382** | **0.466** | 0.567 | 0.49 | 0.090 | 1426 |
| ~6.2 m | ours | **18.58** | 0.349 | 0.462 | 0.415 | **0.573** | 0.45 | **0.000** | |

Full table incl. SSIM, paint recall, texture ratio: `summary_scene_16.csv`; per-image rows: `eval_scene_16.json`.
Our U-Net speed: 89 ms/img median on the 3090 (bf16, 960x540, incl. host->GPU copy, 0.77 GiB peak), vs 688 / 1426 ms for
Difix / Difix-ref; 1.65 s/img on CPU (the full eval run was done on CPU while the shared GPU was busy; identical weights).
Fly-through: `flythrough_scene16_clip0.mp4` (raw | ours, sliding 0 -> 6.3 m right along clip_0).

## Notes / caveats
* Difix-ref's reference is the nearest REAL frame of the source lane (what a simulator has). On-path that is the adjacent
  frame (0.1 s away), which explains its very low on-path LPIPS.
* Our U-Net trained for only 21 min (2,542 steps of a 4,500-step one-cycle schedule, so the LR was never annealed) on 2 scenes.
  Its outputs are smooth/blurry (regression to the mean), which favors PSNR. It hurts on-path frames (25.6 → 22.6 dB)
  because it was trained almost only on cross-lane pairs, where it also learns the color/exposure shift between passes.
* Paint metric = white/yellow paint masks (row-wise asphalt percentile + saturation threshold) in the lower 40% of the frame,
  3-px tolerant precision/recall. CLRNet was not set up (no time).
* YOLO: static classes (traffic light, stop sign, …) only, because the splats drop all Para-Lane movable objects
  (parked and moving) so vehicle agreement is not meaningful; "hallucinated vehicles" counts vehicle/person boxes where the real
  frame has no movable object. With 1–3 traffic lights per frame the YOLO F1 is noisy (s.e. ≈ 0.05–0.1).
