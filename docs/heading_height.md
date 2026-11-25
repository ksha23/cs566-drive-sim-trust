# Heading (yaw) and camera-height axis on Para-Lane

Code: `src/splats/render_deviation.py` (renders), `src/eval/heading_geom.py` (conventions, warps),
`src/eval/heading.py` (metrics, summary), `src/eval/heading_figures.py`, entry point `scripts/15_heading_height.sh`.
Results: `results/heading/`, figures `results/figures/trust_vs_heading_height.png`, `results/figures/panel_heading_*.jpg`.

## What is rendered
Test scenes scene_16 and scene_17, every lane's full splat (the same splats as the lateral evaluation):
* **on-path**: each lane's held-out frames (every 8th, never trained on; 101 frames = the 0 m rows of the lateral
  table) at yaw ±5, ±10, ±15° (height unchanged) and camera height ±0.5, ±1.0 m (heading unchanged);
* **lateral + yaw**: lane A's splat at lane B's *real* poses (the cross-lane eval pairs, every 3rd of the stride-2
  subset: 257 frames, ~3 m and ~6 m) rotated by ±10°.
The recorded-pose condition is not re-rendered or re-fixed: its directory is a symlink to the lateral evaluation's
renders and fixer outputs, and our metrics at 0° reproduce the lateral evaluation's per-image values for all seven
methods (2,506 rows; max |difference| ≤ 1e-5 on every image metric, paint P/R/F1, fake vehicles, static F1, texture,
lane F1, and 0.0005 on NTL-IoU from the 1-px border erosion of the valid mask; `results/heading/checks.json`).

## Conventions
World = Para-Lane COLMAP frame, x forward, y left, z up (checked: camera pitch 5.0–5.3° down, roll < 0.7°). Camera
frame = OpenCV. **Yaw** ψ rotates the camera about the world z axis through its own center (R' = Rz(ψ) R, C' = C);
ψ > 0 turns **left**. **Height** dz moves the center along world +z (R' = R); dz > 0 is higher (truck), dz < 0 lower.
The camera is 1.74–1.97 m above the LiDAR ground (per frame: 10th percentile of LiDAR z within 4 m), so −1 m is a
low car (~0.9 m) and +1 m a truck/van (~2.9 m). Condition names: `y{ψ:+d}_z{dz:+.1f}`.
Every fixer gets the same inputs as in the lateral evaluation; Difix-ref and our Difix-ref reimplementation get the
nearest real frame of the source lane (by camera center) as reference, i.e. at the recorded heading.

## Ground truth without a real image
1. **Yaw: rotation homography (exact).** Rotating a camera about its optical center changes no parallax, so the real
   frame warped by H = K R1ᵀR0 K⁻¹ *is* the image the rotated camera would have taken, for every point at any depth
   (not only at infinity), except pixels whose source lies outside the recorded frame (black; excluded; 90 / 81 / 74 %
   of the frame is valid at 5 / 10 / 15°). This makes yaw a real-ground-truth axis on Para-Lane: PSNR on valid pixels,
   SSIM/LPIPS on the largest fully valid column band, paint P/R/F1 on valid pixels, CLRNet lanes of the real frame
   warped by H, static-object boxes warped by H, fake vehicles against the warped movable-object mask.
   Caveat: a real vehicle turns about its rear axle, which adds a sideways shift of L·sin ψ; we rotate about the camera
   center (the lateral + yaw condition covers rotation combined with a real lateral shift).
2. **Height: road-plane homography (road only).** H = K (I + dz n nᵀ / h) K⁻¹ (n = world up in camera coordinates, h =
   camera height) is exact for points on the road plane and wrong for everything above it, so against it we only
   report road metrics (road PSNR/LPIPS on the lower 40 %, paint P/R/F1, CLRNet lanes below the horizon); fake vehicles
   use the plane-warped movable-object mask dilated by 41 px vertically. Lowering the camera shows mostly road the
   recorded camera never saw (at −1 m only rows ~324–363 of 540 are covered), so the −1 m pseudo-ground truth is thin.
3. **Matched 0-deviation baseline.** Scoring a smaller or different image region at larger deviations biases the raw
   numbers. Every deviated row therefore also stores `ref0_*`: the same method's 0-deviation output against the real
   frame on the SAME scene content (the evaluated region warped back into the recorded frame); `d_k = k − ref0_k`.
4. **Consistency with the same method at 0°** (no ground truth needed): CLRNet lanes of the deviated output vs. the
   0-deviation output's lanes warped by H (`lane_stab_*`), YOLO boxes likewise (`det_stab_f1`, yaw only),
   LPIPS(output, H(0-deviation output)) (`self_lpips`), road-texture energy relative to the 0-deviation output (`tex_stab`).

**A bias we found and fixed.** With bilinear warping the pseudo-ground truth is slightly blurred (real frame warped
there and back: 31.9 dB), which made raw renders score ~0.8 dB *better* at ±5–10° than at 0° on the same content.
With Lanczos-4 (round trip 41.9 dB) the raw render at ±5/10° scores the same as the 0° render on the same content
(25.78 vs. 25.75 dB, 152 frames; bilinear: 26.57 vs. 25.75), as a pure rotation should (`results/heading/checks.json`);
all reported numbers use Lanczos-4.

**Other caveats.** `det_stab_f1` (YOLO boxes under the warp) is noisy: most on-path frames have zero or one small
detection (often a distant traffic light near the 0.3 confidence threshold), so it is low even for the raw render and
is not used for conclusions. At −1 m CLRNet finds almost no lanes on any method's output (also on its own
self-consistency test), so we cannot separate render damage (the near road is visibly smeared, see panels) from the
detector's own shift for a camera ~0.9 m above the road.

CLRNet runs with the fixed Para-Lane crop (horizon row 202): a pure yaw or height change leaves the horizon row
unchanged, so the crop stays correct.
