# MV2: car <-> scooter, the camera-height axis with real ground truth (stage 17)

Entry point: `scripts/17_mv2.sh` (`download | masks | geom | fit | fitall | fix | fixall | lanes | eval | summary`),
driver `src/mv2/chain.sh`; code `src/mv2/{download,data,fit,evaluate,figures}.py`; results `results/mv2/`.

MV2 [9] (HF `sanjay810/mv2-dataset`, academic use) recorded Indian urban roads with a car, a scooter and a drone at the
same time. We use the car's front camera and the scooter's front camera (GoPro, 1920x1080, 100 frames per sequence).
The two vehicles drive **simultaneously in different lanes with different camera heights**, so rendering a splat built
from one vehicle at the other's real poses is a real-ground-truth test of lateral offset + camera height together.
It is also a transfer test: every fixer is used as trained (Para-Lane or generic), with no MV2 fine-tuning.

## Poses, calibration, time sync, scale
* `poses/<cam>_<k>.npy` are 4x4 **world-to-camera** matrices, OpenCV axes (x right, y down, z forward), one world
  frame for all vehicles of a sequence, world z up (checked on seq1: read as world-to-camera, the camera's forward
  axis points along the direction of travel, cos 0.996; read as camera-to-world it does not).
* `intrinsics/<cam>_<k>.npy` = `[fx, fy, cx, cy, W, H]` for **960x540**, while the JPGs are 1920x1080: images are
  resized to 960x540 and K is used as is (pinhole; GoPro distortion is not modeled, MV2 ships none).
* Time sync: frame k of the car and of the scooter are simultaneous (MV2 samples every sensor at 2 FPS on a common
  clock); the car->scooter offset is nearly constant within a sequence (std 0.15–0.5 m laterally).
* **Metric scale.** MV2's poses are normalized (nerfstudio-style `scale_factor.npy` / `transform.npy`), not metric. We
  triangulate points from each vehicle's own views with the given poses (SIFT, frame gaps 1–3, reprojection < 1.5 px,
  parallax > 1°; `src/mv2/data.py`), fit a local road plane under each camera (RANSAC, normal within 15° of up), and
  set the scale so that the **car camera is 1.40 m above the road (assumption)**. The scooter's height is then
  measured, not assumed. Everything metric below (lateral offsets, heights) inherits this one assumption; a 10 % error
  in the car height scales all of them by 10 %.
* Masks: YOLO11m-seg movable objects (as EUVS) + the car's bonnet (bottom 18 rows at 960x540).

## Sequences
Chosen for clear, steady car/scooter separation (of 50 sequences, 14 screened; `$DATA/mv2/select_stats.json`), final geometry from
`$DATA/mv2/cache/<seq>/geom.json`:

| seq | scooter − car lateral (m, median) | std (m) | forward (m) | camera-center height diff (m) | scooter camera above road (m, measured) | triangulated points car / scooter | m per MV2 unit |
|---|---|---|---|---|---|---|---|
| seq1 | 7.5 | 0.51 | 2.29 | -0.75 | 0.92 | 28874 / 22018 | 117 |
| seq21 | 3.9 | 0.22 | 0.93 | -0.54 | 1.01 | 22673 / 18715 | 102 |
| seq17 | 5.2 | 0.29 | 1.01 | -0.70 | 1.25 | 34252 / 32772 | 94 |
| seq12 | 8.8 | 0.15 | 2.05 | -0.60 | 1.02 | 28198 / 25760 | 107 |
| seq13 | 8.3 | 0.39 | 1.84 | -0.57 | 0.97 | 27637 / 19113 | 100 |

Car camera = 1.40 m above the road by assumption in all sequences. 14 of the 50 sequences were screened
(`$DATA/mv2/select_stats.json`); these 5 have a steady offset (lateral std ≤ 0.51 m) and the full 100 frames per
vehicle. The measured camera-height difference (scooter − car, above the local road plane) is −0.38 to −0.48 m in four
of them and −0.15 m in seq17 (`results/mv2/per_sequence.md`).
seq21 and seq17 are the closest pairs (~4–5 m), seq1/12/13 are two or more lanes apart (7.5–8.8 m).

## Protocol
* `car2scooty`: splat fitted on the car's frames (every 8th frame held out as the 0 m on-path check), rendered at the
  scooter's real simultaneous poses (every 2nd frame); `scooty2car`: the reverse.
* Splat = the EUVS model (`src/euvs/fit.py`: gsplat SH3, sky map, masked L1 + D-SSIM, 15k iterations), initialized
  from the training vehicle's triangulated points only.
* Reference view for ref-based fixers: nearest training-vehicle frame (center distance [m] + 0.1 × angle [deg]).
* Metrics: `src/mv2/evaluate.py`, as Para-Lane/EUVS (movable objects and the ego bonnet pasted from the real image).
  CIs: 95 % bootstrap over sequences.

## Results
Tables: `results/mv2/summary.md` (per direction × on-path/cross × method, all metrics, CIs over sequences, paired
deltas vs. raw), `results/mv2/per_sequence.md`; discussion in RESULTS §14. In short:
* Cross-vehicle renders are at the Para-Lane ~6 m level: CLRNet lane F1 ≤ 0.02 raw and ≤ 0.09 for any fixer.
* Per sequence, the raw loss tracks the lateral separation (LPIPS r = +0.98 / +0.75). The height difference varies
  little (0.38–0.48 m, 0.15 m in seq17), so MV2 does not isolate the height axis. Rendering upward (scooter → car) is
  worse than downward (LPIPS 0.592 vs 0.499), but that comparison also swaps the training vehicle.
* Released Difix-ref helps most (LPIPS −0.095 / −0.164); NVIDIA Fixer has the best PSNR but the most invented
  vehicles; the Para-Lane-trained arms transfer partially; our small U-Net hurts.

## What did not work / caveats
* No metric scale in MV2 (normalized poses): one assumed car-camera height (1.40 m) sets every metric number.
* GoPro lens distortion is not modeled (MV2 ships a pinhole K only); residual distortion is part of every error.
* Some renders are visibly offset from the real frame (seq1): large offset and/or pose/scale error, not separable here.
* CLRNet finds lanes on 83–96 % of the real cross-vehicle frames, so lane
  metrics are defined but low for everyone.
* 5 sequences only; CIs over sequences are wide.

