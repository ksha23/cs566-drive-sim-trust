# EUVS [6]: heading axis with real ground truth

Code: `src/euvs/` (download, COLMAP loader + metric alignment, splat fit + test renders, evaluation), entry point
`scripts/16_euvs.sh`. Results: `results/euvs/`.

## Data actually available
EUVS is the gated Hugging Face dataset `ai4ce/EUVS-Benchmark` (downloaded with the student's login, user `ksha23`).
Its three levels (paper Sec. 3): Setting1 = translation only (nuPlan lane changes), **Setting2 = rotation only**
(nuPlan: train on the front and rear cameras, test on the two side cameras), **Setting3 = translation + rotation**
(MARS / Argoverse2, other routes through the same place).

* **Setting2 (rotation only) cannot be run from Hugging Face.** Its archives contain only the COLMAP model, the
  train/test lists and geo-registration, **no images** (checked on `Setting2/nuplan_vegas_location_53.tar.gz`:
  12 files, `sparse/0/{cameras,images,points3D}.{bin,txt}`, `train_set.txt` (324), `test_set.txt` (108),
  `geo_registration.txt`; no `images/`). The images must be extracted from the nuPlan sensor logs named in
  `Data_extractor_for_setting1_and_2/Setting2_data_config.json` (e.g. vegas 53: `2021.06.23.14.06.20_veh-26_01563_02494.db`,
  cameras 1–8) with `download_nuplan_data.py`, which needs the nuPlan database *and* the nuPlan camera blobs. Those
  come from nuplan.org under Motional's own registration and license (not Hugging Face) and are large per log. We did
  not attempt this (separate account and terms needed from the student). Setting1 has the same limitation.
* **Setting3 ships images** (MARS: 3 cameras × ~5 traversals, 1440×928 JPEG; COLMAP PINHOLE model 1427×911, which we
  treat, as the EUVS 3DGS loader does, as covering the full image). We use the three smallest MARS scenes:
  `MARS_location_41_scene_02`, `MARS_location_43_scene_01`, `MARS_location_24_scene_01`.

## Two levels from the MARS scenes
* **L3 = EUVS level 3, translation + rotation (official split).** Training = the official `train_set.txt` (4
  traversals, all 3 cameras), test = the official `test_set.txt` (a held-out traversal along another route, all 3
  cameras).
* **rot60 = rotation only, EUVS Setting2 *protocol* on MARS data.** EUVS's rotation level trains on some cameras and
  tests on other cameras of the same traversals. MARS has a front camera (channel 2) and two cameras at ±60°
  (channels 1, 3; ~0.1 SfM units = ~0.7 m from the front camera). We train on channel 2 of every traversal and test on
  channels 1 and 3 of the EUVS test traversal. This is not the EUVS Setting2 benchmark itself (other data, 60° instead
  of nuPlan's side cameras); it follows its protocol with the data that is available.
Both are far beyond the Para-Lane yaw range (±15°): median rotation to the nearest training camera is ~60° (rot60)
and ~79° (L3; median distance ~12 m).

## Pipeline
* **Metric, z-up frame.** Each COLMAP model is aligned to the scene's `geo_registration.txt` positions with a
  similarity (Umeyama): 6.2–7.7 m per SfM unit, median residual 0.24–0.33 m; camera up then points to +z.
* **Movable-object masks.** EUVS uses Grounded-SAM-2 masks, which are not shipped; we use YOLO11m-seg instance masks
  (person, bicycle, car, motorcycle, bus, train, truck; conf 0.25; dilated 9 px like the Para-Lane masks), used for the
  training loss and pasted from the real image before every metric.
* **Splat.** The Para-Lane model (gsplat, SH3, sky environment map, masked 0.8·L1 + 0.2·D-SSIM, anisotropy
  regularizer, default densification) initialized from the EUVS SfM points (no LiDAR, so no depth loss), 15k
  iterations at 960×608.
* **Fixers.** The same seven arms as on Para-Lane, unchanged (all were trained on Para-Lane or are generic, so this is
  also a transfer test to another city, camera and resolution). Difix-ref and our Difix-ref reimplementation get as
  reference the training image minimizing center distance [m] + 0.1 × viewing angle [deg].
* **Metrics** as on Para-Lane (`src/eval/evaluate.py`): PSNR/SSIM/LPIPS (movable objects pasted), road LPIPS,
  lane-paint P/R/F1, texture ratio, YOLO fake vehicles and static F1; CLRNet NTL-IoU / lane F1 vs. CLRNet on the real
  view **only for the front camera** (a CULane lane detector is meaningless on ±60° side cameras, so rot60 has no lane
  agreement). CIs: bootstrap over (scene, camera) sequences.

## Outcome notes
* CLRNet (CULane weights) finds lanes on only 4 % of the REAL MARS front-camera test views (US city streets and
  intersections), so lane agreement on EUVS is defined on only 7–11 frames per method and is not informative
  (`results/euvs/rows_L3.json`: `n_lanes_real`); lane-paint P/R/F1 is the usable road-marking metric there.
* The median nearest-training-camera statistics in `results/euvs/summary.md` are by camera-center distance
  (L3 5.5 m / 89°, rot60 0.3 m / 66°): in rot60 the nearest center is often another traversal's front camera.
* No EUVS fine-tune of the fixers was run (time budget); all fixers are used exactly as on Para-Lane.
