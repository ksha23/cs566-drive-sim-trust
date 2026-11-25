# Data, weights, external code and licenses

Nothing below is in the repository; paths are relative to `CS566D_DATA` (default `~/cs566/data/D`) or
`CS566D_SMOKE` (default `~/cs566/smoke/D`).

| Item | Source | License | Where / size |
|---|---|---|---|
| Para-Lane 0315 scene_16, 17, 19–24, and scene_10, 14, 15 (test; screened: 3, 4, 11, 18) | ModelScope `nizqleo/Paralane` split ZIP (per-scene part ranges: `configs/paralane_scene_index.json`) | CC BY-NC 4.0 | `DATA/paralane/extracted` 20 GB (+ scene_16 in `SMOKE/data/paralane`) |
| Splats, pairs, fixed renders | produced by this repo | – | `DATA/{splats,pairs,fixed}` |
| Difix / Difix-ref weights, SD-Turbo | HF `nvidia/difix`, `nvidia/difix_ref`, `stabilityai/sd-turbo` | NVIDIA non-commercial / SAI community | `SMOKE/hf` (15 GB HF cache) |
| NVIDIA Fixer weights | HF `nvidia/Fixer` | NVIDIA open model license | `DATA/nv_fixer` |
| YOLO11m | Ultralytics | AGPL-3.0 | `SMOKE/weights/yolo11m.pt` |
| CLRNet + CULane DLA34/R18 weights | github.com/Turoad/CLRNet @ 7269e9d | Apache-2.0 | `DATA/lanes` 0.25 GB |
| HUGSIM code | github.com/hyzhou404/HUGSIM @ 62c690d | MIT | `SMOKE/HUGSIM` |
| HUGSIM released scenes (nuScenes 0064, 0071, 0167, 0383, 0661), scenarios, map cache, 3DRealCar assets | HF `XDimLab/HUGSIM` | MIT (nuScenes-derived: nuScenes terms) | `DATA/hugsim` 18 GB |
| LTF policy (NAVSIM) | github.com/hyzhou404/NAVSIM @ ca0ca7e, weights autonomousvision/navsim_baselines | Apache-2.0 | `DATA/hugsim/navsim_client`, `ltf_dl` |
| Difix3D code | github.com/nv-tlabs/Difix3D @ c76edc5 | NVIDIA | `SMOKE/Difix3D` |
| DL3DV-10K-Sample, 11 scenes (COLMAP + images_4; Difix reimplementation check) | HF `DL3DV/DL3DV-10K-Sample` (gated; access granted to the student's account; Benchmark / ALL-960P not authorised) | DL3DV terms of use (non-commercial research) | `DATA/dl3dv` 3 GB |
| MV2 car front camera + scooter, 50 sequences | HF `sanjay810/mv2-dataset` | academic research use | `DATA/mv2/raw` |

The repo keeps only small result files and a few figures/frames derived from the data (for the report).
