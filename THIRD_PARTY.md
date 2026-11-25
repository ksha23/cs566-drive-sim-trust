# Third-party code, weights and data

Our own code is MIT-licensed (`LICENSE`). This repository contains **no** third-party model weights, **no** dataset
files and **no** checkpoints. The table lists everything the pipeline downloads or calls, and the one file that is
a modified copy of third-party code. Licenses are as stated by each project at the commit we used; check the
upstream project before redistributing anything.

## Code inside this repository that is not ours

| File | Origin | License | Changes |
|---|---|---|---|
| `src/hugsim/ltf_e2e_seed.py` | `ltf_e2e.py` from [hyzhou404/NAVSIM](https://github.com/hyzhou404/NAVSIM) @ ca0ca7e (HUGSIM's LTF client) | Apache-2.0 (notice kept in the file header) | checkpoint chosen by `LTF_SEED` |
| `src/fixers/nv_fixer/te_shim/` | written by us; re-implements the small part of the [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) Python API that Cosmos-Predict2 imports (RMSNorm, DotProductAttention, apply_rotary_pos_emb) in plain PyTorch | ours (MIT); the API it mirrors is Apache-2.0 | no TE code is copied |

## Code called at run time (cloned outside the repository, not redistributed)

| Project | Commit / version | License | Used for | How we use it |
|---|---|---|---|---|
| [nv-tlabs/Difix3D](https://github.com/nv-tlabs/Difix3D) | c76edc5 | NVIDIA license (see repo; non-commercial) | released Difix / Difix-ref pipeline; official trainer for the matched comparison | pipeline imported unchanged; the official trainer is run from a patched **local** copy (`$CS566D_DATA/difix_reimpl/official_src`, 5 patches listed in `docs/difix_reimpl_validation.md`), not included here |
| [nv-tlabs/Fixer](https://github.com/nv-tlabs/Fixer) (NVIDIA Fixer) | main at download time | NVIDIA license (see repo) | NVIDIA Fixer inference and fine-tune pilot | cloned by `src/fixers/nv_fixer/setup_env.sh`; imported with our TE shim, official `tokenizer.patch` applied at run time |
| [hyzhou404/HUGSIM](https://github.com/hyzhou404/HUGSIM) | 62c690d | MIT | simulator, closed-loop evaluation, HD-Score | imported unchanged; our loop `src/hugsim/closed_loop_fix.py` re-implements its `create_gym_env` loop |
| [hyzhou404/NAVSIM](https://github.com/hyzhou404/NAVSIM) | ca0ca7e | Apache-2.0 | LTF policy client | pixi env; one file copied (above) |
| [Turoad/CLRNet](https://github.com/Turoad/CLRNet) | 7269e9d | Apache-2.0 | lane detector (CULane DLA34 weights) | imported unchanged; NMS op compiled locally |
| [wangyuanbiubiubiu/FaithFusion](https://github.com/wangyuanbiubiubiu/FaithFusion) | main at download time | see repo | EIG maps (its modified rasterizer) | built unchanged by `scripts/14_faithfusion.sh setup`; our `src/progressive/faithfusion/eig.py` re-implements their EIG procedure (no code copied) |
| [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) | 1.4.0 (pip) | Apache-2.0 | Gaussian-splat rasterization | pip package |
| [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) | 8.x (pip), `yolo11m.pt` | AGPL-3.0 | vehicle detector for the hallucination and NTA-IoU metrics | pip package, weights downloaded by the user |
| diffusers, transformers, lpips, torchmetrics, bitsandbytes, ... | `envs/*.txt` | their own (Apache-2.0 / BSD / MIT) | | pip packages |

## Model weights (downloaded by the user, never committed)

| Weights | Source | License |
|---|---|---|
| Difix, Difix-ref | HF `nvidia/difix`, `nvidia/difix_ref` | NVIDIA license on the model cards (non-commercial) |
| SD-Turbo (initialization of our Difix reimplementation) | HF `stabilityai/sd-turbo` | Stability AI community license |
| NVIDIA Fixer | HF `nvidia/Fixer` | NVIDIA Open Model license |
| LTF (NAVSIM) | HF `autonomousvision/navsim_baselines` | Apache-2.0 |
| CLRNet CULane DLA34 | CLRNet GitHub releases | Apache-2.0 |
| YOLO11m | Ultralytics | AGPL-3.0 |
| Our trained checkpoints (U-Net, Difix-FT, Difix reimplementation) | not released | fine-tunes of NVIDIA / Stability weights inherit their licenses; the team has not decided on a release |

## Datasets (downloaded by the user, never committed)

See `docs/DATA_AND_LICENSES.md` for sizes and paths. Dataset-derived media that are in this repository
(`results/figures/`, `results/final/`, `results/videos/`, `results/hugsim/`, `results/faithfusion/`) carry the
dataset's license and attribution:

| Dataset | License | Derived media in this repo |
|---|---|---|
| Para-Lane (ModelScope `nizqleo/Paralane`) | CC BY-NC 4.0 | yes (figures, grids, videos; attribution in each video footer) |
| nuScenes via HUGSIM's released scenes (HF `XDimLab/HUGSIM`) | nuScenes terms of use, CC BY-NC-SA 4.0 | yes (HUGSIM renders in `results/hugsim/`, two videos) |
| EUVS benchmark / MARS (HF `ai4ce/EUVS-Benchmark`) | CC BY-NC-SA 4.0 | yes (one video) |
| MV2 (HF `sanjay810/mv2-dataset`) | academic research use only | **no** (MV2 frames are git-ignored; numbers only) |
| DL3DV-10K sample (HF `DL3DV/DL3DV-10K-Sample`, gated) | DL3DV terms of use (no redistribution) | **no** (numbers only) |
