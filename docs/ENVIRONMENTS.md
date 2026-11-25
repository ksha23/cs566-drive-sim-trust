# Environments

| Env | Used for | Spec | How it was made |
|---|---|---|---|
| project (`SMOKE/envs/difix`) | splats, pairs, fixers, eval, HUGSIM NTA-IoU and fixer server | `envs/difix_env.txt` (pip freeze, Python 3.10, torch 2.4.1+cu121, gsplat 1.4.0, diffusers 0.25.1, bitsandbytes, lpips, ultralytics, pandas) | conda-style env from Difix3D's requirements + gsplat |
| CLRNet (`DATA/lanes/env`) | lane detection only | `envs/lanes_env.txt` | venv `--system-site-packages` over the project env + mmcv 1.7.2 (no compiled ops), addict, yapf 0.32; CLRNet NMS op built with nvcc 12.6 for sm_86 |
| HUGSIM (pixi, CUDA 11.8, tiny-cuda-nn, apex) | HUGSIM rendering / simulator | `envs/hugsim_pixi.toml`, `envs/hugsim_pixi_list.txt` | `SMOKE/hugsim_env.sh` + `pixi install` in `SMOKE/HUGSIM` (two build fixes, see smoke-test notes) |
| NAVSIM/LTF client (pixi) | driving policy | `envs/navsim_ltf_pixi.toml`, `envs/navsim_ltf_pixi_list.txt` | `pixi install` in `DATA/hugsim/navsim_client` (lock URLs moved from a ZJU mirror to PyPI) |
| Difix reimplementation (`DATA/difix_reimpl/venv`) | our Difix trainer, official trainer, FID | `envs/difix_reimpl_venv_local.txt` (packages on top of the project env) | venv `--system-site-packages` over the project env |
| NVIDIA Fixer, FaithFusion | see `docs/nv_fixer.md`, `docs/IMPLEMENTATION_NOTES_difix.md`, `docs/faithfusion.md` | | |

`scripts/lib/py.sh` = project env + `PYTHONPATH=src`. `scripts/lib/env.sh` holds all machine paths
(`CS566D_DATA`, `CS566D_SMOKE`, `CS566D_GPUJOB`).
