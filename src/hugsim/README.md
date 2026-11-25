# HUGSIM track (NTA-IoU + closed loop)

Environments (outside the repo): HUGSIM pixi env (`$CS566D_SMOKE/HUGSIM`, source `$CS566D_SMOKE/hugsim_env.sh`),
project env (`scripts/lib/py.sh`), NAVSIM/LTF client pixi env (`$CS566D_DATA/hugsim/navsim_client`, cloned from
hyzhou404/NAVSIM; its `pixi.lock` pointed every PyPI wheel at `mirrors.zju.edu.cn`, which timed out from here, so the
URLs were rewritten to `files.pythonhosted.org` — the original is kept as `pixi.lock.orig`).
Data (not in repo) under `$CS566D_DATA/hugsim/`: released scenes (XDimLab/HUGSIM `scenes/nuscenes/*.zip`),
`scenarios.zip`, `nusc_map_cache.zip` (trajdata cache: HD map + nuScenes agent tracks, used both by the simulator and for
our NTA-IoU ground truth), the 25 3DRealCar assets used by our scenarios, LTF weights (autonomousvision/navsim_baselines
`ltf/ltf_seed_{0,1,2}.ckpt`).

| file | what |
|---|---|
| `render_offsets.py` | render CAM_FRONT at 0/±1/±2/±3 m + project nuScenes vehicle boxes (HUGSIM env) |
| `scripts/10_hugsim_render.sh` | GPU job wrapper for the above |
| `nta_iou.py` | apply fixers (raw / Difix / Difix-ref / Difix-FT / our U-Nets) + YOLO11 + NTA-IoU with bootstrap CIs |
| `scripts/11_hugsim_nta.sh` | GPU job wrapper |
| `plot_nta.py` | NTA-IoU-vs-offset plot + example panel |
| `closed_loop_fix.py` | HUGSIM closed loop with a fixer between simulator and policy (HUGSIM env) |
| `fixer_server.py` | fixer process (Difix env) talking to the simulator through named pipes |
| `ltf_e2e_seed.py` | NAVSIM LTF client (copy of `ltf_e2e.py`, checkpoint seed selectable) |
| `scripts/12_hugsim_closed_loop.sh` | GPU job: run scenarios × seeds for one fixer (resumable) |
| `summarize_closed_loop.py` | HD-Score and components, std / bootstrap CI / paired deltas |
| `route_deviation.py` | distance of the ego from the recorded path at every policy step (CPU; reads `data.pkl` + `ground_param.pkl`) |

Re-run with new fixer checkpoints:
```
# NTA-IoU (renders are cached; fixer outputs are cached per method name -> use a new name or --recompute)
$GJ scripts/11_hugsim_nta.sh --methods raw,difix,difix_ref,difix_ft,ours_best \
    --unet ours_best=$CS566D_DATA/fixers/<run>/best.pt --difix_ft $CS566D_DATA/difix_ft/best_unet.pt
scripts/lib/py.sh src/hugsim/plot_nta.py
# closed loop (25 scenarios of scenes 0064/0071/0167/0383/0661; SEEDS = LTF checkpoints)
SC=$(ls $CS566D_DATA/hugsim/scenarios/nuscenes/scene-{0383,0071,0661,0064,0167}-*.yaml | tr '\n' ' ')
SEEDS="0 1 2" $GJ scripts/12_hugsim_closed_loop.sh ours_best "$SC" --unet ours_best=$CS566D_DATA/fixers/<run>/best.pt
SEEDS="0" $GJ scripts/12_hugsim_closed_loop.sh difix_ft "$SC" --difix_ft $CS566D_DATA/difix_ft/best_unet.pt
scripts/13_hugsim_reproduction.sh
```
