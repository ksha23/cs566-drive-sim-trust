# HUGSIM closed loop (policy level): what ran and what did not

## What runs
* **Simulator**: HUGSIM (commit 62c690d) with its released nuScenes scenes 0064, 0071, 0167, 0383 and 0661, and all 5 released
  scenarios per scene (easy / medium-00 / medium-01 / hard / extreme; 25 scenarios in all). The scenarios insert the released
  3DRealCar vehicles.
  * HUGSIM builds its HD map / agent routes from the released trajdata cache (`nusc_map_cache.zip`); no raw nuScenes data
    is needed.
  * `UnifiedDataset(cache_location=...)` is patched from our wrapper, because HUGSIM hard-codes `~/.unified_data_cache`.
  * HUGSIM's scenario files point at `<car>/postprocess/shadow.pth`, while the release ships `<car>/gs.pth` + `wlh.json`.
    We symlinked `shadow.pth/{gs.pth,wlh.json}` to those files.
* **Policy**: LTF (Latent TransFuser), from the HUGSIM NAVSIM client (hyzhou404/NAVSIM), with the released NAVSIM weights
  `ltf_seed_{0,1,2}.ckpt`.
  * The client's `pixi.lock` pins every wheel to `mirrors.zju.edu.cn`, which timed out repeatedly from sbel. We rewrote the
    URLs to `files.pythonhosted.org`, and the install then took <1 min.
  * The simulator and the client run on the **same 3090**; peak use was about 4 GB, or about 12 GB with Difix loaded.
  * `closed_loop_fix.py` re-implements HUGSIM's `create_gym_env` loop. It launches the client with bash, because HUGSIM's
    launcher needs zsh and hard-coded NAS paths.
* **Fixer hook**: a fixer server runs in the Difix env and talks to the loop through named pipes.
  * It fixes the three cameras that LTF consumes (FRONT, FRONT_LEFT, FRONT_RIGHT) before the observation is sent to the policy.
  * Collision and route checks use the simulator state, so the fixer does not affect them.
* **Metric**: HUGSIM's own `hugsim_evaluate`.
  * HD-Score = RC × mean over time of NC·DAC·(5·TTC + 2·C)/7.
  * RC is route completion; NC is no collision; DAC is drivable-area compliance; TTC is time to collision; C is comfort.

## Results (mean over runs; 95% bootstrap CI over runs; `closed_loop_summary.csv`, per run: `closed_loop_runs.json`; final table: `../final/table4_closed_loop.csv`)
Each fixer ran on the same 25 scenarios with the same 3 LTF checkpoints (seeds 0, 1, 2) as the no-fixer baseline
(Sep 30 for Difix-FT, NVIDIA Fixer and Difix seeds 1-2; `scripts/12_hugsim_closed_loop.sh`, NVIDIA Fixer through
`fixers/registry.py`'s subprocess server). Paired Δ = fixer − no fixer on the same (scenario, seed).

| fixer into the policy | seeds | runs | HD-Score [CI] | RC | NC | complete / collision / off-route | paired Δ HD-Score [CI] | fixer ms per step (3 cams) |
|---|---|---|---|---|---|---|---|---|
| none | 0,1,2 | 75 | 0.408 [0.326, 0.486] | 0.635 | 0.603 | 23 / 46 / 6 | – | – |
| Difix, zero-shot | 0,1,2 | 75 | 0.392 [0.312, 0.472] | 0.619 | 0.567 | 24 / 47 / 4 | −0.015 [−0.062, +0.034] | 1411 |
| Difix-FT (Para-Lane fine-tune) | 0,1,2 | 75 | 0.403 [0.319, 0.484] | 0.606 | 0.581 | 23 / 43 / 9 | −0.005 [−0.054, +0.046] | 1410 |
| NVIDIA Fixer | 0,1,2 | 75 | 0.356 [0.286, 0.427] | 0.577 | 0.579 | 13 / 51 / 11 | −0.051 [−0.099, −0.003] | 472 |
| our Difix re-implementation (nearest recorded frame as reference) | 0,1,2 | 75 | 0.385 [0.302, 0.467] | 0.619 | 0.546 | 24 / 43 / 8 | −0.023 [−0.074, +0.028] | 1654 |
| our U-Net, selected `base_s0` | 0 | 23** | 0.254 [0.159, 0.361] | 0.464 | 0.464 | 0 / 19 / 4 | −0.205 [−0.350, −0.065] | 222 |
| our U-Net v0 | 0 | 25 | 0.114 [0.055, 0.186] | 0.285 | 0.341 | 0 / 19 / 6 | −0.343 [−0.481, −0.211] | 191 |

Per LTF seed (HD-Score, 25 runs each): none 0.457 / 0.306 / 0.460; Difix 0.453 / 0.285 / 0.439; Difix-FT 0.504 / 0.317 /
0.389; NVIDIA Fixer 0.306 / 0.316 / 0.447. The spread across LTF seeds is larger than the effect of any fixer run with all three seeds (our U-Nets, run with
seed 0 only, are the exception). With seed 0
only, Difix-FT looked better (+0.046) and NVIDIA Fixer much worse (−0.151); with 3 seeds Difix and Difix-FT are
indistinguishable from no fixer and NVIDIA Fixer is slightly worse (CI just clear of 0). NVIDIA Fixer's outputs look
clean in the policy videos (`<run>/policy_view.mp4`); we did not find a cause for the drop beyond seed 0.

\** The selected U-Net's run was time-boxed to 15 min of GPU at the end; it did not reach scene-0661-medium-00 or scene-0661-medium-01.

"Our U-Net v0" is the first-night checkpoint (`~/cs566/data/D/fixer/fixer.pt`: 21 min of training, Para-Lane cross-lane pairs
only, no identity pairs). On nuScenes frames (overcast, a different camera) it collapses the image into Para-Lane-like washed-out
textures (see `nta_example_*.jpg`), and LTF then drives off the road. The selected U-Net (`base_s0`) was retrained on 5 Para-Lane scenes with identity pairs. It is less destructive than v0,
but it still halves HD-Score, and it never completes a route. **This is a negative transfer result for our fixer at the policy level.**

## What was not run, and why
* **UniAD / VAD clients (hyzhou404/UniAD_SIM, VAD_SIM): not attempted.**
  * They need the original UniAD stack: python 3.8, torch 1.9.1+cu111, mmcv-full 1.4.0, mmdet 2.14, mmseg 0.14.1, and
    mmdet3d 0.17.1 with its CUDA ops compiled.
  * sbel has only the CUDA 12.6 toolkit (`/usr/local/cuda-12.6`) and no conda. Compiling mmdet3d for cu111 would first need
    an 11.x nvcc installed through pixi/conda-forge, which we did not attempt.
  * UniAD_SIM's launcher (`tools/e2e.sh`) also hard-codes zsh + conda + NAS paths.
  * The GPU was heavily contended by three projects, so we spent the time on LTF with 3 seeds × 25 scenarios instead.
* **Released Difix-ref in closed loop: not run.** A reference-view fixer was run with our Difix re-implementation instead
  (final round, RESULTS §19): at each step the reference is the recorded frame nearest to the vehicle's ground position
  (`src/hugsim/render_refs.py`); see the table above.
* **Our U-Nets with seeds 1 and 2: not run** (they clearly break the policy with seed 0 already).
