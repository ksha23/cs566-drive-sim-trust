"""HUGSIM closed loop with an optional image fixer between the simulator and the driving policy.

Same loop as HUGSIM's closed_loop.py (create_gym_env), plus:
  * paths from our machine (scenes, 3DRealCar, HUGSIM's released trajdata map cache), no zsh launcher;
  * the policy client (LTF from hyzhou404/NAVSIM, released NAVSIM weights ltf_seed_0.ckpt) is started with bash on the
    same GPU;
  * if --fixer != none, the 3 front cameras the policy consumes (CAM_FRONT/_LEFT/_RIGHT) are sent to a fixer server
    (fixer_server.py, Difix env) and the FIXED images are what the policy receives (the collision / route checks use the
    simulator state and are unaffected). Saved video = what the policy saw.
  * reference-view fixers (--fixer_args with --refs, from render_refs.py): each request also carries the scene and the
    ego's ground (x, z) position, from which the server picks the nearest recorded frame as reference. Our Difix
    re-implementation (--fixer difix_reimpl --fixer_args "--difix_reimpl best.pt --refs ...") runs in its own venv.

Run in HUGSIM's pixi env:
  (see scripts/12_hugsim_closed_loop.sh) cd $CS566D_SMOKE/HUGSIM && pixi run python <repo>/src/hugsim/closed_loop_fix.py \
      --scenario $CS566D_DATA/hugsim/scenarios/nuscenes/scene-0071-easy-00.yaml --fixer none --out <dir>
"""
import argparse, json, os, pickle, subprocess, sys, time
SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
from common.config import HUGSIM_ROOT as HUGSIM, HUGSIM_DATA, SMOKE, REPO, DATA
sys.path.insert(0, HUGSIM); os.chdir(HUGSIM)
import numpy as np
import gymnasium
import hugsim_env  # noqa: F401
from omegaconf import OmegaConf
import open3d as o3d
import sim.utils.plan as plan_mod
from sim.utils.sim_utils import traj2control, traj_transform_to_global
from sim.utils.score_calculator import hugsim_evaluate

D = HUGSIM_DATA
CACHE = f"{D}/nusc_map_cache"
FRONT = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]

# HUGSIM hard-codes cache_location="~/.unified_data_cache"; point trajdata at the released cache instead.
_UD = plan_mod.UnifiedDataset
def _patched_ud(*args, **kw):
    kw["cache_location"] = CACHE
    return _UD(*args, **kw)
plan_mod.UnifiedDataset = _patched_ud


def build_cfg(scenario):
    sc = OmegaConf.load(scenario)
    base = OmegaConf.load(f"{HUGSIM}/configs/sim/nuscenes_base.yaml")
    base.realcar_path = f"{D}/3DRealCar"
    base.model_base = f"{D}/scenes/nuscenes"
    base.HD_map.path = f"{D}/nuscenes_raw_unused"  # only the cache is read
    cfg = OmegaConf.merge({"scenario": sc}, {"base": base},
                          {"camera": OmegaConf.load(f"{HUGSIM}/configs/sim/nuscenes_camera.yaml")},
                          {"kinematic": OmegaConf.load(f"{HUGSIM}/configs/sim/kinematic.yaml")})
    model_path = os.path.join(cfg.base.model_base, cfg.scenario.scene_name)
    cfg.update(OmegaConf.load(os.path.join(model_path, "cfg.yaml")))
    cfg.model_path = model_path
    return cfg


def fifo(p):
    if not os.path.exists(p):
        os.mkfifo(p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--fixer", default="none"); ap.add_argument("--fixer_args", default="")
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0, help="which released LTF checkpoint (ltf_seed_{0,1,2})")
    a = ap.parse_args()
    cfg = build_cfg(a.scenario); out = a.out; os.makedirs(out, exist_ok=True)
    procs = []
    # policy client (LTF)
    nav = f"{D}/navsim_client"
    import shutil
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ltf_e2e_seed.py"), f"{nav}/ltf_e2e_seed.py")
    lf = open(f"{out}/ltf_client.log", "w")
    procs.append(subprocess.Popen(["bash", "-c", f"source {SMOKE}/hugsim_env.sh; cd {nav} && "
                                   f"LTF_SEED={a.seed} pixi run python ltf_e2e_seed.py output={out}"], stdout=lf, stderr=lf))
    fin = fout = None
    refs = "--refs" in a.fixer_args; scene = cfg.scenario.scene_name
    if a.fixer != "none":
        ff = open(f"{out}/fixer.log", "w")
        py = f"{REPO}/scripts/lib/py.sh"
        if a.fixer == "difix_reimpl":  # its own venv, as in scripts/18_more_scenes.sh
            py = (f"HF_HOME={SMOKE}/hf TORCH_HOME={SMOKE}/cache/torch XDG_CACHE_HOME={SMOKE}/cache PYTHONPATH={REPO}/src "
                  f"{DATA}/difix_reimpl/venv/bin/python")
        procs.append(subprocess.Popen(["bash", "-c", f"{py} {REPO}/src/hugsim/fixer_server.py "
                                       f"--dir {out} --method {a.fixer} {a.fixer_args}"], stdout=ff, stderr=ff))
        fin, fout = fifo(f"{out}/fix_in"), fifo(f"{out}/fix_out")
    try:
        env = gymnasium.make("hugsim_env/HUGSim-v0", cfg=cfg, output=out)
        obs_pipe, plan_pipe = fifo(f"{out}/obs_pipe"), fifo(f"{out}/plan_pipe")
        obs, info = env.reset(); done = False; cnt = 0; t0 = time.time(); fix_t = []
        save = {"type": "closeloop", "frames": []}; frames = []
        while not done:
            if fin:
                t = time.time()
                req = {c: obs["rgb"][c] for c in FRONT}
                if refs:
                    eb = info.get("ego_box")
                    req["_scene"], req["_ego"] = scene, (None if eb is None else [-eb[1], eb[0]])
                with open(fin, "wb") as f:
                    f.write(pickle.dumps(req))
                with open(fout, "rb") as f:
                    fixed = pickle.loads(f.read())
                for c in FRONT:
                    obs["rgb"][c] = fixed[c]
                fix_t.append(time.time() - t)
            frames.append(np.concatenate([obs["rgb"][c] for c in ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]], 1)[::2, ::2])
            with open(obs_pipe, "wb") as p:
                p.write(pickle.dumps((obs, info)))
            with open(plan_pipe, "rb") as p:
                plan_traj = pickle.loads(p.read())
            if plan_traj is None:
                print("policy returned None (crash)"); break
            acc, steer_rate = traj2control(plan_traj, info)
            obs, reward, terminated, truncated, info = env.step({"acc": acc, "steer_rate": steer_rate})
            cnt += 1; done = terminated or truncated or cnt > a.max_steps
            imu = plan_traj[:, [1, 0]]; imu[:, 1] *= -1
            save["frames"].append({"time_stamp": info["timestamp"], "is_key_frame": True, "ego_box": info["ego_box"],
                                   "obj_boxes": info["obj_boxes"], "obj_names": ["car" for _ in info["obj_boxes"]],
                                   "planned_traj": {"traj": traj_transform_to_global(imu, info["ego_box"]), "timestep": 0.5},
                                   "collision": info["collision"], "rc": info["rc"]})
        with open(obs_pipe, "wb") as p:
            p.write(pickle.dumps("Done"))
        if fin:
            with open(fin, "wb") as f:
                f.write(pickle.dumps("Done"))
        pickle.dump([save], open(f"{out}/data.pkl", "wb"))
        import imageio
        imageio.mimwrite(f"{out}/policy_view.mp4", frames, fps=4, quality=6)
        g = np.asarray(o3d.io.read_point_cloud(f"{out}/ground.ply").points)
        s = np.asarray(o3d.io.read_point_cloud(f"{out}/scene.ply").points)
        res = hugsim_evaluate([save], g, s)
        res = {k: v for k, v in res.items() if k != "details"}
        res.update(seed=a.seed, steps=cnt, wall_s=time.time() - t0, fixer=a.fixer, scenario=os.path.basename(a.scenario),
                   fix_ms_per_step=float(np.median(fix_t) * 1000) if fix_t else 0.0)
        json.dump(res, open(f"{out}/eval.json", "w"), default=float)
        print("EVAL", json.dumps(res, default=float), flush=True)
    finally:
        time.sleep(2)
        for p in procs:
            if p.poll() is None:
                p.kill()


if __name__ == "__main__":
    main()
