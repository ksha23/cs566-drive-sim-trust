"""Reference views for reference-view fixers in the HUGSIM closed loop: every recorded frame of the three front
cameras rendered at its recorded pose. This is the simulator's counterpart of the reference used on Para-Lane (the
real source-lane frame nearest to the target pose); nta_iou.py likewise uses the on-path render as Difix-ref's reference.
At each closed-loop step fixer_server.py --refs picks the recorded frame nearest to the ego position.

Output <out>/<scene>/<CAM>/<i:05d>.png (recorded frame i, cameras in timestamp order) and <out>/<scene>/path.npy:
the recorded front-camera path in the simulator's ground (x, z) coordinates (ground_param.pkl, as in
route_deviation.py), one row per frame i. The script checks that the frames and the path have the same count and
matching step lengths.
usage (HUGSIM pixi env, GPU): cd $CS566D_SMOKE/HUGSIM && pixi run python <repo>/src/hugsim/render_refs.py \
        --scenes scene-0064,scene-0071,scene-0167,scene-0383,scene-0661
"""
import argparse, json, os, pickle, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common.config import HUGSIM_ROOT as HUGSIM, HUGSIM_DATA
sys.path.insert(0, HUGSIM)
from hugsim.render_offsets import load_scene

FRONT = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--scene_root", default=f"{HUGSIM_DATA}/scenes/nuscenes")
    ap.add_argument("--out", default=f"{HUGSIM_DATA}/refs")
    a = ap.parse_args()
    from gaussian_renderer import render
    from scene.cameras import Camera
    for scene in a.scenes.split(","):
        sp = f"{a.scene_root}/{scene}"; od = f"{a.out}/{scene}"
        M = json.load(open(f"{sp}/meta_data.json"))
        cams = {c: sorted([f for f in M["frames"] if f"/{c}/" in f["rgb_path"]], key=lambda f: f["timestamp"]) for c in FRONT}
        cam_poses = pickle.load(open(f"{sp}/ground_param.pkl", "rb"))[0]
        path = np.stack([cam_poses[:, 0, 3], cam_poses[:, 2, 3]], -1)
        n = len(path)
        assert all(len(v) == n for v in cams.values()), (scene, n, {c: len(v) for c, v in cams.items()})
        # same frames in the same order: per-step travel of the scene-world front camera vs. the ground path
        w = np.array([np.array(f["camtoworld"])[:3, 3] for f in cams["CAM_FRONT"]])
        sw, sg = np.linalg.norm(np.diff(w, axis=0), axis=1), np.linalg.norm(np.diff(path, axis=0), axis=1)
        r = np.corrcoef(sw, sg)[0, 1] if sw.std() > 0 and sg.std() > 0 else 1.0
        print(f"{scene}: {n} frames, step-length correlation scene world vs ground path {r:.3f}", flush=True)
        assert r > 0.9, "frame order of meta_data.json and ground_param.pkl disagree"
        os.makedirs(od, exist_ok=True); np.save(f"{od}/path.npy", path)
        g, dyn = load_scene(sp); bg = torch.zeros(3, device="cuda")
        for c in FRONT:
            os.makedirs(f"{od}/{c}", exist_ok=True)
            for i, f in enumerate(cams[c]):
                p = f"{od}/{c}/{i:05d}.png"
                if os.path.exists(p):
                    continue
                W, H = f["width"], f["height"]
                dynamics = {k: torch.tensor(v, dtype=torch.float32, device="cuda") for k, v in dict(f["dynamics"]).items() if k in dyn}
                vp = Camera(width=W, height=H, image=np.zeros((H, W, 3)), K=np.array(f["intrinsics"]),
                            c2w=np.array(f["camtoworld"]), image_name="", timestamp=f["timestamp"], dynamics=dynamics)
                with torch.no_grad():
                    out = render(viewpoint=vp, prev_viewpoint=None, pc=g, dynamic_gaussians=dyn, unicycles={}, bg_color=bg)
                Image.fromarray((out["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(p)
        print(f"{scene}: references written to {od}", flush=True)


if __name__ == "__main__":
    main()
