"""Render HUGSIM's released nuScenes scenes (CAM_FRONT) at lateral offsets and project ground-truth vehicle boxes.

Runs in HUGSIM's pixi env (needs HUGSIM's gaussian_renderer + trajdata):
  (see scripts/10_hugsim_render.sh) cd $CS566D_SMOKE/HUGSIM && pixi run python <repo>/src/hugsim/render_offsets.py --scene scene-0071 ...

Ground truth: nuScenes vehicle annotations (all vehicles, parked and moving) from HUGSIM's released trajdata map cache
(nusc_map_cache/<scene>/agent_data_dt0.10.feather + scene_metadata extents), mapped into HUGSIM's world with the
scene's meta_data["inv_pose"] (checked: the moving actor's trajdata track lands within ~1 m of HUGSIM's own
dynamic-actor pose). The 8 box corners are projected into each (offset) camera -> 2D box, clipped to the image.
Kept: all corners in front of the camera (z > 0.5 m), center depth < 60 m, clipped area >= 400 px^2 and >= 40% of the
unclipped box inside the image. Occlusion: "vis" = fraction of the box's central 60% where the simulator's own rendered
depth is not in front of the object nearest corner by more than 1 m + 15% (expected-depth tolerance); NTA-IoU uses boxes with vis >= 0.5 (same for every method).

Output: <out>/<scene>/off{+d}/<idx>.png and <out>/<scene>/index.json with per-image GT boxes.
"""
import argparse, glob, json, os, sys
import numpy as np
import torch
import dill
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common.config import HUGSIM_ROOT as HUGSIM, HUGSIM_DATA, SMOKE
sys.path.insert(0, HUGSIM)


def load_scene(sp):
    from gaussian_renderer import GaussianModel
    from scene.obj_model import ObjModel
    g = GaussianModel(3, affine=True)
    params, _ = torch.load(f"{sp}/scene.pth", weights_only=False); g.restore(params, None)
    dyn = {}
    for p in glob.glob(f"{sp}/dynamic_*.pth"):
        tid = os.path.basename(p)[8:-4]; o = ObjModel(3, feat_mutable=False)
        mp, _ = torch.load(p, weights_only=False); o.restore(list(mp), None); dyn[tid] = o
    return g, dyn


def gt_vehicles(cache, scene):
    from trajdata import AgentType
    md = dill.load(open(f"{cache}/{scene}/scene_metadata_dt0.10.dill", "rb"))
    ext = {a.name: (a.extent.length, a.extent.width, a.extent.height) for a in md.agents
           if a.type == AgentType.VEHICLE and a.name != "ego"}
    d = pd.read_feather(f"{cache}/{scene}/agent_data_dt0.10.feather")
    d = d[d.agent_id.isin(ext.keys())]
    return d, ext


def corners(x, y, z, yaw, l, w, h):
    c = np.array([[sx * l / 2, sy * w / 2, sz * h / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return c @ R.T + np.array([x, y, z])


def visibility(depth, box, zmin):
    """fraction of the box's central region where the rendered surface is NOT clearly in front of the object"""
    if depth is None:
        return None
    s = depth.shape[1] / 1.0 / 800 if depth.shape[1] != 800 else 1.0
    x1, y1, x2, y2 = box; cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 0.6, (y2 - y1) * 0.6
    sub = depth[int((cy - h / 2) * s):int((cy + h / 2) * s) + 1, int((cx - w / 2) * s):int((cx + w / 2) * s) + 1]
    if sub.size == 0:
        return None
    sub = np.where(sub <= 0.01, np.inf, sub)  # empty / sky pixels do not occlude
    return float((sub >= zmin - (1.0 + 0.15 * zmin)).mean())  # tolerance for alpha-blended (expected) depth


def project_boxes(d_t, ext, inv_pose, c2w, K, W, H, depth=None):
    w2c = np.linalg.inv(c2w) @ inv_pose  # global -> camera
    out = []
    for _, r in d_t.iterrows():
        l, w, h = ext[r.agent_id]
        P = corners(r.x, r.y, r.z, r.heading, l, w, h)
        pc = (w2c[:3, :3] @ P.T).T + w2c[:3, 3]
        if (pc[:, 2] < 0.5).any():
            continue
        cen = pc.mean(0)
        if cen[2] > 60:
            continue
        uv = (K[:3, :3] @ pc.T).T; uv = uv[:, :2] / uv[:, 2:3]
        x1, y1 = uv.min(0); x2, y2 = uv.max(0)
        full = (x2 - x1) * (y2 - y1)
        cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        area = (cx2 - cx1) * (cy2 - cy1)
        if area < 400 or area < 0.4 * full:
            continue
        b = [float(cx1), float(cy1), float(cx2), float(cy2)]
        out.append(dict(id=r.agent_id, box=b, depth=float(cen[2]), speed=float(np.hypot(r.vx, r.vy)),
                        vis=visibility(depth, b, float(pc[:, 2].min()))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--scene_root", default=f"{HUGSIM_DATA}/scenes/nuscenes")
    ap.add_argument("--cache", default=f"{HUGSIM_DATA}/nusc_map_cache/nusc_trainval")
    ap.add_argument("--out", default=f"{HUGSIM_DATA}/renders")
    ap.add_argument("--n_frames", type=int, default=30)
    ap.add_argument("--offsets", default="0,1,2,3,-1,-2,-3")
    ap.add_argument("--real_root", default=None, help="optional nuScenes image folder (scene-0383 only) to save real frames")
    ap.add_argument("--gt_only", action="store_true", help="recompute GT boxes into an existing index.json (CPU, no render)")
    a = ap.parse_args()
    sp = f"{a.scene_root}/{a.scene}"
    if not os.path.exists(sp):
        sp = f"{SMOKE}/data/hugsim_scenes/{a.scene}"
    M = json.load(open(f"{sp}/meta_data.json")); inv_pose = np.array(M["inv_pose"])
    fr = [f for f in M["frames"] if "/CAM_FRONT/" in f["rgb_path"]]
    pick = np.linspace(5, len(fr) - 6, a.n_frames).round().astype(int)
    d, ext = gt_vehicles(a.cache, a.scene)
    offs = [int(o) for o in a.offsets.split(",")]
    od = f"{a.out}/{a.scene}"
    if a.gt_only:
        index = json.load(open(f"{od}/index.json"))
        for rec in index:
            f = fr[rec["idx"]]; K = np.array(f["intrinsics"]); c2w0 = np.array(f["camtoworld"])
            d_t = d[d.scene_ts == int(round(f["timestamp"] / 0.1))]
            for o, v in rec["views"].items():
                c2w = c2w0.copy(); c2w[:3, 3] += int(o) * c2w[:3, 0]
                dp = f"{a.out}/{v['path'][:-4]}_depth.npy"
                dep = np.load(dp).astype(np.float32) if os.path.exists(dp) else None
                v["gt"] = project_boxes(d_t, ext, inv_pose, c2w, K, f["width"], f["height"], dep)
        json.dump(index, open(f"{od}/index.json", "w")); print(a.scene, "GT recomputed"); return
    from gaussian_renderer import render
    from scene.cameras import Camera
    g, dyn = load_scene(sp)
    for o in offs:
        os.makedirs(f"{od}/off{o:+d}", exist_ok=True)
    bg = torch.zeros(3, device="cuda"); index = []
    for i in pick:
        f = fr[i]; K = np.array(f["intrinsics"]); c2w0 = np.array(f["camtoworld"]); W, H = f["width"], f["height"]
        ts = int(round(f["timestamp"] / 0.1)); d_t = d[d.scene_ts == ts]
        dynamics = {k: torch.tensor(v, dtype=torch.float32, device="cuda") for k, v in dict(f["dynamics"]).items() if k in dyn}
        rec = dict(scene=a.scene, idx=int(i), timestamp=f["timestamp"], W=W, H=H, views={})
        if a.real_root:
            rp = os.path.join(a.real_root, f["rgb_path"])
            if os.path.exists(rp):
                os.makedirs(f"{od}/real", exist_ok=True)
                Image.open(rp).convert("RGB").save(f"{od}/real/{i:05d}.png"); rec["real"] = f"{a.scene}/real/{i:05d}.png"
        for o in offs:
            c2w = c2w0.copy(); c2w[:3, 3] += o * c2w[:3, 0]
            vp = Camera(width=W, height=H, image=np.zeros((H, W, 3)), K=K, c2w=c2w, image_name="",
                        timestamp=f["timestamp"], dynamics=dynamics)
            with torch.no_grad():
                r = render(viewpoint=vp, prev_viewpoint=None, pc=g, dynamic_gaussians=dyn, unicycles={}, bg_color=bg)
            img = (r["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
            p = f"{a.scene}/off{o:+d}/{i:05d}.png"; Image.fromarray(img).save(f"{a.out}/{p}")
            dep = r["depth"].squeeze().float().cpu().numpy()[::2, ::2]  # half-res rendered depth, for occlusion
            np.save(f"{a.out}/{p[:-4]}_depth.npy", dep.astype(np.float16))
            rec["views"][str(o)] = dict(path=p, gt=project_boxes(d_t, ext, inv_pose, c2w, K, W, H, dep))
        index.append(rec)
    json.dump(index, open(f"{od}/index.json", "w"))
    n = np.mean([len(r["views"]["0"]["gt"]) for r in index])
    print(f"{a.scene}: {len(index)} frames x {len(offs)} offsets, mean GT vehicles/on-path view {n:.1f}", flush=True)


if __name__ == "__main__":
    main()
