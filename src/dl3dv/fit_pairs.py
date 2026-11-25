"""Difix-style training / evaluation pairs on one DL3DV scene, the way Difix3D+ [1] curates DL3DV (Sec. 3.2):
  * sparse reconstruction: a 3DGS splat fitted on every S-th frame of the video only (S = 30 or 60), rendered at all the other
    (held-out) frames -> (render, real) pairs ("sparse");
  * model underfitting: the same fit stopped early (25% of the schedule), rendered at ALL frames ("underfit").
Reference view (for Difix-ref style models): the nearest TRAINING frame (center distance + 0.1 x angle in degrees,
distances in units of the median training-camera spacing), i.e. what the reconstruction was built from.

Camera: COLMAP OPENCV model at 3840x2160 -> images_4 (960x540) are undistorted to a pinhole K (cv2, alpha 0 crop, same
size), so the splat renders and the targets share one pinhole camera. Splat model/loss as src/euvs/fit.py (gsplat,
SH3, env-map background, 0.8 L1 + 0.2 D-SSIM, anisotropy regulariser, DefaultStrategy), SfM-point init, 6k iterations.
DL3DV scenes are static, so there are no movable-object masks (fg = empty).

Outputs ($CS566D_DATA/dl3dv/work/<scene>/): real/<frame>.png (undistorted), fg_empty.png, and per sparsity level
s<S>/{sparse/<frame>.png, underfit/<frame>.png, splat.pt, meta.json [{scene, name, idx, train, ref, sparse, underfit,
stride}], log.json (held-out PSNR)}. Two levels are used: S = 30 (mild) and S = 60 (strong artifacts).
usage: scripts/lib/py.sh src/dl3dv/fit_pairs.py <scene_dir> [--stride 30] [--iters 6000]
"""
import argparse, json, os, time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat.strategy import DefaultStrategy
from scipy.spatial import cKDTree
from torchmetrics.image import StructuralSimilarityIndexMeasure

from common.config import DATA
from splats.gs import SkyEnv
from euvs.data import read_cameras_bin, read_images_bin, read_points_bin, qvec2rot
from euvs.fit import render

C0 = 0.28209479177387814


def load(scene_dir):
    root = f"{scene_dir}/colmap"
    sp = f"{root}/colmap/sparse/0"
    cams = read_cameras_bin(f"{sp}/cameras.bin"); ims = read_images_bin(f"{sp}/images.bin")
    names = sorted(n for n in ims if os.path.exists(f"{root}/images_4/{n}"))
    c = cams[ims[names[0]]["cam"]]; assert c["model"] == "OPENCV" and len(cams) == 1, cams
    im0 = Image.open(f"{root}/images_4/{names[0]}"); W, H = im0.size
    fx, fy, cx, cy, k1, k2, p1, p2 = c["params"]; sx, sy = W / c["w"], H / c["h"]
    K0 = np.array([[fx * sx, 0, (cx + 0.5) * sx - 0.5], [0, fy * sy, (cy + 0.5) * sy - 0.5], [0, 0, 1]])
    dist = np.array([k1, k2, p1, p2])
    K, _ = cv2.getOptimalNewCameraMatrix(K0, dist, (W, H), 0, (W, H))
    maps = cv2.initUndistortRectifyMap(K0, dist, None, K, (W, H), cv2.CV_32FC1)
    c2w = {}
    for n in names:
        w2c = np.eye(4); w2c[:3, :3] = qvec2rot(ims[n]["qvec"]); w2c[:3, 3] = ims[n]["tvec"]
        c2w[n] = np.linalg.inv(w2c).astype(np.float32)
    pts, rgb = read_points_bin(f"{sp}/points3D.bin")
    return dict(root=root, names=names, K=K.astype(np.float32), maps=maps, size=(W, H), c2w=c2w, points=pts, colors=rgb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir"); ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--iters", type=int, default=6000); ap.add_argument("--underfit_frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    name = os.path.basename(a.scene_dir.rstrip("/"))
    base = f"{DATA}/dl3dv/work/{name}"; sub = f"s{a.stride}"; out = f"{base}/{sub}"
    if os.path.exists(f"{out}/meta.json"):
        print("done already", out); return
    for d in ("sparse", "underfit"):
        os.makedirs(f"{out}/{d}", exist_ok=True)
    os.makedirs(f"{base}/real", exist_ok=True)
    sc = load(a.scene_dir); W, H = sc["size"]; names = sc["names"]
    tr = names[::a.stride]; te = [n for n in names if n not in set(tr)]
    real = {}
    for n in names:
        im = cv2.remap(np.array(Image.open(f"{sc['root']}/images_4/{n}").convert("RGB")), *sc["maps"], cv2.INTER_LANCZOS4)
        real[n] = im
        if not os.path.exists(f"{base}/real/{os.path.splitext(n)[0]}.png"):
            Image.fromarray(im).save(f"{base}/real/{os.path.splitext(n)[0]}.png")
    Image.fromarray(np.zeros((H, W), np.uint8)).save(f"{base}/fg_empty.png")
    K = torch.tensor(sc["K"], device="cuda"); c2w = {n: torch.tensor(sc["c2w"][n], device="cuda") for n in names}
    imgs = {n: torch.from_numpy(real[n]).cuda() for n in tr}
    cams = torch.stack([c2w[n][:3, 3] for n in tr])
    spacing = float(torch.median((cams[1:] - cams[:-1]).norm(dim=-1)))
    extent = float((cams - cams.mean(0)).norm(dim=-1).max())
    pts = torch.tensor(sc["points"], device="cuda"); rgb = torch.tensor(sc["colors"], device="cuda").float() / 255
    near = torch.cdist(pts, cams).min(1).values < 5 * max(extent, 1e-3)
    pts, rgb = pts[near], rgb[near]; N = pts.shape[0]
    dist, _ = cKDTree(pts.cpu().numpy()).query(pts.cpu().numpy(), k=4)
    dist = torch.tensor(dist[:, 1:].mean(1), device="cuda").clamp(1e-4 * extent, 0.05 * extent).float()
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(pts), "scales": torch.nn.Parameter(torch.log(dist)[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(F.normalize(torch.randn(N, 4, device="cuda"), dim=-1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.3, device="cuda"))),
        "sh0": torch.nn.Parameter(((rgb - 0.5) / C0)[:, None]), "shN": torch.nn.Parameter(torch.zeros(N, 15, 3, device="cuda"))})
    sky = SkyEnv().cuda()
    scene_scale = extent * 1.1
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([params[k]], lr=v, eps=1e-15) for k, v in lrs.items()}
    sky_opt = torch.optim.Adam(sky.parameters(), lr=1e-2)
    strategy = DefaultStrategy(refine_start_iter=500, refine_stop_iter=int(a.iters * 0.6), reset_every=3000,
                               refine_every=100, verbose=False)
    strategy.check_sanity(params, opts); st = strategy.initialize_state(scene_scale=scene_scale)
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1 / a.iters))
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    print(f"{name}: {len(names)} frames, {len(tr)} train (every {a.stride}th) / {len(te)} held out, {W}x{H}, "
          f"{N} SfM points", flush=True)

    def save_renders(kind, which, deg):
        with torch.no_grad():
            for n in which:
                img = render(params, sky, c2w[n], K, W, H, sh_degree=deg)[0]
                Image.fromarray((img.cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(f"{out}/{kind}/{os.path.splitext(n)[0]}.png")

    t0 = time.time(); uf = int(a.iters * a.underfit_frac)
    for it in range(a.iters):
        if it == uf:
            save_renders("underfit", names, min(3, it // 1000))
            print(f"underfit renders at it {it}", flush=True)
        n = tr[np.random.randint(len(tr))]
        gt = imgs[n].float() / 255
        img, alpha, info = render(params, sky, c2w[n], K, W, H, sh_degree=min(3, it // 1000))
        strategy.step_pre_backward(params, opts, st, it, info)
        loss = 0.8 * (img - gt).abs().mean() + 0.2 * (1 - ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        sc_ = torch.exp(params["scales"]); loss = loss + 0.01 * F.relu(sc_.max(-1).values / sc_.min(-1).values - 10).mean()
        loss.backward()
        strategy.step_post_backward(params, opts, st, it, info, packed=False)
        for o in opts.values():
            o.step(); o.zero_grad(set_to_none=True)
        sky_opt.step(); sky_opt.zero_grad(set_to_none=True); sched.step()
        if it % 1000 == 0:
            print(f"it {it} loss {loss.item():.4f} N {params['means'].shape[0]} t {time.time()-t0:.0f}s", flush=True)
    save_renders("sparse", te, 3)
    torch.save({"params": {k: v.detach() for k, v in params.items()}, "sky": sky.state_dict(), "train": tr}, f"{out}/splat.pt")
    # references: nearest training frame
    Ct = np.stack([sc["c2w"][n][:3, 3] for n in tr]); Rt = np.stack([sc["c2w"][n][:3, :3] for n in tr])
    meta, ps = [], []
    for k, n in enumerate(names):
        R, C = sc["c2w"][n][:3, :3], sc["c2w"][n][:3, 3]
        d = np.linalg.norm(Ct - C, axis=1) / max(spacing, 1e-9)
        rot = np.degrees(np.arccos(np.clip((np.einsum("kij,ij->k", Rt, R) - 1) / 2, -1, 1)))
        score = d + 0.1 * rot
        if n in tr:
            score[tr.index(n)] = 1e9  # never the frame itself
        j = int(score.argmin()); stem = os.path.splitext(n)[0]
        m = dict(scene=name, name=stem, idx=k, train=n in tr, ref=f"{name}/real/{os.path.splitext(tr[j])[0]}.png",
                 ref_dist_spacing=float(d[j]), ref_rot_deg=float(rot[j]), real=f"{name}/real/{stem}.png",
                 underfit=f"{name}/{sub}/underfit/{stem}.png", sparse=None if n in tr else f"{name}/{sub}/sparse/{stem}.png",
                 stride=a.stride)
        meta.append(m)
        if n not in tr:
            x = np.array(Image.open(f"{out}/sparse/{stem}.png")).astype(np.float32) / 255; y = real[n].astype(np.float32) / 255
            ps.append(float(-10 * np.log10(((x - y) ** 2).mean())))
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=0)
    json.dump(dict(scene=name, n_frames=len(names), n_train=len(tr), stride=a.stride, iters=a.iters, underfit_iter=uf,
                   heldout_psnr_mean=float(np.mean(ps)), heldout_psnr_median=float(np.median(ps)), fit_s=time.time() - t0,
                   n_gauss=int(params["means"].shape[0])), open(f"{out}/log.json", "w"), indent=1)
    print(f"DONE {name}: held-out sparse PSNR mean {np.mean(ps):.2f} dB, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
