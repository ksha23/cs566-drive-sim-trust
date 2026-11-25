"""Fit a street splat on an EUVS [6] scene's TRAINING views and render its TEST views (real ground truth).

Protocols (both use EUVS Setting3 MARS scenes; images are only shipped for Setting3, see docs/euvs.md):
  L3       EUVS level 3, translation + rotation: the official train_set / test_set (test = a held-out traversal through
           the same place along another route, all 3 cameras)
  rot60    rotation only, EUVS Setting2 protocol (train on some cameras, test on other cameras of the SAME
           traversals) applied to MARS: train on the front camera (channel 2) of every traversal, test on the +-60 deg
           cameras (channels 1, 3) of the EUVS test traversal. Camera baseline ~0.7 m.
Splat: same model as the Para-Lane fits (gsplat, SH3, sky environment map, movable-object-masked 0.8 L1 + 0.2 D-SSIM,
scale-anisotropy regularizer, DefaultStrategy densification) but initialized from the EUVS SfM points (no LiDAR, so no
depth loss and no LiDAR sky mask), 15k iterations. Scene aligned to meters, z up (src/euvs/data.py:align_metric).

Outputs ($CS566D_DATA/euvs/out/<proto>/<scene>/): splat.pt, raw/<name>.png, real/<name>.png, fg/<name>.png,
ref/<name>.png (the Difix-ref reference: the training image with the smallest  center distance [m] + 0.1 x viewing
angle [deg]), meta.json [{scene, proto, name, channel, path, real, fg, ref, dist_m, rot_deg, fwd_deg, horizon}].

usage: scripts/lib/py.sh src/euvs/fit.py <scene_dir> <proto> [--iters 15000]
"""
import argparse, json, math, os, re, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from scipy.spatial import cKDTree
from torchmetrics.image import StructuralSimilarityIndexMeasure

from common.config import DATA
from splats.gs import SkyEnv
from euvs.data import load_scene, align_metric, load_image, make_dyn_masks, load_dyn

C0 = 0.28209479177387814


def pixel_dirs(K, W, H):
    v, u = torch.meshgrid(torch.arange(H, device="cuda") + 0.5, torch.arange(W, device="cuda") + 0.5, indexing="ij")
    return torch.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], torch.ones_like(u)], -1)


def render(params, sky, c2w, K, W, H, sh_degree=3):
    colors = torch.cat([params["sh0"], params["shN"]], 1)
    out, alpha, info = rasterization(params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
                                     torch.sigmoid(params["opacities"]), colors, torch.linalg.inv(c2w)[None], K[None], W, H,
                                     sh_degree=sh_degree, packed=False)
    rgb, alpha = out[0], alpha[0, ..., 0]
    dw = F.normalize(pixel_dirs(K, W, H) @ c2w[:3, :3].T, dim=-1)
    return (rgb + (1 - alpha[..., None]) * sky(dw)).clamp(0, 1), alpha, info


def split(sc, proto):
    ch = lambda n: int(re.search(r"channel_(\d+)", n).group(1))
    tr_ = lambda n: re.search(r"trav_(\d+)", n).group(1)
    if proto == "L3":
        tr = [n for n in sc["names"] if sc["split"][n] == "train"]; te = [n for n in sc["names"] if sc["split"][n] == "test"]
    elif proto == "rot60":
        test_trav = {tr_(n) for n in sc["names"] if sc["split"][n] == "test"}
        tr = [n for n in sc["names"] if ch(n) == 2]
        te = [n for n in sc["names"] if ch(n) != 2 and tr_(n) in test_trav]
    else:
        raise ValueError(proto)
    return tr, te, ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir"); ap.add_argument("proto", choices=["L3", "rot60"])
    ap.add_argument("--iters", type=int, default=15000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    name = os.path.basename(a.scene_dir.rstrip("/"))
    out = f"{DATA}/euvs/out/{a.proto}/{name}"; os.makedirs(out, exist_ok=True)
    sc = load_scene(a.scene_dir); scale = align_metric(sc); W, H = sc["size"]
    tr, te, ch = split(sc, a.proto)
    print(f"{name} {a.proto}: {len(tr)} train / {len(te)} test views, {W}x{H}, {scale:.2f} m per SfM unit, "
          f"geo residual {sc['align']['resid_med_m']:.2f} m", flush=True)
    t0 = time.time(); make_dyn_masks(sc, f"{DATA}/euvs/weights/yolo11m-seg.pt")
    print(f"dynamic masks {time.time()-t0:.0f}s", flush=True)
    K = {n: torch.tensor(sc["K"][n], device="cuda") for n in sc["names"]}
    c2w = {n: torch.tensor(sc["c2w"][n], device="cuda") for n in sc["names"]}
    ck = f"{out}/splat.pt"
    if not os.path.exists(ck):
        imgs = {n: torch.from_numpy(load_image(sc, n)).cuda() for n in tr}
        keep = {n: ~torch.from_numpy(load_dyn(sc, n)).cuda() for n in tr}
        cams = torch.stack([c2w[n][:3, 3] for n in tr])
        pts = torch.tensor(sc["points"], device="cuda"); rgb = torch.tensor(sc["colors"], device="cuda").float() / 255
        near = torch.cdist(pts, cams).min(1).values < 80
        pts, rgb = pts[near], rgb[near]; N = pts.shape[0]
        dist, _ = cKDTree(pts.cpu().numpy()).query(pts.cpu().numpy(), k=4)
        dist = torch.tensor(dist[:, 1:].mean(1), device="cuda").clamp(0.01, 1.0).float()
        params = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(pts), "scales": torch.nn.Parameter(torch.log(dist)[:, None].repeat(1, 3)),
            "quats": torch.nn.Parameter(F.normalize(torch.randn(N, 4, device="cuda"), dim=-1)),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.3, device="cuda"))),
            "sh0": torch.nn.Parameter(((rgb - 0.5) / C0)[:, None]), "shN": torch.nn.Parameter(torch.zeros(N, 15, 3, device="cuda"))})
        sky = SkyEnv().cuda()
        scene_scale = float((cams - cams.mean(0)).norm(dim=-1).max()) * 1.1
        lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
        opts = {k: torch.optim.Adam([params[k]], lr=v, eps=1e-15) for k, v in lrs.items()}
        sky_opt = torch.optim.Adam(sky.parameters(), lr=1e-2)
        strategy = DefaultStrategy(refine_start_iter=500, refine_stop_iter=int(a.iters * 0.6), reset_every=3000,
                                   refine_every=100, verbose=False)
        strategy.check_sanity(params, opts); st = strategy.initialize_state(scene_scale=scene_scale)
        sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1 / a.iters))
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
        print(f"init {N} SfM points, scene scale {scene_scale:.1f} m", flush=True); t0 = time.time()
        for it in range(a.iters):
            n = tr[np.random.randint(len(tr))]
            gt = imgs[n].float() / 255; kp = keep[n].float()[..., None]
            img, alpha, info = render(params, sky, c2w[n], K[n], W, H, sh_degree=min(3, it // 1000))
            strategy.step_pre_backward(params, opts, st, it, info)
            l1 = ((img - gt).abs() * kp).sum() / (kp.sum() * 3 + 1)
            pm = img * kp + gt * (1 - kp)
            loss = 0.8 * l1 + 0.2 * (1 - ssim(pm.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
            sc_ = torch.exp(params["scales"]); loss = loss + 0.01 * F.relu(sc_.max(-1).values / sc_.min(-1).values - 10).mean()
            loss.backward()
            strategy.step_post_backward(params, opts, st, it, info, packed=False)
            for o in opts.values():
                o.step(); o.zero_grad(set_to_none=True)
            sky_opt.step(); sky_opt.zero_grad(set_to_none=True); sched.step()
            if it % 1000 == 0:
                print(f"it {it} loss {loss.item():.4f} N {params['means'].shape[0]} t {time.time()-t0:.0f}s", flush=True)
        torch.save({"params": {k: v.detach() for k, v in params.items()}, "sky": sky.state_dict(), "train": tr, "test": te,
                    "align": sc["align"]}, ck)
        print(f"TRAIN done {time.time()-t0:.0f}s peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
        del imgs
    s = torch.load(ck, map_location="cuda"); params = s["params"]; sky = SkyEnv().cuda(); sky.load_state_dict(s["sky"])
    # reference view per test image (what a simulator built from the training drives has)
    Ct = np.stack([sc["c2w"][n][:3, 3] for n in tr]); Rt = np.stack([sc["c2w"][n][:3, :3] for n in tr])
    meta = []
    for d in ("raw", "real", "fg", "ref"):
        os.makedirs(f"{out}/{d}", exist_ok=True)
    with torch.no_grad():
        for n in te:
            R, C = sc["c2w"][n][:3, :3], sc["c2w"][n][:3, 3]
            dist = np.linalg.norm(Ct - C, axis=1)
            rot = np.degrees(np.arccos(np.clip((np.einsum("kij,ij->k", Rt, R) - 1) / 2, -1, 1)))
            fwd = np.degrees(np.arccos(np.clip(Rt[:, :, 2] @ R[:, 2], -1, 1)))
            j = int(np.argmin(dist + 0.1 * rot)); jn = int(dist.argmin())
            stem = os.path.splitext(n)[0] + ".png"
            img = render(params, sky, c2w[n], K[n], W, H)[0]
            Image.fromarray((img.cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(f"{out}/raw/{stem}")
            Image.fromarray(load_image(sc, n)).save(f"{out}/real/{stem}")
            Image.fromarray(load_dyn(sc, n).astype(np.uint8) * 255).save(f"{out}/fg/{stem}")
            Image.fromarray(load_image(sc, tr[j])).save(f"{out}/ref/{stem}")
            pitch = math.asin(max(-1.0, min(1.0, -float(R[2, 2]))))
            Kn = sc["K"][n]
            meta.append(dict(scene=name, proto=a.proto, name=n, channel=ch(n), path=f"{a.proto}/{name}/raw/{stem}",
                             real=f"{a.proto}/{name}/real/{stem}", fg=f"{a.proto}/{name}/fg/{stem}",
                             ref=f"{a.proto}/{name}/ref/{stem}", ref_name=tr[j],
                             dist_m=float(dist[jn]), rot_deg=float(rot[jn]), fwd_deg=float(fwd[jn]),
                             ref_dist_m=float(dist[j]), ref_rot_deg=float(rot[j]),
                             horizon=float(Kn[1, 2] - Kn[1, 1] * math.tan(pitch))))
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=0)
    print(f"rendered {len(meta)} test views; nearest-train distance median {np.median([m['dist_m'] for m in meta]):.2f} m, "
          f"rotation median {np.median([m['rot_deg'] for m in meta]):.1f} deg", flush=True)


if __name__ == "__main__":
    main()
