"""MV2 [9] cross-vehicle test: fit a splat on ONE vehicle's camera (car or scooter) and render the OTHER vehicle's
real, simultaneous poses (different lane AND different camera height), plus the training vehicle's own held-out
frames (0 m reference).

Directions:  car2scooty  train = car front camera, test = scooter front camera (the MV2 paper's T_{C->S})
             scooty2car  train = scooter, test = car
Split: training vehicle = all frames except every 8th (k % 8 == 4, held out for the on-path check); test vehicle =
every 2nd frame (50 of 100; the MV2 paper uses every 5th).
Splat: the EUVS model unchanged (src/euvs/fit.py: gsplat SH3, sky environment map, masked 0.8 L1 + 0.2 D-SSIM,
scale-anisotropy regularizer, DefaultStrategy densification, 15k iterations), initialized from points triangulated
from the TRAINING vehicle's views only (src/mv2/data.py), in the metric z-up frame.

Outputs ($CS566D_DATA/mv2/out/<dir>/<seq>/): splat.pt, raw/ real/ fg/ ref/ <v>_<k>.png, meta.json with, per test view:
  kind (onpath | cross), lat_m / fwd_m / dz_m = offset of the test camera from the nearest training camera, in that
  camera's heading frame (left +, forward +, up +), dh_m = measured camera-height difference above the road
  (test vehicle - train vehicle, per sequence), rot_deg, ref (nearest training image by center distance [m] +
  0.1 x viewing angle [deg], the EUVS rule), horizon row.

usage: scripts/lib/py.sh src/mv2/fit.py <seq> <car2scooty|scooty2car> [--iters 15000]
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat.strategy import DefaultStrategy
from scipy.spatial import cKDTree
from torchmetrics.image import StructuralSimilarityIndexMeasure

from common.config import DATA
from splats.gs import SkyEnv
from euvs.fit import render, C0
from mv2.data import load_seq, load_image, load_mask, ego_mask, make_dyn_masks, cache_dir, W, H

DIRS = {"car2scooty": ("car", "scooty"), "scooty2car": ("scooty", "car")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq"); ap.add_argument("dir", choices=list(DIRS))
    ap.add_argument("--iters", type=int, default=15000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    A, B = DIRS[a.dir]
    out = f"{DATA}/mv2/out/{a.dir}/{a.seq}"; os.makedirs(out, exist_ok=True)
    R, info = load_seq(a.seq)
    for v in (A, B):
        make_dyn_masks(a.seq, v, R[v]["ids"])
    ego = {v: ego_mask(a.seq, v) for v in (A, B)}
    idsA, idsB = R[A]["ids"], R[B]["ids"]
    tr = [j for j, k in enumerate(idsA) if k % 8 != 4]; ho = [j for j, k in enumerate(idsA) if k % 8 == 4]
    teB = [j for j, k in enumerate(idsB) if k % 2 == 0]
    c2wA = torch.tensor(R[A]["c2w"], dtype=torch.float32, device="cuda"); KA = torch.tensor(R[A]["K"], dtype=torch.float32, device="cuda")
    c2wB = torch.tensor(R[B]["c2w"], dtype=torch.float32, device="cuda"); KB = torch.tensor(R[B]["K"], dtype=torch.float32, device="cuda")
    print(f"{a.seq} {a.dir}: {len(tr)} train / {len(ho)} held-out {A} / {len(teB)} {B} views; car cam {info['h_car_m']:.2f} m, "
          f"scooter cam {info['h_scooty_m']:.2f} m above road; synced offset lat {info['sync_lat_m']:.2f} m", flush=True)
    ck = f"{out}/splat.pt"
    if not os.path.exists(ck):
        imgs = {j: torch.from_numpy(load_image(a.seq, A, idsA[j])).cuda() for j in tr}
        keep = {j: ~torch.from_numpy(load_mask(a.seq, A, idsA[j], ego[A])).cuda() for j in tr}
        cams = c2wA[tr, :3, 3]
        P = np.load(f"{cache_dir(a.seq)}/pts_{A}.npz")
        pts = torch.tensor(P["xyz"], device="cuda"); rgb = torch.tensor(P["rgb"], device="cuda").float() / 255
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
        print(f"init {N} triangulated points, scene scale {scene_scale:.1f} m", flush=True); t0 = time.time()
        for it in range(a.iters):
            j = tr[np.random.randint(len(tr))]
            gt = imgs[j].float() / 255; kp = keep[j].float()[..., None]
            img, alpha, inf = render(params, sky, c2wA[j], KA[j], W, H, sh_degree=min(3, it // 1000))
            strategy.step_pre_backward(params, opts, st, it, inf)
            l1 = ((img - gt).abs() * kp).sum() / (kp.sum() * 3 + 1)
            pm = img * kp + gt * (1 - kp)
            loss = 0.8 * l1 + 0.2 * (1 - ssim(pm.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
            sc_ = torch.exp(params["scales"]); loss = loss + 0.01 * F.relu(sc_.max(-1).values / sc_.min(-1).values - 10).mean()
            loss.backward()
            strategy.step_post_backward(params, opts, st, it, inf, packed=False)
            for o in opts.values():
                o.step(); o.zero_grad(set_to_none=True)
            sky_opt.step(); sky_opt.zero_grad(set_to_none=True); sched.step()
            if it % 3000 == 0:
                print(f"it {it} loss {loss.item():.4f} N {params['means'].shape[0]} t {time.time()-t0:.0f}s", flush=True)
        torch.save({"params": {k: v.detach() for k, v in params.items()}, "sky": sky.state_dict(),
                    "train": [idsA[j] for j in tr], "info": info}, ck)
        print(f"TRAIN done {time.time()-t0:.0f}s peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
        del imgs
    s = torch.load(ck, map_location="cuda"); params = s["params"]; sky = SkyEnv().cuda(); sky.load_state_dict(s["sky"])
    TA = R[A]["c2w"]; Ct = TA[tr, :3, 3]; Rt = TA[tr, :3, :3]
    dh = (info["h_scooty_m"] - info["h_car_m"]) * (1 if B == "scooty" else -1)
    for d in ("raw", "real", "fg", "ref"):
        os.makedirs(f"{out}/{d}", exist_ok=True)
    meta = []
    views = [("onpath", A, j) for j in ho] + [("cross", B, j) for j in teB]
    with torch.no_grad():
        for kind, v, j in views:
            T = R[v]["c2w"][j]; Rm, C = T[:3, :3], T[:3, 3]; k = R[v]["ids"][j]
            dist = np.linalg.norm(Ct - C, axis=1)
            rot = np.degrees(np.arccos(np.clip((np.einsum("kij,ij->k", Rt, Rm) - 1) / 2, -1, 1)))
            jr = int(np.argmin(dist + 0.1 * rot)); jn = int(dist.argmin())
            f = Rt[jn][:, 2].copy(); f[2] = 0; f /= np.linalg.norm(f); left = np.cross([0, 0, 1.0], f); dv = C - Ct[jn]
            stem = f"{v}_{k:06d}.png"
            c2w = torch.tensor(T, dtype=torch.float32, device="cuda"); K = torch.tensor(R[v]["K"][j], dtype=torch.float32, device="cuda")
            img = render(params, sky, c2w, K, W, H)[0]
            Image.fromarray((img.cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(f"{out}/raw/{stem}")
            Image.fromarray(load_image(a.seq, v, k)).save(f"{out}/real/{stem}")
            Image.fromarray(load_mask(a.seq, v, k, ego[v]).astype(np.uint8) * 255).save(f"{out}/fg/{stem}")
            Image.fromarray(load_image(a.seq, A, idsA[tr[jr]])).save(f"{out}/ref/{stem}")
            pitch = math.asin(max(-1.0, min(1.0, -float(Rm[2, 2])))); Kn = R[v]["K"][j]
            p = f"{a.dir}/{a.seq}"
            meta.append(dict(scene=a.seq, dir=a.dir, kind=kind, vehicle=v, frame=k, name=stem, channel=0,
                             path=f"{p}/raw/{stem}", real=f"{p}/real/{stem}", fg=f"{p}/fg/{stem}", ref=f"{p}/ref/{stem}",
                             ref_frame=idsA[tr[jr]], dist_m=float(dist[jn]), rot_deg=float(rot[jn]),
                             lat_m=float(dv @ left), fwd_m=float(dv @ f), dz_m=float(dv[2]),
                             dh_m=0.0 if kind == "onpath" else float(dh),
                             ref_dist_m=float(dist[jr]), ref_rot_deg=float(rot[jr]),
                             horizon=float(Kn[1, 2] - Kn[1, 1] * math.tan(pitch))))
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=0)
    cr = [m for m in meta if m["kind"] == "cross"]
    print(f"rendered {len(meta)} views; cross: |lat| median {np.median([abs(m['lat_m']) for m in cr]):.2f} m, "
          f"dz median {np.median([m['dz_m'] for m in cr]):.2f} m, dh {dh:.2f} m, rot median {np.median([m['rot_deg'] for m in cr]):.1f} deg",
          flush=True)


if __name__ == "__main__":
    main()
