"""Fit one Para-Lane lane clip as a Gaussian splat (gsplat).

Improvements over the smoke-test fit:
  * LiDAR initialization (merged per-clip cloud, voxelized, colored from the nearest un-occluded view,
    points that ever land on a movable-object mask are dropped),
  * LiDAR depth supervision (min-z splatted into a 1/4-res grid),
  * a learnable sky environment map + an alpha->0 penalty on a LiDAR-skyline/color sky mask,
  * movable-object masking of the photometric loss, scale-anisotropy regularizer,
  * every 8th frame held out for on-path evaluation.

Also used for
  * cycle reconstruction (--cycle_from A.pt): render A's splat at laterally shifted poses, fit a NEW splat only on
    those renders (+ LiDAR depth at the shifted poses), later rendered back at the original poses (src/pairs/render_pairs.py),
  * progressive refinement (--init_splat + --pseudo): continue training an existing splat on real views plus fixed
    pseudo-views at off-path poses (loss restricted to a validity mask: no sky, no far region), as in
    Difix3D+/ReconDreamer.

usage: scripts/lib/py.sh src/splats/fit_splat.py <scene> <clip> <out_dir> [--iters N] [--name splat] [--cycle_from p --cycle_shift 3]
                           [--init_splat p --pseudo pseudo.pt --pseudo_prob 0.4]
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from gsplat.strategy import DefaultStrategy
from torchmetrics.image import StructuralSimilarityIndexMeasure

from splats.paralane import W, H, load_clip, load_lidar
from splats.gs import SkyEnv, pixel_dirs, render, load_splat
from eval.metrics import Metrics

G = 4  # depth-grid cell size (pixels)
HOLDOUT = 8


def project(pts, c2w, K):
    w2c = torch.linalg.inv(c2w)
    pc = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    uv = pc[:, :2] / z.clamp(min=1e-3)[:, None] * torch.stack([K[0, 0], K[1, 1]]) + torch.stack([K[0, 2], K[1, 2]])
    return uv, z


def depth_grid(pts, c2w, K, zmax=80.0):
    """Min depth per GxG cell; returns grid (H/G, W/G) with 0 = invalid, and per-point (cell index, z, in-view)."""
    gh, gw = H // G, W // G
    uv, z = project(pts, c2w, K)
    inv = (z > 0.5) & (z < zmax) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    cell = (uv[:, 1] / G).long().clamp(0, gh - 1) * gw + (uv[:, 0] / G).long().clamp(0, gw - 1)
    grid = torch.full((gh * gw,), 1e9, device=pts.device)
    grid.scatter_reduce_(0, cell[inv], z[inv], reduce="amin")
    grid[grid > 1e8] = 0
    return grid.view(gh, gw), cell, z, inv, uv


def init_from_lidar(clip, tr, voxel=0.08):
    pts = torch.tensor(load_lidar(clip["scene"], clip["clip"]), device="cuda")
    cams = clip["c2w"][:, :3, 3]
    # keep a corridor around the drive
    d = torch.cdist(pts[:, :2], cams[:, :2]).min(1).values
    pts = pts[d < 60]
    q = torch.floor(pts / voxel).long(); q -= q.min(0).values
    key = (q[:, 0] * 100000 + q[:, 1]) * 10000 + q[:, 2]
    _, idx = np.unique(key.cpu().numpy(), return_index=True)
    pts = pts[torch.from_numpy(idx).cuda()]
    N = pts.shape[0]
    best_z = torch.full((N,), 1e9, device="cuda"); col = torch.zeros(N, 3, device="cuda")
    fg_hit = torch.zeros(N, dtype=torch.bool, device="cuda")
    fgK = F.max_pool2d(clip["fg"].float()[:, None], G, G)[:, 0] > 0  # fg per grid cell
    for i in tr:
        grid, cell, z, inv, uv = depth_grid(pts, clip["c2w"][i], clip["K"])
        front = inv & (z <= grid.view(-1)[cell] * 1.03 + 0.2)  # un-occluded in this view
        fg_hit |= front & fgK[i].view(-1)[cell]
        better = front & (z < best_z)
        u = uv[better].long()
        col[better] = clip["imgs"][i][u[:, 1].clamp(0, H - 1), u[:, 0].clamp(0, W - 1)].float() / 255
        best_z[better] = z[better]
    keep = (best_z < 1e8) & ~fg_hit
    print(f"lidar init: {N} voxels, {int(keep.sum())} kept ({int(fg_hit.sum())} on movable objects)", flush=True)
    return pts[keep], col[keep]


def sky_mask(img, dgrid):
    """Sky = above the topmost LiDAR return in each grid column AND sky-colored (blue or bright/gray)."""
    gh, gw = dgrid.shape
    valid = dgrid > 0
    rows = torch.arange(gh, device=img.device)[:, None].expand(gh, gw)
    top = torch.where(valid, rows, torch.full_like(rows, gh)).min(0).values  # (gw,)
    above = rows < (top[None] - 1)
    above = F.interpolate(above[None, None].float(), size=(H, W), mode="nearest")[0, 0] > 0
    x = img.float() / 255; r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1).values, x.min(-1).values
    skycol = ((b > 0.45) & (b > r + 0.05)) | ((mx > 0.8) & (mx - mn < 0.12))
    return above & skycol


class RoadSurface:
    """ViSE-style road-surface prior (our simplified version): a smooth height field z = h(x, y) on a 0.5 m grid,
    fitted to low LiDAR returns near the drive (per-cell 10th-percentile height, hole-filled and Gaussian-smoothed),
    so it carries both height and slope. Gaussians lying within 0.3 m of the surface are pulled onto it and
    flattened along its normal (extent along the normal -> ~0)."""

    def __init__(self, pts, cams, cell=0.5, band=12.0):
        from scipy.ndimage import gaussian_filter, distance_transform_edt
        P = pts.cpu().numpy(); C = cams.cpu().numpy()
        near = cKDTree(C[:, :2]).query(P[:, :2])[0]
        cam_h = np.median(C[:, 2]) - np.percentile(P[near < 3, 2], 10)
        Q = P[near < band]
        x0, y0 = Q[:, 0].min(), Q[:, 1].min()
        ix, iy = ((Q[:, 0] - x0) / cell).astype(int), ((Q[:, 1] - y0) / cell).astype(int)
        H = np.full((ix.max() + 1, iy.max() + 1), np.nan)
        order = np.lexsort((Q[:, 2], iy, ix))
        ix, iy, z = ix[order], iy[order], Q[order, 2]
        key = ix * 100000 + iy; starts = np.r_[0, np.where(np.diff(key))[0] + 1]; ends = np.r_[starts[1:], len(key)]
        for a, b in zip(starts, ends):
            if b - a >= 3:
                H[ix[a], iy[a]] = z[a + int(0.1 * (b - a))]
        # reject cells far from the expected ground height (curbs, cars, walls)
        cz = C[:, 2]; exp = np.median(cz) - cam_h
        H[np.abs(H - exp) > 1.5] = np.nan
        valid = ~np.isnan(H)
        idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        Hf = gaussian_filter(H[tuple(idx)], 2.0)
        self.valid = torch.tensor(distance_transform_edt(~valid) < 4, device="cuda")
        self.H = torch.tensor(Hf, dtype=torch.float32, device="cuda")
        self.x0, self.y0, self.cell = float(x0), float(y0), cell
        gx, gy = np.gradient(Hf, cell)
        self.N = F.normalize(torch.tensor(np.stack([-gx, -gy, np.ones_like(gx)], -1), dtype=torch.float32, device="cuda"), dim=-1)
        print(f"road surface: grid {H.shape}, {valid.mean():.2f} observed, camera height {cam_h:.2f} m", flush=True)

    def loss(self, params):
        m = params["means"]
        ix = ((m[:, 0] - self.x0) / self.cell).long(); iy = ((m[:, 1] - self.y0) / self.cell).long()
        ins = (ix >= 0) & (ix < self.H.shape[0]) & (iy >= 0) & (iy < self.H.shape[1])
        ix, iy = ix.clamp(0, self.H.shape[0] - 1), iy.clamp(0, self.H.shape[1] - 1)
        h = self.H[ix, iy]; dz = m[:, 2] - h
        sel = ins & self.valid[ix, iy] & (dz.abs() < 0.3)
        if sel.sum() == 0:
            return torch.zeros((), device="cuda")
        n = self.N[ix[sel], iy[sel]]
        q = F.normalize(params["quats"][sel], dim=-1); w, x, y, z = q.unbind(-1)
        R = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                         2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                         2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)
        s = torch.exp(params["scales"][sel])
        ext_n = torch.sqrt((((R.transpose(1, 2) @ n[..., None])[..., 0] * s) ** 2).sum(-1) + 1e-12)  # extent along normal
        return dz[sel].abs().mean() + ext_n.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene"); ap.add_argument("clip"); ap.add_argument("out")
    ap.add_argument("iters_pos", nargs="?", type=int, default=None)
    ap.add_argument("--iters", type=int, default=12000); ap.add_argument("--name", default="splat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cycle_from", default=None); ap.add_argument("--cycle_shift", type=float, default=3.0)
    ap.add_argument("--init_splat", default=None); ap.add_argument("--pseudo", default=None)
    ap.add_argument("--pseudo_prob", type=float, default=0.4)
    ap.add_argument("--renderer", default="gsplat", choices=["gsplat", "own"],
                    help="gsplat.rasterization (default) or our hybrid renderer src/splats/own_render.py")
    ap.add_argument("--road_prior", type=float, default=0.0, help="weight of the road-surface (height+slope) prior")
    a = ap.parse_args()
    scene, clipname, out = a.scene, a.clip, a.out
    iters = a.iters_pos or a.iters
    rend = render  # renderer used for training + on-path eval
    if a.renderer == "own":
        from splats.own_render import render as rend
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    clip = load_clip(scene, clipname); clip["scene"], clip["clip"] = scene, clipname
    n = len(clip["ts"]); te = np.arange(n)[np.arange(n) % HOLDOUT == HOLDOUT // 2]; tr = np.setdiff1d(np.arange(n), te)
    print(f"{scene}/{clipname}: {len(tr)} train / {len(te)} held-out frames, mode "
          f"{'cycle' if a.cycle_from else 'refine' if a.init_splat else 'fit'}", flush=True)
    t0 = time.time()
    pts, rgb = init_from_lidar(clip, tr)
    fgK = F.max_pool2d(clip["fg"].float()[:, None], G, G)[:, 0] > 0
    K = clip["K"]; cam_dirs = pixel_dirs(K)

    # ---- training views: real (lane's own train frames) and/or pseudo (renders at off-path poses) ----
    real_views, pseudo_views = [], []
    if not a.cycle_from:
        for i in tr:
            g = depth_grid(pts, clip["c2w"][i], K)[0]; g[fgK[i]] = 0
            real_views.append(dict(c2w=clip["c2w"][i], img=clip["imgs"][i], keep=~clip["fg"][i], dgrid=g,
                                   sky=sky_mask(clip["imgs"][i], g)))
    if a.cycle_from:
        sp, ssky = load_splat(a.cycle_from)
        with torch.no_grad():
            for k, i in enumerate(tr):
                c = clip["c2w"][i].clone(); c[:3, 3] += (a.cycle_shift if k % 2 == 0 else -a.cycle_shift) * c[:3, 0]
                img = (render(sp, ssky, c, K, cam_dirs=cam_dirs)[0] * 255).round().to(torch.uint8)
                g = depth_grid(pts, c, K)[0]
                pseudo_views.append(dict(c2w=c, img=img, keep=torch.ones_like(clip["fg"][i]), dgrid=g, sky=sky_mask(img, g)))
        del sp, ssky; torch.cuda.empty_cache()
    if a.pseudo:
        P = torch.load(a.pseudo, map_location="cuda")
        # optional FaithFusion-style per-pixel loss weights (src/progressive/faithfusion): weight (M,H,W) scales the
        # L1 term of each pseudo-view pixel, scalar ssim_w scales its SSIM term; absent -> plain masked loss
        W_ = P.get("weight"); ssim_w = float(P.get("ssim_w", 1.0))
        for k, (c, img, keep) in enumerate(zip(P["c2w"], P["imgs"], P["keep"])):
            g = depth_grid(pts, c, K)[0]
            pseudo_views.append(dict(c2w=c, img=img, keep=keep, dgrid=g, sky=sky_mask(img, g) & ~keep,
                                     weight=None if W_ is None else W_[k].float(), ssim_w=ssim_w))
    p_pseudo = 1.0 if a.cycle_from else (a.pseudo_prob if pseudo_views else 0.0)
    print(f"preprocessing {time.time()-t0:.0f}s; {len(real_views)} real + {len(pseudo_views)} pseudo views", flush=True)

    C0 = 0.28209479177387814
    if a.init_splat:
        ck = torch.load(a.init_splat, map_location="cuda")
        params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in ck["params"].items()})
        sky = SkyEnv().cuda(); sky.load_state_dict(ck["sky"])
    else:
        N = pts.shape[0]
        dist, _ = cKDTree(pts.cpu().numpy()).query(pts.cpu().numpy(), k=4)
        dist = torch.tensor(dist[:, 1:].mean(1), device="cuda").clamp(0.01, 1.0).float()
        params = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(pts), "scales": torch.nn.Parameter(torch.log(dist)[:, None].repeat(1, 3)),
            "quats": torch.nn.Parameter(F.normalize(torch.randn(N, 4, device="cuda"), dim=-1)),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.3, device="cuda"))),
            "sh0": torch.nn.Parameter(((rgb - 0.5) / C0)[:, None]), "shN": torch.nn.Parameter(torch.zeros(N, 15, 3, device="cuda"))})
        sky = SkyEnv().cuda()
    cams = clip["c2w"][:, :3, 3]
    scene_scale = float((cams - cams.mean(0)).norm(dim=-1).max()) * 1.1
    refine = a.init_splat is not None
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    if refine:  # continuing from a converged splat: fresh Adam steps must be small (means at ~its final LR)
        lrs = {k: v * (0.01 if k == "means" else 0.2) for k, v in lrs.items()}
    opts = {k: torch.optim.Adam([params[k]], lr=v, eps=1e-15) for k, v in lrs.items()}
    sky_opt = torch.optim.Adam(sky.parameters(), lr=1e-2)
    strategy = DefaultStrategy(refine_start_iter=500, refine_stop_iter=int(iters * 0.6),
                               reset_every=(iters + 1) if refine else 3000, refine_every=100, verbose=False)
    strategy.check_sanity(params, opts); st = strategy.initialize_state(scene_scale=scene_scale)
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1 / iters))
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    road = RoadSurface(pts, cams) if a.road_prior > 0 else None

    def save(path):
        torch.save({"params": {k: v.detach() for k, v in params.items()}, "sky": sky.state_dict(),
                    "scene": scene, "clip": clipname, "train_idx": tr.tolist(), "test_idx": te.tolist(),
                    "mode": "cycle" if a.cycle_from else "refine" if refine else "fit"}, path)

    t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    for it in range(iters):
        V = pseudo_views if np.random.rand() < p_pseudo else real_views
        v = V[np.random.randint(len(V))]
        gt = v["img"].float() / 255; keep = v["keep"].float()[..., None]
        img, alpha, depth, info = rend(params, sky, v["c2w"], K, sh_degree=3 if refine else min(3, it // 1000),
                                       cam_dirs=cam_dirs, with_depth=True)
        strategy.step_pre_backward(params, opts, st, it, info)
        pm, gm = img * keep + gt * (1 - keep), gt
        wpix = keep if v.get("weight") is None else keep * v["weight"][..., None]  # per-pixel weights (pseudo-views only)
        l1 = ((img - gt).abs() * wpix).sum() / (keep.sum() * 3 + 1)
        loss = 0.8 * l1 + 0.2 * v.get("ssim_w", 1.0) * (1 - ssim(pm.permute(2, 0, 1)[None], gm.permute(2, 0, 1)[None]))
        # LiDAR depth supervision
        dt = v["dgrid"]; dr = F.avg_pool2d(depth[None, None], G, G)[0, 0]; vd = dt > 0
        ld = ((dr[vd] - dt[vd]).abs() / dt[vd]).mean()
        wd = 0.2 if (it < iters // 2 and not refine) else 0.05
        loss = loss + wd * ld
        # sky: no Gaussians in front of the sky
        s = v["sky"]
        if s.any():
            loss = loss + 0.05 * alpha[s].mean()
        sc = torch.exp(params["scales"])
        loss = loss + 0.01 * F.relu(sc.max(-1).values / sc.min(-1).values - 10).mean()
        if road is not None and it >= 500:
            loss = loss + a.road_prior * road.loss(params)
        loss.backward()
        strategy.step_post_backward(params, opts, st, it, info, packed=False)
        for o in opts.values():
            o.step(); o.zero_grad(set_to_none=True)
        sky_opt.step(); sky_opt.zero_grad(set_to_none=True)
        sched.step()
        if it % 1000 == 0:
            print(f"it {it} loss {loss.item():.4f} l1 {l1.item():.4f} depth {ld.item():.3f} N {params['means'].shape[0]} "
                  f"t {time.time()-t0:.0f}s", flush=True)
        if it + 1 == 2000 and a.name == "splat" and road is None:
            save(f"{out}/splat_underfit.pt")  # deliberately under-fit variant (for degraded training pairs)
    print(f"TRAIN done {time.time()-t0:.0f}s peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB N={params['means'].shape[0]}", flush=True)
    save(f"{out}/{a.name}.pt")

    # on-path held-out evaluation (movable objects pasted from GT, so they don't count)
    M = Metrics()
    res = []
    with torch.no_grad():
        for i in te:
            img = rend(params, sky, clip["c2w"][i], K, cam_dirs=cam_dirs)[0]
            res.append(M(img, clip["imgs"][i].float() / 255, clip["fg"][i]))
    r = {k: float(np.mean([x[k] for x in res])) for k in res[0]}
    r.update(n_test=len(te), n_train=len(tr), n_gauss=int(params["means"].shape[0]), train_s=time.time() - t0, iters=iters)
    print("ON-PATH held-out:", json.dumps(r), flush=True)
    json.dump(r, open(f"{out}/{'onpath_metrics' if a.name == 'splat' else a.name + '_onpath_metrics'}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
