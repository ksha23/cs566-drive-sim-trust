"""Validation of our hybrid renderer (splats.own_render) against gsplat (splats.gs.render) and of both against the
pure-PyTorch reference (splats.torch_ref_render), on D's real fitted splats.

  1. SH: our eval_sh vs gsplat's spherical_harmonics kernel (values + gradients, degrees 0-3, random input).
  2. Full frames (960x540): for each scene, lane clip_0's splat rendered at real poses of the 0 m (held-out on-path),
     ~3 m and ~6 m lanes (4 frames per bin, evenly spaced): max / mean |diff| of RGB, alpha, depth (pixels with
     alpha > 0.5 in both), PSNR(own vs gsplat); gradient cosine (and norm ratio) for a photometric loss
     (masked L1 vs the real image) w.r.t. means, log-scales, quats, logit-opacities, sh0, shN, sky texture and the
     camera pose (c2w); screen-space mean gradients / radii that DefaultStrategy consumes; ms per frame
     (forward only, as for evaluation renders; forward+backward, as in training) and peak extra memory.
  3. 64x64 crops: reference vs gsplat vs own (values and gradients).
Writes <out>/check.json and side-by-side images <out>/vis_*.jpg.

usage: scripts/lib/py.sh src/splats/own_render_check.py <out_dir> [--scenes scene_16,scene_17] [--per_bin 4]
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common.config import DATA
from splats.paralane import W, H, clip_dir, load_cameras
from splats.gs import render as gs_render, load_splat, pixel_dirs
from splats import own_render, torch_ref_render

GROUPS = ["means", "scales", "quats", "opacities", "sh0", "shN"]


def cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def ratio(a, b):
    return float(a.double().norm() / (b.double().norm() + 1e-30))


def check_sh():
    from gsplat.cuda._wrapper import spherical_harmonics
    torch.manual_seed(0)
    dirs = torch.randn(10000, 3, device="cuda", requires_grad=True)
    co = torch.randn(10000, 16, 3, device="cuda", requires_grad=True)
    out = {}
    for d in range(4):
        a = own_render.eval_sh(d, dirs, co); b = spherical_harmonics(d, dirs, co)
        g = torch.randn_like(a)
        ga = torch.autograd.grad(a, [dirs, co], g, allow_unused=True); gb = torch.autograd.grad(b, [dirs, co], g, allow_unused=True)
        out[f"deg{d}"] = dict(max_abs=float((a - b).abs().max()), cos_dirs=cos(ga[0], gb[0]) if d else None,
                              cos_coeffs=cos(ga[1], gb[1]))
    return out


def frames(scene, per_bin):
    meta = json.load(open(f"{DATA}/pairs/{scene}/meta.json"))
    meta = [m for m in meta if m["variant"] == "full" and m["src"] == "clip_0"]
    sel = []
    for name, lo, hi in [("0m", -0.1, 1.0), ("3m", 1.0, 4.5), ("6m", 4.5, 20)]:
        b = [m for m in meta if lo < abs(m["offset_m"]) <= hi]
        for i in np.linspace(0, len(b) - 1, per_bin).round().astype(int):
            sel.append(dict(b[i], bin=name))
    return sel


def timed(fn, reps=3):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return 1000 * float(np.median(ts))


def peak(fn):
    torch.cuda.synchronize(); base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    fn(); torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2 ** 20


def loss_fn(rgb, gt, keep):
    return ((rgb - gt).abs() * keep).sum() / (keep.sum() * 3 + 1)


def grads(renderer, params, sky, c2w, K, gt, keep, cam_dirs, **kw):
    """Photometric loss gradient w.r.t. every parameter group, the sky texture and the pose; plus the screen-space
    mean gradient and radii (what DefaultStrategy uses)."""
    p = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
    c = c2w.detach().clone().requires_grad_(True)
    sky.zero_grad(set_to_none=True)
    rgb, alpha, depth, info = renderer(p, sky, c, K, cam_dirs=cam_dirs, with_depth=True, **kw)
    info["means2d"].retain_grad()
    loss_fn(rgb, gt, keep).backward()
    g = {k: p[k].grad for k in GROUPS}
    g["sky"] = sky.tex.grad.clone(); g["pose"] = c.grad[:3]
    return g, info["means2d"].grad[0], info["radii"][0]


def vis(a, b, gt, path):
    """real | gsplat | ours | |ours - gsplat| x 20 (clipped)."""
    d = ((a - b).abs().max(-1).values * 20).clamp(0, 1)
    heat = d[..., None].expand(-1, -1, 3)
    row = torch.cat([gt, b, a, heat], 1)
    Image.fromarray((row.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).resize((row.shape[1] // 2, row.shape[0] // 2)).save(path, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--scenes", default="scene_16,scene_17")
    ap.add_argument("--per_bin", type=int, default=4); ap.add_argument("--crop", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = dict(sh=check_sh(), frames=[], crops=[])
    print("SH check:", json.dumps(res["sh"]), flush=True)
    for scene in a.scenes.split(","):
        params, sky = load_splat(f"{DATA}/splats/{scene}/clip_0/splat.pt")
        N = params["means"].shape[0]
        cams = {}
        for fi, m in enumerate(frames(scene, a.per_bin)):
            if m["dst"] not in cams:
                K, ts, c2w = load_cameras(clip_dir(scene, m["dst"]))
                cams[m["dst"]] = (torch.tensor(K, device="cuda"), ts, torch.tensor(c2w, device="cuda"))
            K, ts, c2ws = cams[m["dst"]]; c2w = c2ws[ts.index(m["ts"])]
            gt = torch.from_numpy(np.array(Image.open(f"{DATA}/pairs/{m['real']}"))).cuda().float() / 255
            keep = (torch.from_numpy(np.array(Image.open(f"{DATA}/pairs/{m['fg']}"))).cuda() <= 127).float()[..., None]
            dirs = pixel_dirs(K)
            r = dict(scene=scene, dst=m["dst"], ts=m["ts"], bin=m["bin"], offset_m=m["offset_m"], n_gauss=N)
            with torch.no_grad():
                g_rgb, g_al, g_d, g_info = gs_render(params, sky, c2w, K, cam_dirs=dirs, with_depth=True)
                o_rgb, o_al, o_d, o_info = own_render.render(params, sky, c2w, K, cam_dirs=dirs, with_depth=True)
                both = (g_al > 0.5) & (o_al > 0.5)
                dd = (g_d - o_d).abs()[both]
                mse = ((g_rgb - o_rgb) ** 2).mean()
                q = lambda x: (x * 255 + 0.5).clamp(0, 255).to(torch.uint8)
                r.update(rgb_max=float((g_rgb - o_rgb).abs().max()), rgb_mean=float((g_rgb - o_rgb).abs().mean()),
                         alpha_max=float((g_al - o_al).abs().max()), alpha_mean=float((g_al - o_al).abs().mean()),
                         depth_max=float(dd.max()), depth_mean=float(dd.mean()),
                         depth_rel_mean=float((dd / g_d[both]).mean()), psnr_own_vs_gsplat=float(-10 * torch.log10(mse + 1e-20)),
                         png_pixels_differ=float((q(g_rgb) != q(o_rgb)).any(-1).float().mean()),
                         n_visible_gsplat=int((g_info["radii"] > 0).sum()), n_visible_own=o_info["n_visible"],
                         radii_equal=float((g_info["radii"] == o_info["radii"]).float().mean()),
                         n_isect_gsplat=int(g_info["isect_ids"].numel()), n_isect_own=int(o_info["isect_ids"].numel()))
                if fi % a.per_bin == 0:
                    vis(o_rgb, g_rgb, gt, f"{a.out}/vis_{scene}_{m['bin']}_{m['ts']}.jpg")
            # gradients
            gg, g_m2d, g_rad = grads(gs_render, params, sky, c2w, K, gt, keep, dirs)
            go, o_m2d, o_rad = grads(own_render.render, params, sky, c2w, K, gt, keep, dirs)
            for k in gg:
                r[f"cos_{k}"] = cos(go[k], gg[k]); r[f"normratio_{k}"] = ratio(go[k], gg[k])
            r["cos_means2d"] = cos(o_m2d, g_m2d); r["normratio_means2d"] = ratio(o_m2d, g_m2d)
            # DefaultStrategy statistic: per-Gaussian |grad of 2D mean| (normalized screen space) where radius > 0
            s = torch.tensor([W / 2.0, H / 2.0], device="cuda")
            sg, so = (g_m2d * s).norm(dim=-1) * (g_rad > 0), (o_m2d * s).norm(dim=-1) * (o_rad > 0)
            r["cos_grad2d_stat"] = cos(so, sg)
            thr = 0.0002  # DefaultStrategy.grow_grad2d
            r["grow_agree"] = float(((sg > thr) == (so > thr)).float().mean())
            r["grow_n_gsplat"], r["grow_n_own"] = int((sg > thr).sum()), int((so > thr).sum())
            del gg, go
            # speed / memory
            with torch.no_grad():
                r["ms_fwd_gsplat"] = timed(lambda: gs_render(params, sky, c2w, K, cam_dirs=dirs))
                r["ms_fwd_own"] = timed(lambda: own_render.render(params, sky, c2w, K, cam_dirs=dirs))
            P = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}

            def step(fn):
                def f():
                    rgb = fn(P, sky, c2w, K, cam_dirs=dirs, with_depth=True)[0]
                    loss_fn(rgb, gt, keep).backward()
                return f
            r["ms_fwdbwd_gsplat"] = timed(step(gs_render)); r["ms_fwdbwd_own"] = timed(step(own_render.render))
            r["mib_fwdbwd_gsplat"] = peak(step(gs_render)); r["mib_fwdbwd_own"] = peak(step(own_render.render))
            del P; sky.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            res["frames"].append(r)
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)

            # pure-PyTorch reference on crops (2 frames per bin: 0 m and ~3 m bins, first frame of each)
            if fi % a.per_bin == 0 and m["bin"] in ("0m", "3m"):
                cs = a.crop
                for name, (x0, y0) in dict(road=(W // 2 - cs // 2, int(0.8 * H)), horizon=(W // 2 - cs // 2, int(0.45 * H)),
                                           side=(int(0.85 * W), int(0.6 * H))).items():
                    Kc = K.clone(); Kc[0, 2] -= x0; Kc[1, 2] -= y0
                    cd = own_render.camera_dirs(Kc, cs, cs)
                    gtc, kc = gt[y0:y0 + cs, x0:x0 + cs], keep[y0:y0 + cs, x0:x0 + cs]
                    outs, gr = {}, {}
                    for rn, fn in [("gsplat", lambda *x, **k: crop_gsplat(*x, **k, cs=cs)),
                                   ("own", lambda *x, **k: own_render.render(*x, **k, width=cs, height=cs)),
                                   ("ref", lambda *x, **k: torch_ref_render.render(*x, **k, width=cs, height=cs))]:
                        p = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
                        c = c2w.detach().clone().requires_grad_(True); sky.zero_grad(set_to_none=True)
                        torch.cuda.synchronize(); t = time.perf_counter()
                        rgb, al, dep = fn(p, sky, c, Kc, cam_dirs=cd, with_depth=True)[:3]
                        loss_fn(rgb, gtc, kc).backward(); torch.cuda.synchronize()
                        outs[rn] = (rgb.detach(), al.detach(), dep.detach(), 1000 * (time.perf_counter() - t))
                        gr[rn] = {k: p[k].grad for k in GROUPS}; gr[rn]["pose"] = c.grad[:3]; gr[rn]["sky"] = sky.tex.grad.clone()
                    cr = dict(scene=scene, ts=m["ts"], bin=m["bin"], crop=name)
                    for x, y in [("own", "gsplat"), ("ref", "gsplat"), ("ref", "own")]:
                        (ra, aa, da, _), (rb, ab, db, _) = outs[x], outs[y]
                        bm = (aa > 0.5) & (ab > 0.5)
                        cr[f"{x}_vs_{y}"] = dict(rgb_max=float((ra - rb).abs().max()), rgb_mean=float((ra - rb).abs().mean()),
                                                 alpha_max=float((aa - ab).abs().max()),
                                                 depth_max=float((da - db).abs()[bm].max()) if bm.any() else None,
                                                 **{f"cos_{k}": cos(gr[x][k], gr[y][k]) for k in gr[x]})
                    cr.update({f"ms_{k}": v[3] for k, v in outs.items()})
                    res["crops"].append(cr)
                    print("crop", json.dumps(cr), flush=True)
                    del gr; sky.zero_grad(set_to_none=True)
        del params, sky; torch.cuda.empty_cache()
    json.dump(res, open(f"{a.out}/check.json", "w"), indent=1)
    summarize(res, a.out)


def crop_gsplat(params, sky, c2w, K, cam_dirs, with_depth, cs):
    """splats.gs.render at a crop size (gs.render is fixed to 960x540, so the same call inlined)."""
    from gsplat import rasterization
    colors = torch.cat([params["sh0"], params["shN"]], 1)
    out, alpha, info = rasterization(params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
                                     torch.sigmoid(params["opacities"]), colors, torch.linalg.inv(c2w)[None], K[None], cs, cs,
                                     sh_degree=3, packed=False, render_mode="RGB+ED")
    out, alpha = out[0], alpha[0, ..., 0]
    rgb = out[..., :3] + (1 - alpha[..., None]) * sky(F.normalize(cam_dirs @ c2w[:3, :3].T, dim=-1))
    return rgb.clamp(0, 1), alpha, out[..., 3], info


def summarize(res, out):
    import pandas as pd
    df = pd.DataFrame(res["frames"])
    keys = ["rgb_max", "rgb_mean", "alpha_max", "alpha_mean", "depth_max", "depth_mean", "depth_rel_mean",
            "psnr_own_vs_gsplat", "png_pixels_differ", "radii_equal"] + [c for c in df.columns if c.startswith("cos_")] + \
           ["ms_fwd_gsplat", "ms_fwd_own", "ms_fwdbwd_gsplat", "ms_fwdbwd_own", "mib_fwdbwd_gsplat", "mib_fwdbwd_own"]
    agg = {k: dict(mean=float(df[k].mean()), min=float(df[k].min()), max=float(df[k].max()), median=float(df[k].median()))
           for k in keys}
    by_bin = df.groupby(["scene", "bin"])[["rgb_max", "rgb_mean", "psnr_own_vs_gsplat", "cos_means", "cos_pose",
                                            "ms_fwd_gsplat", "ms_fwd_own", "ms_fwdbwd_gsplat", "ms_fwdbwd_own"]].mean()
    print(by_bin.round(4).to_string(), flush=True)
    json.dump(dict(n_frames=len(df), overall=agg, by_bin=by_bin.reset_index().to_dict("records")),
              open(f"{out}/summary.json", "w"), indent=1)
    for k in keys:
        print(f"{k:22s} mean {agg[k]['mean']:.6g}  min {agg[k]['min']:.6g}  max {agg[k]['max']:.6g}", flush=True)


if __name__ == "__main__":
    main()
