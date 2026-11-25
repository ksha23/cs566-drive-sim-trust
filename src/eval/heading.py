"""Trust vs. heading (yaw) and camera height on Para-Lane: per-image metrics and summaries (proposal step 6,
Para-Lane part). Renders: src/splats/render_deviation.py; conventions and warps: src/eval/heading_geom.py.

There is no real image at a deviated pose, so two kinds of reference are used, always side by side:

(1) PSEUDO-GROUND TRUTH = the real recorded frame warped into the deviated camera (heading_geom.homography).
    * yaw (on-path, and lateral + yaw at the real adjacent-lane poses): exact for every point, since the camera rotates
      about its optical center; only pixels whose source lies outside the recorded frame are unknown ('valid' mask).
      This makes yaw a real-ground-truth axis on Para-Lane: PSNR (valid pixels), SSIM / LPIPS (on the largest fully
      valid column band), lane paint P/R/F1, CLRNet lanes (real-frame lanes warped by H), fake vehicles (warped
      movable-object mask), static-object YOLO F1 (warped boxes).
    * height: road-plane homography, exact only on the road. Only ROAD metrics are computed against it (road PSNR /
      LPIPS on road rows, paint P/R/F1, CLRNet lanes = road paint); fake vehicles use the plane-warped movable-object
      mask dilated by 41 px vertically (objects stand on the road; their tops move more than the plane predicts).
    Every deviated row also carries ref0_* = the same method's 0-deviation output scored on the SAME scene content
    (inverse-warped evaluation region); d_k = k - ref0_k is the matched change (removes the bias of scoring a
    smaller / different image region at larger deviations).
(2) CONSISTENCY with the same method's own 0-deviation output (x0), warped by the same H:
    lane_stab_*  CLRNet lanes of x vs. H(lanes of x0)  (does the method keep the SAME lanes when the camera turns?)
    det_stab_f1  YOLO boxes of x vs. H(boxes of x0), same class, IoU >= 0.5 (yaw only; boxes are not planar)
    self_lpips   LPIPS(x, H(x0)) on the valid band (yaw) / road rows (height)
    tex_stab     road texture energy (mean |Laplacian|, road rows, movable objects excluded) of x / of x0 -- a
                 statistic, no warp; tex_ratio = the same vs. the unwarped real frame (as in the lateral table)
    n_veh        vehicle/person detections; veh_unverif = those centered where the warped real frame is unknown

Rows: results/heading/rows_<scene>_<method>.json (one file per method = checkpoint/resume unit).

usage:
  scripts/lib/py.sh src/eval/heading.py eval <scene> --methods raw,difix,...   (GPU: YOLO, LPIPS)
  scripts/lib/py.sh src/eval/heading.py summary                                 (CPU: bootstrap CIs -> results/heading/summary.csv, .md)
"""
import argparse, glob, json, os, sys
import numpy as np
import cv2
import torch
from PIL import Image

from common.config import DATA, REPO, YOLO_WEIGHTS
from eval.evaluate import marking_mask, mark_agree, tex_energy, box_iou, det_f1, STATIC, MOVABLE, ROAD_Y0
from eval.lanes.ntl_iou import raster
from eval.heading_geom import homography, warp_img, warp_lanes, warp_boxes, cond_name

HROOT = f"{DATA}/heading"
OUT = f"{REPO}/results/heading"
ZERO = cond_name(0, 0.0)
T_LANE = 30 * 960 / 1640  # CULane-scale lane thickness at W=960 (as src/eval/lanes/ntl_iou.py)


def lane_compare(lr, lg, valid, y1, t=T_LANE, thr=0.5, min_px=40):
    """ntl_iou.compare restricted to a validity mask: lanes are rasterized, cut to the valid pixels inside the CLRNet
    band (rows < y1), and lanes with < min_px pixels left are dropped on both sides."""
    from scipy.optimize import linear_sum_assignment
    shape = (y1, valid.shape[1]); v = valid[:y1]
    Mr = [m for m in (raster(l, shape, t) & v for l in lr) if m.sum() >= min_px]
    Mg = [m for m in (raster(l, shape, t) & v for l in lg) if m.sum() >= min_px]
    if not Mr and not Mg:
        return dict(ntl_iou=np.nan, lane_f1=np.nan, lane_tp=0, lane_fp=0, lane_fn=0)
    ur = np.any(Mr, 0) if Mr else np.zeros(shape, bool); ug = np.any(Mg, 0) if Mg else np.zeros(shape, bool)
    ntl = float((ur & ug).sum() / max((ur | ug).sum(), 1)); tp = 0
    if Mr and Mg:
        iou = np.array([[(a & b).sum() / max((a | b).sum(), 1) for b in Mg] for a in Mr])
        r, c = linear_sum_assignment(-iou); tp = int((iou[r, c] >= thr).sum())
    fp, fn = len(Mr) - tp, len(Mg) - tp
    return dict(ntl_iou=ntl, lane_f1=2 * tp / (2 * tp + fp + fn), lane_tp=tp, lane_fp=fp, lane_fn=fn)


def col_band(valid, frac=0.95):
    """Largest column range whose columns are >= 95 % valid (for pure yaw the invalid region is a side band; with the
    camera pitched ~5 deg the rotated frame also loses a few rows at the top/bottom of most columns, so requiring 100 %
    would keep only a sliver). The remaining invalid pixels are pasted from the prediction (img_metrics)."""
    full = valid.mean(0) >= frac
    if not full.any():
        return 0, 0
    best, cur, s = (0, 0), 0, 0
    for i, f in enumerate(np.r_[full, False]):
        if f and cur == 0:
            s = i
        cur = cur + 1 if f else 0
        if cur > best[1] - best[0]:
            best = (s, i + 1)
    return best


def psnr_mask(a, b, m):
    if m.sum() < 100:
        return np.nan
    d = (a.astype(np.float32) - b.astype(np.float32)) / 255
    return float(-10 * np.log10((d[m] ** 2).mean() + 1e-10))


class Scorer:
    def __init__(self):
        import lpips
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        from ultralytics import YOLO
        self.lp = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
        self.yolo = YOLO(YOLO_WEIGHTS)
        self._det = {}

    def det(self, key, img):
        if key not in self._det:
            r = self.yolo(np.ascontiguousarray(img[..., ::-1]), verbose=False, conf=0.3, device=0)[0].boxes
            self._det[key] = (r.xyxy.cpu().numpy(), r.cls.cpu().numpy().astype(int))
        return self._det[key]

    @torch.no_grad()
    def img_metrics(self, x, g, fg, valid):
        """LPIPS/SSIM of x vs g (uint8 crops). As src/eval/metrics.py, movable-object pixels (fg) are pasted from the
        reference into x; pixels where the reference is unknown (~valid) are pasted from x into the reference."""
        if x.shape[0] < 32 or x.shape[1] < 32:
            return np.nan, np.nan
        x = np.where((fg & valid)[..., None], g, x)
        g = np.where(valid[..., None], g, x)
        a = torch.from_numpy(x).float().cuda().permute(2, 0, 1)[None] / 255
        b = torch.from_numpy(g).float().cuda().permute(2, 0, 1)[None] / 255
        return float(self.lp(a * 2 - 1, b * 2 - 1)), float(self.ssim(a, b))


def load(p):
    return np.array(Image.open(p).convert("RGB"))


def evaluate_scene(scene, methods, force=False):
    meta = json.load(open(f"{HROOT}/{scene}/meta.json"))
    real_lanes = json.load(open(f"{DATA}/lanes/out/{scene}/real.json"))
    P = f"{DATA}/pairs"
    frames = {}
    for m in meta:
        frames.setdefault((m["src"], m["dst"], m["ts"]), []).append(m)
    S = Scorer(); os.makedirs(OUT, exist_ok=True)
    for meth in methods:
        outp = f"{OUT}/rows_{scene}_{meth}.json"
        if os.path.exists(outp) and not force:
            print("exists", outp, flush=True); continue
        root = HROOT if meth == "raw" else f"{HROOT}/fixed/{meth}"
        lanes = json.load(open(f"{HROOT}/lanes/{scene}/{meth}.json"))
        rows = []; S._det = {}
        for n, (key, ms) in enumerate(frames.items()):
            m0 = ms[0]
            real = load(f"{P}/{m0['real']}"); fg = np.array(Image.open(f"{P}/{m0['fg']}")) > 127
            Hh, W = real.shape[:2]; y0 = int(ROAD_Y0 * Hh)
            zp = f"{root}/{m0['zero']}"
            if not os.path.exists(zp):
                print("missing zero", zp, flush=True); continue
            x0 = load(zp)
            gb, gc = S.det(("real", m0["real"]), real)
            b0, c0 = S.det(("zero", zp), x0)
            L0 = lanes.get(m0["zero"], {}).get("lanes", []); LR = real_lanes.get(m0["real"], {})
            y1 = int(min(LR.get("band", [0, Hh])[1], Hh))
            tex_real = tex_energy(real, fg); tex0 = tex_energy(x0, fg)
            for m in ms:
                p = f"{root}/{m['path']}"
                if not os.path.exists(p):
                    continue
                x = load(p)
                plane = m["dz"] != 0
                Hm = homography(m["K"], m["c2w0"], m["c2w1"], m["cam_h"])
                gw, valid = warp_img(real, Hm)
                fgw = warp_img(fg.astype(np.uint8), Hm, cv2.INTER_NEAREST)[0] > 0
                if plane:
                    fgw = cv2.dilate(fgw.astype(np.uint8), np.ones((41, 9), np.uint8)) > 0
                    valid_ref = valid.copy(); valid_ref[:y0] = False   # road rows only
                else:
                    valid_ref = valid
                excl = fgw | ~valid_ref
                # lanes are road paint anywhere below the horizon (not only in the lower 40% used for paint masks)
                c0p = np.asarray(m["c2w0"]); Kp = np.asarray(m["K"])
                yh = Kp[1, 2] - Kp[1, 1] * np.tan(np.arcsin(-c0p[2, 2]))
                valid_lane = valid.copy()
                if plane:
                    valid_lane[:int(yh) + 10] = False
                r = dict(method=meth, scene=scene, src=m["src"], dst=m["dst"], ts=m["ts"], cond=m["cond"], yaw=m["yaw"],
                         dz=m["dz"], axis=m["axis"], offset_m=m["offset_m"], onpath=m["onpath"],
                         valid_frac=float(valid.mean()))
                # (1) vs pseudo-ground truth (warped real frame)
                road = np.zeros_like(valid); road[y0:] = True
                r["road_psnr"] = psnr_mask(x, gw, valid & road & ~fgw)
                rl = S.img_metrics(x[y0:], gw[y0:], fgw[y0:], valid[y0:])[0]
                r["road_lpips"] = rl
                if not plane:
                    r["psnr"] = psnr_mask(x, gw, valid & ~fgw)
                    c1, c2 = col_band(valid)
                    r["lpips"], r["ssim"] = S.img_metrics(x[:, c1:c2], gw[:, c1:c2], fgw[:, c1:c2], valid[:, c1:c2])
                f1, pr, rc = mark_agree(marking_mask(x, excl) & valid_ref, marking_mask(gw, excl) & valid_ref)
                r.update(mark_f1=f1, mark_prec=pr, mark_rec=rc)
                L = lanes.get(m["path"], {}).get("lanes", [])
                lg = warp_lanes(LR.get("lanes", []), Hm, W, Hh)
                r.update(lane_compare(L, lg, valid_lane, y1))
                # matched 0-deviation baseline: the same method's 0-deviation output vs the real frame, on the SAME scene
                # content (the region of the recorded frame that lands in the evaluated region of the deviated frame)
                Hi = np.linalg.inv(Hm)
                back = lambda msk: warp_img(msk.astype(np.uint8), Hi, cv2.INTER_NEAREST)[0] > 0
                r.update({"ref0_" + k: v for k, v in lane_compare(L0, LR.get("lanes", []), back(valid_lane), y1).items()
                          if k in ("ntl_iou", "lane_f1")})
                r["ref0_road_psnr"] = psnr_mask(x0, real, back(valid & road) & ~fg)
                if not plane:
                    sv = back(valid)
                    r["ref0_psnr"] = psnr_mask(x0, real, sv & ~fg)
                    c1, c2 = col_band(sv)
                    r["ref0_lpips"] = S.img_metrics(x0[:, c1:c2], real[:, c1:c2], fg[:, c1:c2], np.ones_like(fg[:, c1:c2]))[0]
                    r["ref0_mark_f1"] = mark_agree(marking_mask(x0, fg | ~sv) & sv, marking_mask(real, fg | ~sv) & sv)[0]
                b, c = S.det(("dev", p), x)
                mv = np.isin(c, list(MOVABLE)); hall = unver = 0
                for x1, yy1, x2, yy2 in b[mv].astype(int):
                    cx, cy = int(np.clip((x1 + x2) / 2, 0, W - 1)), int(np.clip((yy1 + yy2) / 2, 0, Hh - 1))
                    if not valid[cy, cx]:
                        unver += 1; continue
                    patch = fgw[max(yy1, 0):yy2, max(x1, 0):x2]
                    if patch.size == 0 or patch.mean() < 0.3:
                        hall += 1
                r.update(yolo_halluc_veh=hall, veh_unverif=unver, n_veh=int(mv.sum()), n_veh0=int(np.isin(c0, list(MOVABLE)).sum()))
                if not plane:
                    s = np.isin(c, list(STATIC)); gs = np.isin(gc, list(STATIC))
                    gbw = warp_boxes(gb[gs], Hm, W, Hh)
                    inside = lambda bb: valid[np.clip(((bb[:, 1] + bb[:, 3]) / 2).astype(int), 0, Hh - 1),
                                              np.clip(((bb[:, 0] + bb[:, 2]) / 2).astype(int), 0, W - 1)] if len(bb) else np.zeros(0, bool)
                    ks, kg = inside(b[s]), inside(gbw)
                    r["yolo_static_f1"] = det_f1(gbw[kg], gc[gs][kg], b[s][ks], c[s][ks])
                # (2) consistency with the same method's 0-deviation output
                x0w, v0 = warp_img(x0, Hm)
                vst = v0 & valid_lane if plane else v0
                r.update({"lane_stab_" + k: v for k, v in lane_compare(L, warp_lanes(L0, Hm, W, Hh), vst, y1).items()
                          if k in ("ntl_iou", "lane_f1")})
                if not plane:
                    bw = warp_boxes(b0, Hm, W, Hh); kb, kx = inside(bw), inside(b)
                    r["det_stab_f1"] = det_f1(bw[kb], c0[kb], b[kx], c[kx])
                    c1, c2 = col_band(v0)
                    r["self_lpips"] = S.img_metrics(x[:, c1:c2], x0w[:, c1:c2], fgw[:, c1:c2], v0[:, c1:c2])[0]
                else:
                    r["self_lpips"] = S.img_metrics(x[y0:], x0w[y0:], fgw[y0:], v0[y0:])[0]
                tx = tex_energy(x, fgw)  # statistic over the deviated frame (movable objects at their warped position)
                r.update(tex_ratio=float(tx / (tex_real + 1e-9)), tex_stab=float(tx / (tex0 + 1e-9)))
                rows.append(r)
            if n % 25 == 0:
                print(meth, n, "/", len(frames), flush=True)
        json.dump(rows, open(outp, "w"), default=float)
        print("wrote", outp, len(rows), flush=True)


METRICS = ["psnr", "ssim", "lpips", "road_psnr", "road_lpips", "mark_prec", "mark_rec", "mark_f1", "ntl_iou", "lane_f1",
           "yolo_static_f1", "yolo_halluc_veh", "veh_unverif", "lane_stab_ntl_iou", "lane_stab_lane_f1", "det_stab_f1",
           "self_lpips", "tex_ratio", "tex_stab", "valid_frac",
           "d_psnr", "d_lpips", "d_road_psnr", "d_mark_f1", "d_lane_f1", "d_ntl_iou"]
DELTA = ["psnr", "lpips", "road_psnr", "mark_f1", "lane_f1", "ntl_iou"]  # d_k = k - ref0_k (matched-content change)


def summary():
    from eval.summarize import boot
    import pandas as pd
    rows = sum([json.load(open(p)) for p in sorted(glob.glob(f"{OUT}/rows_scene_*.json"))], [])
    df = pd.DataFrame(rows)
    for k in DELTA:
        if "ref0_" + k in df:
            df["d_" + k] = df[k] - df["ref0_" + k]
    df["seq"] = df.scene + ":" + df.src + ">" + df.dst
    df["level"] = np.where(df.axis == "height", df.dz, df.yaw)
    recs = []
    for (ax, lev, meth), g in df.groupby(["axis", "level", "method"]):
        r = dict(axis=ax, level=lev, method=meth, n=len(g), n_seq=g.seq.nunique(), scenes=",".join(sorted(g.scene.unique())))
        for k in METRICS:
            if k not in g or g[k].isna().all():
                continue
            r[k] = float(np.nanmean(g[k])); r[k + "_clo"], r[k + "_chi"] = boot(g[k].values, groups=g.seq.values)
        recs.append(r)
    S = pd.DataFrame(recs).sort_values(["axis", "method", "level"]); S.to_csv(f"{OUT}/summary.csv", index=False)
    # compact markdown tables per axis (mean [sequence-bootstrap 95% CI] for the key metrics)
    order = ["raw", "difix", "difix_ref", "nv_fixer", "difix_ft", "difix_reimpl_ref", "ours_hfgan_s0"]
    ky = ["psnr", "d_psnr", "lpips", "d_lpips", "mark_f1", "d_mark_f1", "lane_f1", "d_lane_f1", "yolo_halluc_veh",
          "lane_stab_lane_f1", "det_stab_f1", "self_lpips", "tex_stab"]
    keys = {"yaw": ky, "lateral_yaw": ky,
            "height": ["road_psnr", "d_road_psnr", "road_lpips", "mark_f1", "lane_f1", "d_lane_f1", "yolo_halluc_veh",
                       "lane_stab_lane_f1", "self_lpips", "tex_ratio", "tex_stab", "valid_frac"]}
    with open(f"{OUT}/summary.md", "w") as f:
        f.write(f"# Heading / camera-height axis (Para-Lane {', '.join(sorted(df.scene.unique(), key=lambda s: int(s[6:])))}) -- generated by src/eval/heading.py summary\n\n"
                "Mean [95% bootstrap CI over lane-pair sequences]. level = yaw (deg, + = left) or dz (m, + = up).\n")
        for ax, ks in keys.items():
            A = S[S.axis == ax]
            if A.empty:
                continue
            f.write(f"\n## axis = {ax}\n\n| level | method | n | " + " | ".join(ks) + " |\n|" + "---|" * (3 + len(ks)) + "\n")
            for meth in [m for m in order if m in set(A.method)] + sorted(set(A.method) - set(order)):
                for _, r in A[A.method == meth].sort_values("level").iterrows():
                    cells = [f"{r[k]:.3f} [{r[k+'_clo']:.2f},{r[k+'_chi']:.2f}]" if k in r and not pd.isna(r[k]) else "–" for k in ks]
                    f.write(f"| {r['level']:+g} | {meth} | {r['n']} | " + " | ".join(cells) + " |\n")
    compact(df, order)
    print(open(f"{OUT}/summary.md").read()[:6000])


def compact(df, order):
    """results/heading/tables.md: key metrics per method at pooled |yaw| (both signs) and at each dz, mean [CI]."""
    from eval.summarize import boot
    import pandas as pd
    df = df.copy(); df["alev"] = np.where(df.axis == "height", df.dz, df.yaw.abs())
    spec = {"yaw": ("on-path yaw, |deg| (both signs pooled); GT = rotation-warped real frame",
                    ["psnr", "d_psnr", "lpips", "mark_f1", "lane_f1", "lane_stab_lane_f1", "self_lpips", "yolo_halluc_veh"]),
            "lateral_yaw": ("lateral (real adjacent-lane pose, ~3 m and ~6 m) + yaw, |deg|; GT = rotation-warped real adjacent-lane frame",
                            ["psnr", "d_psnr", "lpips", "mark_f1", "lane_f1", "lane_stab_lane_f1", "self_lpips", "yolo_halluc_veh"]),
            "height": ("on-path camera height change dz (m); GT = road-plane-warped real frame, road only",
                       ["road_psnr", "d_road_psnr", "mark_f1", "lane_f1", "d_lane_f1", "lane_stab_lane_f1", "self_lpips", "tex_stab", "valid_frac"])}
    fmt = lambda v, lo, hi: "–" if pd.isna(v) else (f"{v:.3f}" if pd.isna(lo) else f"{v:.3f} [{lo:.2f}, {hi:.2f}]")
    with open(f"{OUT}/tables.md", "w") as f:
        f.write("# Heading / camera height, compact tables (src/eval/heading.py summary)\n\nMean [95% bootstrap CI over "
                "lane-pair sequences]; d_* = change vs. the same method's 0-deviation output on the same scene content.\n")
        for ax, (title, ks) in spec.items():
            A = df[df.axis == ax]
            f.write(f"\n## {title}\n\n| level | method | n | " + " | ".join(ks) + " |\n|" + "---|" * (3 + len(ks)) + "\n")
            for lev in sorted(A.alev.unique()):
                for meth in [m for m in order if m in set(A.method)]:
                    g = A[(A.alev == lev) & (A.method == meth)]
                    cells = []
                    for k in ks:
                        if k not in g or g[k].isna().all():
                            cells.append("–"); continue
                        lo, hi = boot(g[k].values, groups=g.seq.values)
                        cells.append(fmt(float(np.nanmean(g[k])), lo, hi))
                    f.write(f"| {lev:g} | {meth} | {len(g)} | " + " | ".join(cells) + " |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "summary"]); ap.add_argument("scene", nargs="?")
    ap.add_argument("--methods", default="raw,difix,difix_ref,nv_fixer,difix_ft,difix_reimpl_ref,ours_hfgan_s0")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "eval":
        evaluate_scene(a.scene, a.methods.split(","), a.force)
    else:
        summary()


if __name__ == "__main__":
    main()
