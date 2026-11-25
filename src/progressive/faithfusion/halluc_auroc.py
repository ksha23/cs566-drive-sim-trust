"""Does FaithFusion's EIG map (or a reprojection-consistency map, or plain splat coverage) predict which pixels a
fixer hallucinates, relative to the REAL image of the target lane?

Per image (cross-lane eval renders of the test scenes; half resolution 480x270), two per-pixel hallucination labels:
  PAINT (primary)  lane paint in the fixer output that is not in the real image: marking_mask(fixed) and not within
                   5 px (full-res ~10 px) of marking_mask(real). Region: road region (lower 40 %) minus movable objects.
  CHANGE (secondary) pixels the fixer changed noticeably AND got wrong: DeltaE76(fixed, raw) > 10 and
                   DeltaE76(fixed, real) > 20 (CIELAB, after a 7x7 box blur). Region: all pixels minus movable objects
                   minus sky (real image: upper half & blue-ish).
Scores (higher = less trustworthy): gain = FaithFusion log-EIG (100 = uncovered), unobs = reprojection-inconsistency
fraction, 1-alpha. AUROC and AUPRC pooled over pixels (<= 1500 random region pixels per image) per method x offset bin,
with 95% bootstrap CIs over images (100 resamples).

usage: python -m progressive.faithfusion.halluc_auroc --scenes scene_16,scene_17 --methods difix,difix_ref,difix_ft,ours_base_s0 --out results/faithfusion
"""
import argparse, json, os
import numpy as np
import cv2
from PIL import Image

from common.config import DATA as D
from eval.evaluate import marking_mask

HW = (480, 270)
BINS = [("~3m", 1.0, 4.5), ("~6m", 4.5, 20.0)]
SCORES = ["gain", "unobs", "1-alpha"]


def lab(x):
    return cv2.cvtColor(cv2.blur(x, (7, 7)).astype(np.float32) / 255, cv2.COLOR_RGB2LAB)


def auc_roc(s, y):
    y = y.astype(bool); n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = np.empty(len(s)); order = np.argsort(s, kind="mergesort"); r[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_pr(s, y):
    y = y.astype(bool)
    if y.sum() == 0:
        return np.nan
    o = np.argsort(-s, kind="mergesort"); yy = y[o]
    tp = np.cumsum(yy); prec = tp / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / yy.sum())


def labels_for(meth, m):
    rd = lambda p, nearest=False: np.array(Image.open(p).convert("RGB" if not nearest else "L").resize(HW, Image.NEAREST if nearest else Image.BILINEAR))
    real = rd(f"{D}/pairs/{m['real']}"); raw = rd(f"{D}/pairs/{m['path']}"); fx = rd(f"{D}/fixed/{meth}/{m['path']}")
    fg = rd(f"{D}/pairs/{m['fg']}", nearest=True) > 127
    H = HW[1]; road = np.zeros((H, HW[0]), bool); road[int(0.6 * H):] = True; road &= ~fg
    # paint labels (marking_mask works at any resolution; it applies its own road crop)
    pf, pr = marking_mask(fx, fg), marking_mask(real, fg)
    paint = pf & ~(cv2.dilate(pr.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0)
    # change-and-wrong labels
    lf, lr, lw = lab(fx), lab(real), lab(raw)
    d_fr = np.linalg.norm(lf - lr, axis=-1); d_fw = np.linalg.norm(lf - lw, axis=-1)
    x = real.astype(np.float32) / 255
    rows = np.arange(H)[:, None] < H * 0.5
    sky = rows & (x[..., 2] > 0.45) & (x[..., 2] > x[..., 0] + 0.05)
    reg2 = ~fg & ~sky
    change = (d_fw > 10) & (d_fr > 20)
    return dict(paint=(paint, road), change=(change, reg2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="scene_16,scene_17"); ap.add_argument("--methods", default="difix,difix_ref,difix_ft,ours_base_s0")
    ap.add_argument("--maps", default=f"{D}/faithfusion/maps"); ap.add_argument("--out", default="results/faithfusion")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    # per image: sampled (scores, labels) per label type
    samples = {}   # (meth, bin, lab) -> list of (S[n,3], y[n])
    rates = {}
    for scene in a.scenes.split(","):
        meta = [m for m in json.load(open(f"{D}/pairs/{scene}/meta.json")) if m["variant"] == "full"]
        items = [m for k, m in enumerate(meta) if (m["onpath_test"] or k % 2 == 0) and not m["onpath_test"]]
        for m in items:
            mp = f"{a.maps}/{scene}/{m['src']}_to_{m['dst']}/{m['ts']}.npz"
            if not os.path.exists(mp):
                continue
            Z = np.load(mp)
            S = np.stack([Z["gain"].astype(np.float32), Z["unobs"].astype(np.float32), 1 - Z["alpha"].astype(np.float32)], -1)
            b = [n for n, lo, hi in BINS if lo < abs(m["offset_m"]) <= hi]
            if not b:
                continue
            for meth in a.methods.split(","):
                if not os.path.exists(f"{D}/fixed/{meth}/{m['path']}"):
                    continue
                for labname, (y, reg) in labels_for(meth, m).items():
                    idx = np.flatnonzero(reg.ravel())
                    if len(idx) == 0:
                        continue
                    idx = rng.choice(idx, min(1500, len(idx)), replace=False)
                    samples.setdefault((meth, b[0], labname), []).append((S.reshape(-1, 3)[idx], y.ravel()[idx]))
                    rates.setdefault((meth, b[0], labname), []).append(float(y[reg].mean()))
        print(scene, "done", flush=True)
    rows = []
    for (meth, b, labname), L in sorted(samples.items()):
        r = dict(method=meth, bin=b, label=labname, n_images=len(L), pos_rate=float(np.mean(rates[(meth, b, labname)])))
        for j, sname in enumerate(SCORES):
            s = np.concatenate([x[0][:, j] for x in L]); y = np.concatenate([x[1] for x in L])
            r[f"auroc_{sname}"] = auc_roc(s, y); r[f"auprc_{sname}"] = auc_pr(s, y)
            bs = []
            for _ in range(100):
                ii = rng.integers(0, len(L), len(L))
                bs.append(auc_roc(np.concatenate([L[i][0][:, j] for i in ii]), np.concatenate([L[i][1] for i in ii])))
            r[f"auroc_{sname}_lo"], r[f"auroc_{sname}_hi"] = [float(np.nanpercentile(bs, q)) for q in (2.5, 97.5)]
        rows.append(r); print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    json.dump(rows, open(f"{a.out}/halluc_auroc.json", "w"), indent=1)
    with open(f"{a.out}/halluc_auroc.md", "w") as f:
        f.write("| label | method | offset | images | positive rate (share of road-region pixels) | AUROC EIG [95% CI] | AUROC reproj. [95% CI] | AUROC 1-alpha | AUPRC EIG / reproj. / 1-alpha |\n|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['label']} | {r['method']} | {r['bin']} | {r['n_images']} | {r['pos_rate']:.3f} | "
                    f"{r['auroc_gain']:.3f} [{r['auroc_gain_lo']:.3f}, {r['auroc_gain_hi']:.3f}] | "
                    f"{r['auroc_unobs']:.3f} [{r['auroc_unobs_lo']:.3f}, {r['auroc_unobs_hi']:.3f}] | {r['auroc_1-alpha']:.3f} | "
                    f"{r['auprc_gain']:.3f} / {r['auprc_unobs']:.3f} / {r['auprc_1-alpha']:.3f} |\n")
    print(open(f"{a.out}/halluc_auroc.md").read())


if __name__ == "__main__":
    main()
