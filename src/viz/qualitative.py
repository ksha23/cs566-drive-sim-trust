"""Qualitative comparison grids for the final report (results/final/qual_*.jpg): real | raw | fixers on the same
Para-Lane test frame, with the per-image metrics of that frame under each tile. The examples are chosen by fixed rules
from the per-image metric rows (written to results/final/qual_examples.json), not by hand:
  qual_3m_typical      scene_16, one lane over: the frame whose Difix-FT and raw lane F1 are closest to their pooled means
  qual_6m_typical      scene_17, two lanes over: same rule
  qual_hallucination   ~3 m, scene_16/17 (the paint mask misfires on shadows and kerbs of the later roads): among the 30
                       frames per fixer with the largest (raw paint precision - fixer paint precision) + fake vehicles for
                       released Difix / our reimpl., the frame with the largest invented marking; invented paint (orange
                       circles), erased paint (blue squares) and fake vehicles (yellow boxes, YOLO11 as in the metric) are
                       drawn on every fixer tile
  qual_failure         scene_15 (under the highway), two lanes over: real frame has lanes, every fixer's lane F1 = 0
Annotation rules = the metric definitions of src/eval/evaluate.py (paint masks, 3-px tolerance; movable-object mask).
usage: CUDA_VISIBLE_DEVICES= scripts/lib/py.sh src/viz/qualitative.py
"""
import json, os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from common.config import DATA, YOLO_WEIGHTS
from eval.more_scenes_summary import load_rows
from eval.evaluate import marking_mask, MOVABLE
from fixers.apply import eval_items

OUT = "results/final"; P = f"{DATA}/pairs"; F = f"{DATA}/fixed"
METHODS = ["raw", "difix", "nv_fixer", "difix_ft", "difix_reimpl_ref"]
NAME = {"raw": "raw splat", "difix": "released Difix", "nv_fixer": "NVIDIA Fixer", "difix_ft": "Difix-FT (fine-tuned)",
        "difix_reimpl_ref": "our Difix-ref reimpl."}
C_INV, C_ERA, C_VEH = (235, 104, 52), (42, 120, 214), (237, 161, 0)
TW, TH = 640, 360
ATTR = "Data: Para-Lane (Ni et al., 3DV 2025), CC BY-NC 4.0. CS 566 course project, non-commercial."


def font(sz, bold=False):
    for f in (["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf"]):
        for d in ("/usr/share/fonts/truetype/dejavu/", ""):
            try:
                return ImageFont.truetype(d + f, sz)
            except OSError:
                pass
    return ImageFont.load_default()


def meta_for(r):
    for m in eval_items(P, r["scene"], 2):
        if m["src"] == r["src"] and m["dst"] == r["dst"] and str(m["ts"]) == str(r["ts"]):
            return m
    raise KeyError(r)


def components(mask, min_area=150):
    """connected paint regions (nearby fragments merged by a 9-px closing); only regions of >= min_area paint pixels"""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    return [tuple(st[i, :4]) for i in range(1, n) if (mask & (lab == i)).sum() >= min_area]


def invented_area(x, real, fg):
    """largest invented-paint region (pixels) of output x vs real: paint with no real paint within 3 px"""
    px, pg = marking_mask(x, fg), marking_mask(real, fg)
    inv = px & ~(cv2.dilate(pg.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
    m = cv2.morphologyEx(inv.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    return max([(inv & (lab == i)).sum() for i in range(1, n)] or [0])


_yolo = None


def fake_vehicles(img, fg):
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO(YOLO_WEIGHTS)
    r = _yolo(np.ascontiguousarray(img[..., ::-1]), verbose=False, conf=0.3, device="cpu")[0].boxes
    b, c = r.xyxy.numpy(), r.cls.numpy().astype(int); out = []
    for x1, y1, x2, y2 in b[np.isin(c, list(MOVABLE))].astype(int):
        patch = fg[max(y1, 0):y2, max(x1, 0):x2]
        if patch.size == 0 or patch.mean() < 0.3:
            out.append((x1, y1, x2, y2))
    return out


def annotate(im, x, real, fg, s):
    """draw invented / erased paint and fake vehicles of output x (vs real) onto PIL tile im (scale s)"""
    d = ImageDraw.Draw(im); k = np.ones((7, 7), np.uint8)
    px, pg = marking_mask(x, fg), marking_mask(real, fg)
    inv = px & ~(cv2.dilate(pg.astype(np.uint8), k) > 0); era = pg & ~(cv2.dilate(px.astype(np.uint8), k) > 0)
    n = dict(inv=0, era=0, veh=0)
    for (x0, y0, w, h) in components(inv):
        b = [x0 * s - 7, y0 * s - 7, (x0 + w) * s + 7, (y0 + h) * s + 7]
        d.ellipse([b[0] - 1, b[1] - 1, b[2] + 1, b[3] + 1], outline="white", width=5)
        d.ellipse(b, outline=C_INV, width=3); n["inv"] += 1
    for (x0, y0, w, h) in components(era):
        b = [x0 * s - 4, y0 * s - 4, (x0 + w) * s + 4, (y0 + h) * s + 4]
        d.rectangle([b[0] - 1, b[1] - 1, b[2] + 1, b[3] + 1], outline="white", width=5)
        d.rectangle(b, outline=C_ERA, width=3); n["era"] += 1
    for (x1, y1, x2, y2) in fake_vehicles(x, fg):
        d.rectangle([x1 * s, y1 * s, x2 * s, y2 * s], outline="black", width=5)
        d.rectangle([x1 * s, y1 * s, x2 * s, y2 * s], outline=C_VEH, width=3)
        d.text((x1 * s + 3, max(y1 * s - 16, 0)), "fake vehicle", fill=C_VEH, font=font(13, True), stroke_width=2, stroke_fill="black")
        n["veh"] += 1
    return n


def fmt(v, d=2):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def grid(name, rows_by_method, title, note, annotate_all=False):
    r0 = rows_by_method["raw"]; m = meta_for(r0)
    real = np.array(Image.open(f"{P}/{m['real']}").convert("RGB")); fg = np.array(Image.open(f"{P}/{m['fg']}")) > 127
    s = TW / real.shape[1]; cap_h = 44; head = 78; foot = 26 + (26 if annotate_all else 0)
    W = 3 * TW; H = head + 2 * (TH + cap_h) + foot
    can = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(can)
    d.text((12, 8), title, fill=(20, 20, 20), font=font(24, True))
    d.text((12, 42), note, fill=(70, 70, 70), font=font(16))
    tiles = [("real image at this pose", real, None)] + [(NAME[me], None, me) for me in METHODS]
    counts = {}
    for i, (lab, img, me) in enumerate(tiles):
        if me is not None:
            p = f"{P}/{m['path']}" if me == "raw" else f"{F}/{me}/{m['path']}"
            img = np.array(Image.open(p).convert("RGB"))
        im = Image.fromarray(img).resize((TW, TH), Image.LANCZOS)
        if annotate_all and me is not None:
            counts[me] = annotate(im, img, real, fg, s)
        x0, y0 = (i % 3) * TW, head + (i // 3) * (TH + cap_h)
        can.paste(im, (x0, y0))
        dd = ImageDraw.Draw(can)
        dd.rectangle([x0, y0, x0 + 8 + dd.textlength(lab, font=font(18, True)) + 8, y0 + 28], fill=(0, 0, 0))
        dd.text((x0 + 8, y0 + 4), lab, fill="white", font=font(18, True))
        if me is not None:
            r = rows_by_method[me]
            cap = (f"lane F1 {fmt(r.get('lane_f1'))} · paint P {fmt(r.get('mark_prec'))} / R {fmt(r.get('mark_rec'))} · "
                   f"LPIPS {fmt(r.get('lpips'), 3)} · fake veh. {int(r.get('yolo_halluc_veh', 0))}")
        else:
            cap = f"{m['scene']}, lane {m['dst'][-1]} (real); splat trained on lane {m['src'][-1]} only; offset {abs(m['offset_m']):.1f} m"
        dd.text((x0 + 8, y0 + TH + 10), cap, fill=(30, 30, 30), font=font(16))
    fy = H - foot + 4
    if annotate_all:
        d.ellipse([12, fy + 2, 30, fy + 20], outline=C_INV, width=3); d.text((36, fy), "invented paint (not in the real image)", fill=(30, 30, 30), font=font(16))
        d.rectangle([380, fy + 2, 398, fy + 20], outline=C_ERA, width=3); d.text((404, fy), "erased paint (real marking missing)", fill=(30, 30, 30), font=font(16))
        d.rectangle([730, fy + 2, 748, fy + 20], outline=C_VEH, width=3); d.text((754, fy), "fake vehicle (YOLO11, no real object there)", fill=(30, 30, 30), font=font(16))
        fy += 26
    d.text((12, fy), ATTR + "  Vehicles differ between passes (lanes driven at different times); image and paint metrics mask them out.",
           fill=(110, 110, 110), font=font(13))
    can.save(f"{OUT}/{name}.jpg", quality=90, optimize=True)
    return dict(file=f"{name}.jpg", scene=m["scene"], src=m["src"], dst=m["dst"], ts=str(m["ts"]),
                offset_m=round(abs(m["offset_m"]), 2), annotations=counts)


def main():
    df = load_rows(); df = df[df.method.isin(METHODS)]
    W = df.pivot_table(index=["scene", "src", "dst", "ts"], columns="method",
                       values=["lane_f1", "mark_prec", "mark_rec", "lpips", "yolo_halluc_veh", "offset_m"], aggfunc="first")
    W = W.dropna(subset=[(k, me) for k in ("lpips",) for me in METHODS])
    rows = lambda key: {me: {k: W.loc[key, (k, me)] for k in ("lane_f1", "mark_prec", "mark_rec", "lpips", "yolo_halluc_veh")} | dict(
        zip(("scene", "src", "dst", "ts"), key)) for me in METHODS}
    off = W[("offset_m", "raw")].abs()
    pooled = df.assign(ab=df.offset_m.abs())
    ex = []

    def typical(scene, lo, hi):
        mean = {me: pooled[(pooled.ab > lo) & (pooled.ab <= hi) & (pooled.method == me)].lane_f1.mean() for me in ("raw", "difix_ft")}
        c = W[(W.index.get_level_values(0) == scene) & (off > lo) & (off <= hi)].dropna(subset=[("lane_f1", "raw"), ("lane_f1", "difix_ft")])
        sc = (c[("lane_f1", "raw")] - mean["raw"]).abs() + (c[("lane_f1", "difix_ft")] - mean["difix_ft"]).abs()
        return sc.idxmin()

    k = typical("scene_16", 1.0, 4.5)
    ex.append(grid("qual_3m_typical", rows(k), f"One lane over ({abs(W.loc[k, ('offset_m', 'raw')]):.1f} m), Para-Lane scene_16: a typical frame",
                   "Frame chosen by a fixed rule: the raw-splat and Difix-FT lane F1 closest to their pooled means at ~3 m."))
    k = typical("scene_17", 4.5, 20)
    ex.append(grid("qual_6m_typical", rows(k), f"Two lanes over ({abs(W.loc[k, ('offset_m', 'raw')]):.1f} m), Para-Lane scene_17: a typical frame",
                   "Frame chosen by a fixed rule: the raw-splat and Difix-FT lane F1 closest to their pooled means at ~6 m."))
    # hallucination: top-30 frames per fixer by paint-precision drop vs raw (+ fake vehicles), then the frame whose
    # largest invented-paint region is biggest (a whole invented marking rather than speckle)
    # candidates from the original two test roads only: on the later roads hard shadows (scene_15) and bright kerb
    # strips (scene_3/4/11) trigger the color-threshold paint mask, so "invented paint" there is often not paint
    c = W[(off > 1.0) & (off <= 4.5) & W.index.get_level_values(0).isin(["scene_16", "scene_17"])].dropna(subset=[("mark_prec", "raw")])
    best = None
    for me in ("difix", "difix_reimpl_ref"):
        sc = (c[("mark_prec", "raw")] - c[("mark_prec", me)]) + 0.3 * c[("yolo_halluc_veh", me)].clip(upper=2)
        for k2 in sc.sort_values(ascending=False).index[:30]:
            m2 = meta_for(dict(zip(("scene", "src", "dst", "ts"), k2)))
            real = np.array(Image.open(f"{P}/{m2['real']}").convert("RGB")); fg = np.array(Image.open(f"{P}/{m2['fg']}")) > 127
            a = invented_area(np.array(Image.open(f"{F}/{me}/{m2['path']}").convert("RGB")), real, fg)
            if best is None or a > best[1]:
                best = (k2, a, me)
    k = best[0]
    ex.append(grid("qual_hallucination", rows(k), f"Hallucination, one lane over ({abs(W.loc[k, ('offset_m', 'raw')]):.1f} m), Para-Lane {k[0]}",
                   f"Rule (scene_16/17): among the 30 frames per fixer with the largest paint-precision drop vs. the raw splat, the largest "
                   f"invented marking ({NAME[best[2]]}); marks use the metric's own paint masks.", annotate_all=True))
    c = W[(W.index.get_level_values(0) == "scene_15") & (off > 4.5)]
    fails = c[(c[[("lane_f1", me) for me in METHODS[1:]]].fillna(1) == 0).all(1)]
    k = (fails if len(fails) else c)[("lpips", "difix_ft")].idxmax()
    ex.append(grid("qual_failure", rows(k), f"Failure case: two lanes over ({abs(W.loc[k, ('offset_m', 'raw')]):.1f} m) under an elevated highway, Para-Lane scene_15",
                   "Every fixer has lane F1 = 0 here: the real yellow center line is missing in every output; "
                   "they look like roads, but not like this road (not annotated: hard shadows break the paint mask)."))
    json.dump(ex, open(f"{OUT}/qual_examples.json", "w"), indent=1, default=str)
    print(json.dumps(ex, indent=1, default=str))


if __name__ == "__main__":
    main()
