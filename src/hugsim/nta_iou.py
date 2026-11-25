"""NTA-IoU (ReconDreamer, CVPR 2025) on HUGSIM offset renders, for raw renders and each fixer.

ReconDreamer's NTA-IoU: project the 3D boxes of vehicles into the novel view -> 2D GT boxes; run a detector (YOLO11)
on the novel-view render; for every GT box take the IoU with the best-matching vehicle detection (0 if none);
average. (ReconDreamer's repo releases no standalone NTA-IoU script outside their Street Gaussians fork, so this is a
re-implementation of the metric as described in the paper.) Vehicle classes: COCO car/motorcycle/bus/truck.
We report mean over GT boxes, with a 95% bootstrap CI over frames (resampling frames, 2000 draws).

Fixers (applied to the offset renders before detection):
  raw        render as-is
  difix      nvidia/difix, 1 step, zero-shot
  difix_ref  nvidia/difix_ref; reference = HUGSIM's ON-PATH (0 m) render at the same timestamp (the best view a
             simulator has of the logged path; we have no real images for most HUGSIM scenes)
  difix_ft   Difix with our fine-tuned U-Net weights (--difix_ft path)
  <name>     our compact U-Net fixer(s) (--unet name=path, repeatable)

usage (difix env): scripts/lib/py.sh src/hugsim/nta_iou.py --renders $CS566D_DATA/hugsim/renders --scenes scene-0071,scene-0661 \
        --methods raw,difix,difix_ref,ours --unet ours=$CS566D_DATA/fixers/base_s0/best.pt --out results/hugsim
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # src/ (also set by scripts/lib/py.sh)
from common.config import DIFIX3D_SRC, HUGSIM_DATA, YOLO_WEIGHTS
sys.path.insert(0, DIFIX3D_SRC)
VEH = {2, 3, 5, 7}


def box_iou(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None] - inter + 1e-9)


def nta_ious(gt, det):
    """per-GT-box best IoU with any vehicle detection"""
    if len(gt) == 0:
        return []
    if len(det) == 0:
        return [0.0] * len(gt)
    return box_iou(gt, det).max(1).tolist()


def bootstrap(per_frame, n=2000, seed=0):
    """per_frame: list of lists of box IoUs. Mean over boxes, CI by resampling frames."""
    rng = np.random.default_rng(seed)
    fr = [np.array(x) for x in per_frame if len(x)]
    if not fr:
        return np.nan, np.nan, np.nan
    mean = np.concatenate(fr).mean(); bs = []
    for _ in range(n):
        s = rng.integers(0, len(fr), len(fr)); bs.append(np.concatenate([fr[i] for i in s]).mean())
    return float(mean), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


class Fixers:
    def __init__(self, a):
        self.a = a; self.cur = None; self.fn = None

    def load(self, meth):
        if self.cur == meth:
            return
        self.fn = None; torch.cuda.empty_cache(); self.cur = meth
        if meth in self.a.unet:
            from fixers.unet.model import UNetFixer
            s = torch.load(self.a.unet[meth], map_location="cuda")
            net = UNetFixer(base=s.get("base", 48)).cuda().eval(); net.load_state_dict(s["net"])

            def fix(img, ref):
                x = torch.from_numpy(np.array(img)).permute(2, 0, 1)[None].float().cuda() / 255
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    y = net(x)
                return Image.fromarray((y[0].float().permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8))
        elif meth == "nv_fixer":  # NVIDIA Fixer [4]: subprocess server in its own env (fixers/registry.py)
            from fixers.registry import _nv_fixer_client
            nv = _nv_fixer_client()

            def fix(img, ref):
                y = nv(torch.from_numpy(np.array(img)).float().cuda() / 255)
                return Image.fromarray((y.cpu().numpy() * 255 + 0.5).astype(np.uint8))
        elif meth == "difix_reimpl":  # our Difix re-implementation, reference view (needs its own venv)
            sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fixers", "difix_reimpl"))
            from infer import load as load_reimpl
            fix = load_reimpl("ours", self.a.difix_reimpl, use_ref=True)  # (img, ref) -> PIL at img.size
        else:
            from pipeline_difix import DifixPipeline
            pipe = DifixPipeline.from_pretrained("nvidia/difix_ref" if meth == "difix_ref" else "nvidia/difix", trust_remote_code=True).to("cuda")
            pipe.set_progress_bar_config(disable=True)
            if meth == "difix_ft":
                pipe.unet.load_state_dict({k: v.float() for k, v in torch.load(self.a.difix_ft, map_location="cuda").items()})

            def fix(img, ref):
                kw = dict(image=img, num_inference_steps=1, timesteps=[199], guidance_scale=0.0)
                if meth == "difix_ref":
                    kw["ref_image"] = ref
                return pipe("remove degradation", **kw).images[0].resize(img.size, Image.BICUBIC)
        self.fn = fix

    def __call__(self, meth, img, ref):
        self.load(meth)
        return self.fn(img, ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", default=f"{HUGSIM_DATA}/renders")
    ap.add_argument("--fixed", default=f"{HUGSIM_DATA}/fixed")
    ap.add_argument("--scenes", required=True); ap.add_argument("--methods", default="raw,difix,difix_ref,ours")
    ap.add_argument("--unet", action="append", default=[], help="name=ckpt (repeatable)")
    ap.add_argument("--difix_ft", default=None); ap.add_argument("--out", default="results/hugsim")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--min_vis", type=float, default=0.5)
    ap.add_argument("--static_scenes", default="scene-0071",
                    help="scenes whose HUGSIM actor tracks disagree with nuScenes annotations: keep only parked GT vehicles")
    a = ap.parse_args()
    a.unet = {kv.split("=")[0]: os.path.expanduser(kv.split("=", 1)[1]) for kv in a.unet}
    os.makedirs(a.out, exist_ok=True)
    from ultralytics import YOLO
    yolo = YOLO(YOLO_WEIGHTS)
    FX = Fixers(a); rows = []; timing = {}
    methods = a.methods.split(",")
    for meth in methods:
        tm = []
        for scene in a.scenes.split(","):
            index = json.load(open(f"{a.renders}/{scene}/index.json"))
            for rec in index:
                ref = Image.open(f"{a.renders}/{rec['views']['0']['path']}").convert("RGB")
                views = dict(rec["views"])
                if "real" in rec:
                    views["real"] = dict(path=rec["real"], gt=rec["views"]["0"]["gt"])
                for off, v in views.items():
                    if off == "real" and meth != "raw":
                        continue
                    src = f"{a.renders}/{v['path']}"
                    if meth == "raw":
                        p = src
                    else:
                        p = f"{a.fixed}/{meth}/{v['path']}"
                        if a.recompute or not os.path.exists(p):
                            os.makedirs(os.path.dirname(p), exist_ok=True)
                            img = Image.open(src).convert("RGB")
                            torch.cuda.synchronize(); t = time.time()
                            o = FX(meth, img, ref)
                            torch.cuda.synchronize(); tm.append(time.time() - t); o.save(p)
                    r = yolo(p, verbose=False, conf=0.25)[0].boxes
                    k = [i for i, c in enumerate(r.cls.tolist()) if int(c) in VEH]
                    det = r.xyxy[k].cpu().numpy()
                    # vehicles judged visible (not occluded) in the ON-PATH view, tracked into every offset view, so the
                    # GT set does not depend on how well the offset render's geometry is reconstructed
                    ok = {g["id"] for g in rec["views"]["0"]["gt"] if g.get("vis") is None or g["vis"] >= a.min_vis}
                    gt = [g["box"] for g in v["gt"] if g["id"] in ok
                          and not (scene in a.static_scenes.split(",") and g.get("speed", 0) >= 0.5)]
                    rows.append(dict(method=meth if off != "real" else "real", scene=scene, idx=rec["idx"],
                                     offset=(0 if off == "real" else int(off)), n_gt=len(gt), n_det=len(det),
                                     ious=nta_ious(gt, det)))
        if tm:
            timing[meth] = float(np.median(tm[3:]) * 1000)
        print(meth, "done", flush=True)
    json.dump(dict(rows=rows, timing_ms=timing), open(f"{a.out}/nta_iou.json", "w"))
    # summary: by method x |offset|
    lines = ["method,abs_offset_m,nta_iou,ci_lo,ci_hi,n_frames,n_boxes,mean_dets"]
    for m in ["real"] + methods:
        for ao in [0, 1, 2, 3]:
            sel = [r for r in rows if r["method"] == m and abs(r["offset"]) == ao]
            if not sel:
                continue
            mu, lo, hi = bootstrap([r["ious"] for r in sel])
            lines.append(f"{m},{ao},{mu:.4f},{lo:.4f},{hi:.4f},{len(sel)},{sum(r['n_gt'] for r in sel)},"
                         f"{np.mean([r['n_det'] for r in sel]):.2f}")
    lines.append("")
    lines.append("per_scene: method,scene,abs_offset_m,nta_iou,ci_lo,ci_hi,n_frames,n_boxes")
    for m in methods:
        for sc in a.scenes.split(","):
            for ao in [0, 1, 2, 3]:
                sel = [r for r in rows if r["method"] == m and r["scene"] == sc and abs(r["offset"]) == ao]
                if sel:
                    mu, lo, hi = bootstrap([r["ious"] for r in sel])
                    lines.append(f"{m},{sc},{ao},{mu:.4f},{lo:.4f},{hi:.4f},{len(sel)},{sum(r['n_gt'] for r in sel)}")
    open(f"{a.out}/nta_iou_summary.csv", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("timing ms/img:", timing)


if __name__ == "__main__":
    main()
