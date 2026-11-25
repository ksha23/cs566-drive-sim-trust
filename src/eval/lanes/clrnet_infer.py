"""Batch lane detection with the released CLRNet (Turoad/CLRNet, CULane weights), for Para-Lane-style frames.

Runs in its OWN env ($CS566D_DATA/lanes/env: a venv on top of the project env + mmcv-lite 1.7.2, CLRNet NMS op
built for sm_86); writes plain JSON so the main env never imports CLRNet.

Geometry (Para-Lane 960x540 front camera -> CULane 1640x590 input):
  CULane frames are 1640x590 and CLRNet only looks at rows >= cut_height=270 (roughly the horizon), resized to 800x320.
  We scale the Para-Lane frame uniformly to width 1640 (s = 1640/W) and crop a 590-row band placed so that the camera
  horizon lands at CULane's cut line (y=270). The horizon row is y_h = cy - fy*tan(pitch); for Para-Lane's front camera
  (fy=677 at 960x540, pitch ~5.7 deg down) y_h ~= 202 px, which is the default (--horizon). The band covers original rows
  [y_h - 270/s, y_h + 320/s] = [44, 389]; road below row ~389 (the nearest ~4-5 m) is outside CULane's field of view,
  so lanes are only detected (and compared) above that row. No non-uniform squashing, so lane geometry is undistorted.
  Output lane points are mapped back to the ORIGINAL image pixel coordinates.

usage (lanes env):
  python clrnet_infer.py --out lanes.json [--root DIR] (--glob 'DIR/**/*.png' | --list paths.txt) [--model dla34|r18]
JSON: {relative_path: {"lanes": [[[x,y],...], ...], "conf": [..], "band": [y0, y1]}, "_meta": {...}}
"""
import argparse, glob, json, os, sys, time
import numpy as np

# (numpy>=2 has np.bool again, which CLRNet uses)
import cv2
import torch

_DATA = os.path.expanduser(os.environ.get("CS566D_DATA", "~/cs566/data/D"))
CLR = os.environ.get("CLRNET_ROOT", f"{_DATA}/lanes/CLRNet")
WEIGHTS = os.environ.get("CLRNET_WEIGHTS", f"{_DATA}/lanes/weights")
sys.path.insert(0, CLR)


def build(model):
    from clrnet.utils.config import Config
    from clrnet.models.registry import build_net
    cfgf = {"dla34": "clr_dla34_culane.py", "r18": "clr_resnet18_culane.py"}[model]
    cfg = Config.fromfile(f"{CLR}/configs/clrnet/{cfgf}")
    cfg.backbone.pretrained = False  # all weights come from the CULane checkpoint
    net = build_net(cfg)
    sd = torch.load(f"{WEIGHTS}/culane_{model}.pth", map_location="cpu")["net"]
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    params = {n for n, _ in net.named_parameters()}
    assert not [m for m in missing if m in params], missing  # only derived buffers (priors, loss weight) may be missing
    return net.cuda().eval(), cfg


def band(W, H, horizon, cfg):
    s = cfg.ori_img_w / W
    y0 = horizon - cfg.cut_height / s
    return s, y0  # CULane row r  <->  original row y0 + r / s


def prep(img_bgr, s, y0, cfg):
    H, W = img_bgr.shape[:2]
    big = cv2.resize(img_bgr, (cfg.ori_img_w, int(round(H * s))), interpolation=cv2.INTER_LINEAR)
    top = int(round(y0 * s))
    canvas = np.zeros((cfg.ori_img_h, cfg.ori_img_w, 3), np.uint8)  # pad if the band leaves the image
    a, b = max(top, 0), min(top + cfg.ori_img_h, big.shape[0])
    canvas[a - top:b - top] = big[a:b]
    x = canvas[cfg.cut_height:]
    x = cv2.resize(x, (cfg.img_w, cfg.img_h), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--root", default="")
    ap.add_argument("--glob", default=None); ap.add_argument("--list", default=None)
    ap.add_argument("--model", default="dla34"); ap.add_argument("--horizon", type=float, default=202.0)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--conf", type=float, default=None)
    a = ap.parse_args()
    paths = sorted(glob.glob(a.glob, recursive=True)) if a.glob else [l.strip() for l in open(a.list) if l.strip()]
    net, cfg = build(a.model)
    if a.conf is not None:
        cfg.test_parameters.conf_threshold = a.conf
    res = {"_meta": dict(model=f"CLRNet-{a.model} CULane", horizon=a.horizon, conf=cfg.test_parameters.conf_threshold,
                         mapping="uniform scale to width 1640, horizon at CULane cut_height=270")}
    t0 = time.time(); n = 0
    for i in range(0, len(paths), a.bs):
        chunk = paths[i:i + a.bs]; ims = [cv2.imread(p) for p in chunk]
        H, W = ims[0].shape[:2]; s, y0 = band(W, H, a.horizon, cfg)
        x = torch.stack([prep(im, s, y0, cfg) for im in ims]).cuda()
        with torch.no_grad():
            out = net(x); lanes = net.heads.get_lanes(out)
        for p, ls in zip(chunk, lanes):
            L, C = [], []
            for lane in ls:
                pts = lane.points  # normalised (x in [0,1] of CULane width, y in [0,1] of CULane height)
                xs = pts[:, 0] * cfg.ori_img_w / s; ys = y0 + pts[:, 1] * cfg.ori_img_h / s
                ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
                if ok.sum() >= 2:
                    L.append(np.stack([xs[ok], ys[ok]], 1).round(1).tolist()); C.append(float(lane.metadata["conf"]))
            key = os.path.relpath(p, a.root) if a.root else p
            res[key] = dict(lanes=L, conf=C, band=[float(y0), float(y0 + cfg.ori_img_h / s)])
            n += 1
    res["_meta"]["ms_per_img"] = (time.time() - t0) / max(n, 1) * 1000
    json.dump(res, open(a.out, "w"))
    print(f"{n} images, {res['_meta']['ms_per_img']:.1f} ms/img -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
