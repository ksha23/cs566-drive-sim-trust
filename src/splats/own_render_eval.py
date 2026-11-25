"""Render splats of lane clip_0 at the evaluation poses (held-out on-path frames and the other lanes' real poses) so
the existing evaluation (src/eval/evaluate.py, CLRNet lanes) can score them next to the original gsplat renders.

Builds <out_root>/pairs/<scene>/ = the evaluation subset of the original pairs (exactly the items
fixers.apply.eval_items(pairs, scene, stride=2) selects, restricted to src == clip_0, so run the evaluation with
--stride 1), with real/, fg/ and render/ symlinked to the original pairs dir ("raw" = the original gsplat-fitted
splat rendered by gsplat). Each --splat name=path:renderer is rendered to <out_root>/fixed/<name>/<meta path>.

usage: scripts/lib/py.sh src/splats/own_render_eval.py <scene> <out_root> --splat own_refit=<p>:own [--splat ...]
"""
import argparse, json, os, time
import numpy as np
import torch
from PIL import Image

from common.config import DATA
from fixers.apply import eval_items
from splats.paralane import clip_dir, load_cameras
from splats.gs import load_splat, pixel_dirs, render as gs_render
from splats.own_render import render as own_render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene"); ap.add_argument("out_root")
    ap.add_argument("--splat", action="append", default=[], help="name=path:gsplat|own")
    ap.add_argument("--src", default="clip_0")
    a = ap.parse_args()
    P0 = f"{DATA}/pairs"; P = f"{a.out_root}/pairs/{a.scene}"
    os.makedirs(P, exist_ok=True)
    items = [m for m in eval_items(P0, a.scene, 2) if m["src"] == a.src]
    json.dump(items, open(f"{P}/meta.json", "w"), indent=0)
    for d in ("real", "fg", "render"):
        if not os.path.exists(f"{P}/{d}"):
            os.symlink(f"{P0}/{a.scene}/{d}", f"{P}/{d}")
    print(f"{len(items)} eval items (src {a.src})", flush=True)
    cams = {}
    for m in items:
        if m["dst"] not in cams:
            K, ts, c2w = load_cameras(clip_dir(a.scene, m["dst"]))
            cams[m["dst"]] = (torch.tensor(K, device="cuda"), ts, torch.tensor(c2w, device="cuda"))
    timing = {}
    for spec in a.splat:
        name, rest = spec.split("=", 1); path, rn = rest.rsplit(":", 1)
        fn = own_render if rn == "own" else gs_render
        params, sky = load_splat(path)
        ts_ = []
        with torch.no_grad():
            for m in items:
                K, ts, c2w = cams[m["dst"]]
                torch.cuda.synchronize(); t = time.perf_counter()
                img = fn(params, sky, c2w[ts.index(m["ts"])], K, cam_dirs=pixel_dirs(K))[0]
                torch.cuda.synchronize(); ts_.append(time.perf_counter() - t)
                o = f"{a.out_root}/fixed/{name}/{m['path']}"; os.makedirs(os.path.dirname(o), exist_ok=True)
                Image.fromarray((img.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(o)
        timing[name] = dict(renderer=rn, splat=path, n_gauss=int(params["means"].shape[0]),
                            median_ms=1000 * float(np.median(ts_[1:])), n=len(ts_))
        print(name, json.dumps(timing[name]), flush=True)
        del params, sky; torch.cuda.empty_cache()
    json.dump(timing, open(f"{a.out_root}/render_timing_{a.scene}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
