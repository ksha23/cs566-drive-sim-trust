"""Fixer server for HUGSIM closed loop: reads pickled {cam: HxWx3 uint8} from <dir>/fix_in, writes fixed images
to <dir>/fix_out (named pipes). Runs in the Difix env (our Difix re-implementation: in its own venv, see
closed_loop_fix.py) so the simulator env needs no diffusers.
Reference-view fixers (difix_ref, difix_reimpl) need --refs <dir of render_refs.py>: the client then also sends the
scene name and the ego's ground (x, z) position, and each camera gets the recorded frame nearest to that position.

usage: scripts/lib/py.sh src/hugsim/fixer_server.py --dir <out> --method difix|difix_ft|<unet name> [--unet name=ckpt] [--difix_ft path]
       [--difix_reimpl best.pt] [--refs $CS566D_DATA/hugsim/refs]
"""
import argparse, os, pickle, sys, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nta_iou import Fixers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--method", required=True)
    ap.add_argument("--unet", action="append", default=[]); ap.add_argument("--difix_ft", default=None)
    ap.add_argument("--difix_reimpl", default=None); ap.add_argument("--refs", default=None)
    a = ap.parse_args()
    a.unet = {kv.split("=")[0]: os.path.expanduser(kv.split("=", 1)[1]) for kv in a.unet}
    FX = Fixers(a); FX.load(a.method)
    fin, fout = f"{a.dir}/fix_in", f"{a.dir}/fix_out"
    for p in (fin, fout):
        if not os.path.exists(p):
            os.mkfifo(p)
    print("fixer ready", a.method, flush=True); times = []; paths = {}; ref_idx = []
    while True:
        with open(fin, "rb") as f:
            data = pickle.loads(f.read())
        if data == "Done":
            break
        t = time.time(); refs = {}
        if a.refs:  # nearest recorded frame to the ego position, same frame index for the three cameras
            scene, ego = data.pop("_scene"), data.pop("_ego")
            if scene not in paths:
                paths[scene] = np.load(f"{a.refs}/{scene}/path.npy")
            i = 0 if ego is None else int(np.argmin(((paths[scene] - np.asarray(ego, float)) ** 2).sum(-1))); ref_idx.append(i)
            refs = {k: Image.open(f"{a.refs}/{scene}/{k}/{i:05d}.png").convert("RGB") for k in data}
        out = {k: np.array(FX(a.method, Image.fromarray(v), refs.get(k))) for k, v in data.items()}
        times.append(time.time() - t)
        with open(fout, "wb") as f:
            f.write(pickle.dumps(out))
    print(f"fixer done, {len(times)} calls, median {np.median(times)*1000:.0f} ms per call ({len(data) if isinstance(data, dict) else 3} cams)"
          + (f", reference frames {min(ref_idx)}-{max(ref_idx)}" if ref_idx else ""), flush=True)


if __name__ == "__main__":
    main()
