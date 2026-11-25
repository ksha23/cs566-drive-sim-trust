"""Apply one fixer to every image in a folder (command-line tool; the fixers of the results, fixers/registry.py).

  --method difix             released Difix (nvidia/difix), 1 step
           difix_ref         released Difix-ref; needs --ref_dir (reference image with the same file name)
           difix_ft          released Difix with our fine-tuned U-Net (--ckpt best_unet.pt)
           difix_reimpl      our Difix-ref reimplementation (--ckpt best.pt, needs --ref_dir; run with the
                             reimplementation venv, see src/fixers/difix_reimpl/)
           nv_fixer          NVIDIA Fixer (runs in its own env as a subprocess; src/fixers/nv_fixer/)
           ours              our 15M-parameter U-Net (--ckpt best.pt)
usage: scripts/lib/py.sh src/tools/fix_folder.py <in_dir> <out_dir> --method difix [--ckpt P] [--ref_dir D]
example: scripts/lib/py.sh src/tools/fix_folder.py /tmp/offpath_3m /tmp/offpath_3m_difix --method difix
"""
import argparse, glob, os, time
import numpy as np
import torch
from PIL import Image

from fixers.registry import load_fixer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--method", required=True)
    ap.add_argument("--ckpt", default=None); ap.add_argument("--ref_dir", default=None)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    fix = load_fixer(a.method, a.ckpt)
    files = sorted(f for e in ("png", "jpg", "jpeg") for f in glob.glob(f"{a.inp}/*.{e}"))
    load = lambda p: torch.from_numpy(np.array(Image.open(p).convert("RGB"))).float().cuda() / 255
    t0 = time.time()
    for p in files:
        ref = load(f"{a.ref_dir}/{os.path.basename(p)}") if a.ref_dir else None
        y = fix(load(p), ref)
        Image.fromarray((y.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(
            f"{a.out}/{os.path.splitext(os.path.basename(p))[0]}.png")
    print(f"{a.method}: {len(files)} images -> {a.out} ({(time.time() - t0) / max(len(files), 1):.2f} s/img)")


if __name__ == "__main__":
    main()
