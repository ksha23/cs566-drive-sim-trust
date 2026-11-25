"""Stdin/stdout server around NVIDIA Fixer so code in the project env (fixers/registry.py -> progressive loop) can call
it: reads length-prefixed uint8 HxWx3 frames, writes the fixed frames back. Runs in the Fixer env (run.sh)."""
import os, struct, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer import load_fixer, run_one


def read_exact(f, n):
    b = b""
    while len(b) < n:
        c = f.read(n - len(b))
        if not c:
            return None
        b += c
    return b


def main():
    model = load_fixer(); inp = sys.stdin.buffer
    out = os.fdopen(os.dup(1), "wb"); os.dup2(2, 1)  # keep the protocol pipe clean of library prints
    out.write(b"READY\n"); out.flush()
    while True:
        hdr = read_exact(inp, 8)
        if hdr is None:
            break
        h, w = struct.unpack("<II", hdr)
        img = np.frombuffer(read_exact(inp, h * w * 3), np.uint8).reshape(h, w, 3)
        y = np.asarray(run_one(model, Image.fromarray(img)), np.uint8)
        out.write(struct.pack("<II", *y.shape[:2]) + y.tobytes()); out.flush()


if __name__ == "__main__":
    main()
