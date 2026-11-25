"""Uniform interface to the fixers: load_fixer(name, ckpt) -> fix(img) with img a (H,W,3) float tensor in [0,1] on
cuda, returning the same shape. Names: none | ours (our U-Net, ckpt = best.pt) | difix (released) | difix_ft (released
Difix with our fine-tuned U-Net, ckpt = best_unet.pt) | difix_ref (needs ref=...) | nv_fixer (NVIDIA Fixer [4]; runs in its
own env as a subprocess server, src/fixers/nv_fixer/server.py)."""
import os, sys
import numpy as np
import torch
from PIL import Image
from common.config import DIFIX3D_SRC

sys.path.insert(0, DIFIX3D_SRC)  # released Difix pipeline code


def load_fixer(name, ckpt=None):
    if name == "none":
        return lambda img, ref=None: img
    if name == "nv_fixer":
        return _nv_fixer_client()
    if name == "difix_reimpl":  # our Difix reimplementation [1] (with reference view), ckpt = best.pt of train.py
        return _difix_reimpl(ckpt)
    if name == "ours":
        from fixers.unet.model import UNetFixer
        s = torch.load(ckpt, map_location="cuda")
        net = UNetFixer(base=s.get("base", 48)).cuda().eval(); net.load_state_dict(s["net"])

        @torch.no_grad()
        def fix(img, ref=None):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return net(img.permute(2, 0, 1)[None])[0].float().permute(1, 2, 0).clamp(0, 1)
        return fix
    from pipeline_difix import DifixPipeline
    pipe = DifixPipeline.from_pretrained("nvidia/difix_ref" if name == "difix_ref" else "nvidia/difix", trust_remote_code=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    if name == "difix_ft":
        pipe.unet.load_state_dict({k: v.float() for k, v in torch.load(ckpt, map_location="cuda").items()})
    to_pil = lambda x: Image.fromarray((x.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8))

    @torch.no_grad()
    def fix(img, ref=None):
        im = to_pil(img)
        kw = dict(image=im, num_inference_steps=1, timesteps=[199], guidance_scale=0.0)
        if name == "difix_ref":
            kw["ref_image"] = to_pil(ref) if torch.is_tensor(ref) else ref
        o = pipe("remove degradation", **kw).images[0].resize(im.size, Image.BICUBIC)
        return torch.from_numpy(np.array(o)).float().cuda() / 255
    return fix


def _nv_fixer_client():
    import struct, subprocess
    from common.config import REPO
    p = subprocess.Popen([f"{REPO}/src/fixers/nv_fixer/run.sh", f"{REPO}/src/fixers/nv_fixer/server.py"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    line = p.stdout.readline()
    while line and line.strip() != b"READY":
        line = p.stdout.readline()
    assert line, "nv_fixer server failed to start"

    def fix(img, ref=None):
        x = (img.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
        p.stdin.write(struct.pack("<II", *x.shape[:2]) + x.tobytes()); p.stdin.flush()
        h, w = struct.unpack("<II", p.stdout.read(8))
        y = np.frombuffer(p.stdout.read(h * w * 3), np.uint8).reshape(h, w, 3)
        return torch.from_numpy(y.copy()).float().cuda() / 255
    fix.proc = p
    return fix


def _difix_reimpl(ckpt):
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "difix_reimpl")
    sys.path.insert(0, here)
    from infer import load as load_reimpl
    f = load_reimpl("ours", ckpt, use_ref=True)
    to_pil = lambda x: Image.fromarray((x.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8))

    def fix(img, ref=None):
        assert ref is not None, "difix_reimpl needs a reference image"
        o = f(to_pil(img), to_pil(ref) if torch.is_tensor(ref) else ref)
        return torch.from_numpy(np.array(o)).float().cuda() / 255
    return fix
