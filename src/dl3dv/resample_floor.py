"""How much does running a model at 448x768 (and resizing back to 960x540) cost by itself? The raw render resized
960x540 -> 768x448 -> 960x540 (bicubic, as infer.py) vs. the real target: PSNR / LPIPS-Alex, both levels.
-> results/dl3dv/resample_floor.json.  usage: CUDA_VISIBLE_DEVICES= scripts/lib/py.sh src/dl3dv/resample_floor.py"""
import json
import numpy as np
import torch, lpips
from PIL import Image
from common.config import DATA, REPO

lp = lpips.LPIPS(net="alex", verbose=False).eval(); out = {}
for L in (30, 60):
    T = json.load(open(f"{DATA}/dl3dv/dl3dv_test_s{L}.json"))["test"]; P, Lp, P0, L0 = [], [], [], []
    for k, it in T.items():
        x = Image.open(it["image"]).convert("RGB"); y = np.array(Image.open(it["target_image"]).convert("RGB")) / 255.0
        xr = np.array(x.resize((768, 448), Image.BICUBIC).resize(x.size, Image.BICUBIC)) / 255.0; x0 = np.array(x) / 255.0
        t = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].float() * 2 - 1
        with torch.no_grad():
            for a, PP, LL in ((xr, P, Lp), (x0, P0, L0)):
                PP.append(float(-10 * np.log10(((a - y) ** 2).mean()))); LL.append(float(lp(t(a), t(y))))
    out[f"s{L}"] = dict(n=len(P), raw_psnr=float(np.mean(P0)), raw_lpips=float(np.mean(L0)),
                        raw_resampled_448_psnr=float(np.mean(P)), raw_resampled_448_lpips=float(np.mean(Lp)))
    print(L, out[f"s{L}"], flush=True)
json.dump(out, open(f"{REPO}/results/dl3dv/resample_floor.json", "w"), indent=1)
