"""Image metrics of several fixers on the held-out VAL triplets (PSNR / SSIM / LPIPS-Alex vs. the real target,
movable objects pasted from the target; plus distribution metrics FID and KID over the set, torch-fidelity).
usage: python val_metrics.py out.json name=dir [name=dir ...]      (dir holds <key>.png; name "raw" = the renders)
       PAIRS_JSON=<json> TARGETS_DIR=<dir> python val_metrics.py ...  (another pairs JSON, e.g. DL3DV; default: driving)
Also writes per-image rows (<out>_rows.json) so CIs can be bootstrapped (e.g. over scenes).
"""
import json, os, sys, tempfile
import numpy as np
import torch
import lpips
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import config  # noqa: E402
DX = os.path.join(config.DATA, "difix_reimpl")


def main():
    out = sys.argv[1]; methods = dict(x.split("=", 1) for x in sys.argv[2:])
    items = json.load(open(os.environ.get("PAIRS_JSON", f"{DX}/driving_pairs.json")))["test"]
    lp = lpips.LPIPS(net="alex", verbose=False).cuda().eval(); ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    import torch_fidelity
    tgt_dir = os.environ.get("TARGETS_DIR", f"{DX}/val_targets")
    if not os.path.isdir(tgt_dir):
        os.makedirs(tgt_dir)
        for k, it in items.items():
            Image.open(it["target_image"]).convert("RGB").save(f"{tgt_dir}/{k}.png")
    res, rows = {}, []
    for name, d in methods.items():
        P, S, L = [], [], []
        with tempfile.TemporaryDirectory() as tmp:
            for k, it in items.items():
                p = it["image"] if name == "raw" else f"{d}/{k}.png"
                x = np.array(Image.open(p).convert("RGB")).astype(np.float32) / 255
                y = np.array(Image.open(it["target_image"]).convert("RGB")).astype(np.float32) / 255
                fg = (np.array(Image.open(it["fg"])) > 127)[..., None]
                x = np.where(fg, y, x); Image.fromarray((x * 255 + .5).astype(np.uint8)).save(f"{tmp}/{k}.png")
                X, Y = (torch.from_numpy(a).permute(2, 0, 1)[None].cuda() for a in (x, y))
                P.append(float(-10 * torch.log10(((X - Y) ** 2).mean()))); S.append(float(ssim(X, Y))); L.append(float(lp(X * 2 - 1, Y * 2 - 1)))
                rows.append(dict(method=name, key=k, scene=it.get("scene", ""), psnr=P[-1], ssim=S[-1], lpips=L[-1]))
            m = torch_fidelity.calculate_metrics(input1=tmp, input2=tgt_dir, cuda=True, fid=True, kid=True, verbose=False,
                                                 kid_subset_size=min(100, len(items)))
        res[name] = dict(n=len(items), psnr=float(np.mean(P)), ssim=float(np.mean(S)), lpips=float(np.mean(L)),
                         fid=float(m["frechet_inception_distance"]), kid=float(m["kernel_inception_distance_mean"]))
        print(name, {k: round(v, 4) for k, v in res[name].items()}, flush=True)
    json.dump(res, open(out, "w"), indent=1)
    json.dump(rows, open(out.replace(".json", "_rows.json"), "w"))


if __name__ == "__main__":
    main()
