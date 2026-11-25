"""Frozen NVIDIA Fixer (nv-tlabs/Fixer, Cosmos-Predict2 0.6B, 1-step, timestep 250) applied to a list of images.
Runs in the Fixer env (/env, default /nv_fixer/env), not the project env;
use src/fixers/nv_fixer/run.sh.

Differences from the official inference script (src/inference_pretrained_model.py), all documented in docs/nv_fixer.md:
  * Transformer Engine is replaced by a pure-PyTorch shim (te_shim/), since TE has no prebuilt PyTorch wheel and we
    cannot use NVIDIA's Docker container on sbel;
  * no torch.compile (compile time dominates for a few hundred images); batch size 1; bf16 as in the official script;
  * resolution: the official 960-wide setting (960x544); our 960x540 renders are resized to 960x544 (bilinear, as the
    official script does) and the output is resized back to the input size (bilinear, as the official script does).

usage: python infer.py --list pairs.txt  (lines "input_path<TAB>output_path")  [--timestep 250] [--timing out.json]
       python infer.py --paralane scene_16,scene_17   (eval renders -> $DATA/fixed/nv_fixer/<rel path>)
       python infer.py --hugsim scene-0064,...        (HUGSIM offset renders -> $DATA/hugsim/fixed/nv_fixer/<rel>)
"""
import argparse, json, os, sys, time, warnings
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # src/ (this file runs in the Fixer env, not via py.sh)
from common.config import DATA  # noqa: E402

FIX = os.path.expanduser(os.environ.get("NV_FIXER_ROOT", os.path.join(DATA, "nv_fixer")))  # repo/, env/, weights/
sys.path.insert(0, os.path.join(HERE, "te_shim"))
sys.path.insert(0, os.path.join(FIX, "repo", "src"))
warnings.filterwarnings("ignore")


def import_fixer_module():
    """Import the official Fixer model module with our TE shim. Megatron-core (needed by cosmos-predict2 only for
    parallel_state) must be imported while TE is hidden, otherwise it wraps TE classes the shim does not provide."""
    sys.modules["transformer_engine"] = None
    import megatron.core  # noqa: F401
    from megatron.core import parallel_state  # noqa: F401
    del sys.modules["transformer_engine"]
    import transformer_engine  # noqa: F401  (our shim; the official script imports TE first as well)
    import pix2pix_turbo_nocond_cosmos_base_faster_tokenizer as P
    return P


def load_fixer(timestep=250, vae_skip_connection=False, pretrained=True, dtype=torch.bfloat16):
    P = import_fixer_module()
    P.config.dit_path = f"{FIX}/weights/base/model_fast_tokenizer.pt"
    P.config.tokenizer["vae_pth"] = f"{FIX}/weights/base/tokenizer_fast.pth"
    m = P.Pix2Pix_Turbo(pretrained_path=f"{FIX}/weights/pretrained/pretrained_fixer.pkl" if pretrained else None,
                        timestep=timestep, vae_skip_connection=vae_skip_connection, batch_size=1)
    return m.to(device="cuda", dtype=dtype)


@torch.no_grad()
def run_one(model, img, size=(960, 544), dtype=torch.bfloat16):
    W0, H0 = img.size
    x = transforms.Normalize([0.5], [0.5])(transforms.ToTensor()(img.resize(size, Image.BILINEAR)))[None].to("cuda", dtype)
    with torch.autocast("cuda", dtype=dtype):
        y = model(x)
    y = (y.float()[0].cpu() * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(y).resize((W0, H0), Image.BILINEAR)


def paralane_items(scenes, stride=2):
    out = []
    for s in scenes:
        meta = [m for m in json.load(open(f"{DATA}/pairs/{s}/meta.json")) if m["variant"] == "full"]
        for k, m in enumerate(meta):  # same selection as the project's eval (run_fixers.eval_items)
            if m["onpath_test"] or k % stride == 0:
                out.append((f"{DATA}/pairs/{m['path']}", f"{DATA}/fixed/nv_fixer/{m['path']}"))
    return out


def hugsim_items(scenes):
    out = []
    R = f"{DATA}/hugsim/renders"
    for s in scenes:
        for rec in json.load(open(f"{R}/{s}/index.json")):
            for off, v in rec["views"].items():
                out.append((f"{R}/{v['path']}", f"{DATA}/hugsim/fixed/nv_fixer/{v['path']}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list"); ap.add_argument("--paralane"); ap.add_argument("--hugsim")
    ap.add_argument("--timestep", type=int, default=250); ap.add_argument("--timing", default=None)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--skip_existing", action="store_true")
    a = ap.parse_args()
    items = []
    if a.list:
        items += [tuple(l.rstrip("\n").split("\t")) for l in open(a.list) if l.strip()]
    if a.paralane:
        items += paralane_items(a.paralane.split(","))
    if a.hugsim:
        items += hugsim_items(a.hugsim.split(","))
    if a.skip_existing:
        items = [it for it in items if not os.path.exists(it[1])]
    if a.limit:
        items = items[:a.limit]
    print(len(items), "images", flush=True)
    t0 = time.time(); model = load_fixer(a.timestep); print(f"model loaded in {time.time()-t0:.0f}s", flush=True)
    torch.cuda.reset_peak_memory_stats(); times = []
    for k, (src, dst) in enumerate(items):
        img = Image.open(src).convert("RGB")
        torch.cuda.synchronize(); t = time.time()
        o = run_one(model, img)
        torch.cuda.synchronize(); times.append(time.time() - t)
        os.makedirs(os.path.dirname(dst), exist_ok=True); o.save(dst)
        if k % 100 == 0:
            print(k, f"{times[-1]*1000:.0f} ms", flush=True)
    res = dict(ms_per_img_median=float(np.median(times[3:]) * 1000) if len(times) > 3 else None, n=len(times),
               peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30, timestep=a.timestep, resolution="960x544 (resized back)")
    print(res, flush=True)
    if a.timing:
        old = json.load(open(a.timing)) if os.path.exists(a.timing) else {}
        old[a.paralane or a.hugsim or a.list] = res
        json.dump(old, open(a.timing, "w"), indent=1)


if __name__ == "__main__":
    main()
