"""Run a Difix model on (a) the held-out VAL triplets of the pairs JSON or (b) the test-scene eval renders.

Models:
  ours       our re-implementation (model.py) with a checkpoint from train.py (best.pt / last.pt)
  official   a checkpoint of the OFFICIAL trainer (model_*.pkl), loaded with the official Difix class
  released   NVIDIA's released weights: nvidia/difix_ref (with reference) or nvidia/difix (--no_ref)
All run with the same reference view (nearest real frame of the source lane) at 1024x576 and are resized back to
the render size (960x540).

usage:
  python infer.py --model ours --ckpt best.pt --val --out DIR                         # val triplets -> DIR/<key>.png
  python infer.py --model ours --ckpt best.pt --scenes scene_16,scene_17 --method difix_reimpl_ref   # -> fixed/<method>/...
  python infer.py --model released --val --json $DATA/dl3dv/dl3dv_test.json --out DIR    # any pairs JSON (DL3DV)
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import ref_path, config

DX = os.path.join(config.DATA, "difix_reimpl")
PAIRS = os.path.join(config.DATA, "pairs")
FIXED = os.path.join(config.DATA, "fixed")
SIZE = (1024, 576)  # W, H


def eval_items(scene, stride=2):
    meta = [m for m in json.load(open(f"{PAIRS}/{scene}/meta.json")) if m["variant"] == "full"]
    return [m for k, m in enumerate(meta) if m["onpath_test"] or k % stride == 0]


def load(kind, ckpt, use_ref):
    to_t = lambda im: torch.from_numpy(np.array(im.convert("RGB").resize(SIZE, Image.BICUBIC))).permute(2, 0, 1).float().cuda() / 127.5 - 1
    if kind == "ours":
        from model import DifixRe
        net = DifixRe().cuda().eval()
        net.load_trained_state_dict({k: v.float() for k, v in torch.load(ckpt, map_location="cuda").items()} if not ckpt.endswith("last.pt")
                                    else torch.load(ckpt, map_location="cuda")["model"])

        @torch.no_grad()
        def f(img, ref):
            x = torch.stack([to_t(img)] + ([to_t(ref)] if use_ref else []))[None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = net(x)[0].float()
            return Image.fromarray(((o.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)).resize(img.size, Image.BICUBIC)
        return f
    if kind == "official":
        sys.path.insert(0, f"{DX}/official_src")
        from model import Difix as OfficialDifix   # noqa (official class, for the official trainer's checkpoint)
        net = OfficialDifix(pretrained_path=ckpt, timestep=199, mv_unet=use_ref); net.set_eval()

        @torch.no_grad()
        def f(img, ref):
            return net.sample(img, width=SIZE[0], height=SIZE[1], ref_image=ref if use_ref else None, prompt="remove degradation")
        return f
    sys.path.insert(0, config.DIFIX3D_SRC)
    from pipeline_difix import DifixPipeline
    pipe = DifixPipeline.from_pretrained("nvidia/difix_ref" if use_ref else "nvidia/difix", trust_remote_code=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def f(img, ref):
        kw = dict(image=img.resize(SIZE, Image.BICUBIC), num_inference_steps=1, timesteps=[199], guidance_scale=0.0)
        if use_ref:
            kw["ref_image"] = ref.resize(SIZE, Image.BICUBIC)
        return pipe("remove degradation", **kw).images[0].resize(img.size, Image.BICUBIC)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["ours", "official", "released"]); ap.add_argument("--ckpt", default=None)
    ap.add_argument("--no_ref", action="store_true"); ap.add_argument("--val", action="store_true")
    ap.add_argument("--out", default=None); ap.add_argument("--scenes", default=None); ap.add_argument("--method", default=None)
    ap.add_argument("--size", default="576x1024", help="HxW the model runs at")
    ap.add_argument("--json", default=None, help="pairs JSON for --val (default: the driving pairs); its 'test' split is run")
    a = ap.parse_args()
    global SIZE
    h, w = map(int, a.size.split("x")); SIZE = (w, h)
    f = load(a.model, a.ckpt, not a.no_ref)
    jobs = []
    if a.val:
        for key, it in json.load(open(a.json or f"{DX}/driving_pairs.json"))["test"].items():
            jobs.append((it["image"], it["ref_image"], f"{a.out}/{key}.png"))
    else:
        cache = {}
        for s in a.scenes.split(","):
            for m in eval_items(s):
                jobs.append((f"{PAIRS}/{m['path']}", ref_path(PAIRS, m, cache), f"{FIXED}/{a.method}/{m['path']}"))
    times = []
    for src, ref, out in jobs:
        img, rim = Image.open(src).convert("RGB"), Image.open(ref).convert("RGB")
        torch.cuda.synchronize(); t = time.time(); o = f(img, rim); torch.cuda.synchronize(); times.append(time.time() - t)
        os.makedirs(os.path.dirname(out), exist_ok=True); o.save(out)
    print(f"{len(jobs)} images, {np.median(times[3:]) * 1000:.0f} ms/img median, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
