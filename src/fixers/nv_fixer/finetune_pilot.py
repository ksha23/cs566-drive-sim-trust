"""24 GB fine-tune pilot for NVIDIA Fixer on our Para-Lane driving pairs (NOT a full training run).

Mirrors the core of the official trainer (src/train_pix2pix_turbo_nocond_cosmos_base_faster_tokenizer.py):
  trainable = full DiT (674M) + full VAE except its time_conv layers (official defaults: train_full_unet, VAE not frozen),
  input/target resized to 1024x576 ("resize_576x1024"), 1-step denoising at timestep 250,
  loss = 1.0 * L2 + 0.3 * LPIPS-VGG on a random 128..512 square crop (lambda values of the README fine-tune command),
  AdamW lr 2e-5, grad clip 1.0.
Deliberately omitted in the pilot (documented): the CLIP-based GAN discriminator and CLIP-similarity loss
(vision-aided-loss / OpenAI CLIP deps), the Gram loss (starts only after 2000 warm-up steps anyway), accelerate/multi-GPU.
Movable-object pixels are masked out of the L2 (as in all our fixers).
The official tokenizer.patch (removing @torch.no_grad from the tokenizer encode/decode) is applied at runtime by
unwrapping those methods.

Phase 1 (--probe): a few steps for each memory configuration, reporting peak memory and s/step:
  bf16_adamw     official-like: whole model cast to bf16, torch AdamW (states in bf16)
  bf16_adam8     bf16 weights + bitsandbytes AdamW8bit
  fp32m_adam8    fp32 master weights + AdamW8bit + bf16 autocast
Phase 2 (--steps N --config C): short fine-tune with val LPIPS on scene_19 before and after.
"""
import argparse, json, os, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from infer import load_fixer, import_fixer_module, DATA, FIX  # noqa: E402

TRAIN = ["scene_20", "scene_21", "scene_22", "scene_23", "scene_24"]
SIZE = (1024, 576)


def unpatch_no_grad():
    import inspect
    import cosmos_predict2.tokenizers.tokenizer as T
    n = 0
    for _, cls in inspect.getmembers(T, inspect.isclass):
        for name in ("encode", "decode"):
            f = cls.__dict__.get(name)
            if f is not None and hasattr(f, "__wrapped__"):
                setattr(cls, name, f.__wrapped__); n += 1
    return n


def load_pairs(scenes):
    it = []
    for s in scenes:
        it += [m for m in json.load(open(f"{DATA}/pairs/{s}/meta.json")) if not m["onpath_test"]]
    return it


def to_t(p, size=SIZE):
    x = torch.from_numpy(np.array(Image.open(p).convert("RGB").resize(size, Image.BILINEAR))).permute(2, 0, 1).float() / 255
    return x * 2 - 1


def batch(items, rng):
    m = items[rng.integers(len(items))]
    x, y = to_t(f"{DATA}/pairs/{m['path']}"), to_t(f"{DATA}/pairs/{m['real']}")
    fg = torch.from_numpy(np.array(Image.open(f"{DATA}/pairs/{m['fg']}").resize(SIZE, Image.NEAREST)) > 127)[None].float()
    if rng.random() < 0.5:
        x, y, fg = x.flip(-1), y.flip(-1), fg.flip(-1)
    return x[None].cuda(), y[None].cuda(), fg[None].cuda()


def build(cfg):
    model = load_fixer(timestep=250, pretrained=True, dtype=torch.float32 if cfg.startswith("fp32") else torch.bfloat16)
    model.set_train()
    params = [p for p in list(model.unet.parameters()) + list(model.vae.parameters()) if p.requires_grad]
    if cfg == "bf16_adamw":
        opt = torch.optim.AdamW(params, lr=2e-5, weight_decay=1e-2)
    else:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=2e-5, weight_decay=1e-2)
    return model, params, opt


def step(model, params, opt, lp, x, y, fg, rng):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = model(x.to(next(model.unet.parameters()).dtype))
    o = o.float(); k = 1 - fg
    l2 = (((o - y) ** 2) * k).sum() / (k.sum() * 3 + 1)
    c = int(rng.integers(128, 513)); t, l = int(rng.integers(0, SIZE[1] - c + 1)), int(rng.integers(0, SIZE[0] - c + 1))
    op = (o * k + y * fg)[..., t:t + c, l:l + c]
    llp = lp(op, y[..., t:t + c, l:l + c]).mean()
    loss = l2 + 0.3 * llp
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    return l2.item(), llp.item()


@torch.no_grad()
def validate(model, items, lp_alex):
    model.eval(); ls = []
    for m in items:
        x = to_t(f"{DATA}/pairs/{m['path']}", (960, 544))[None].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(x.to(next(model.unet.parameters()).dtype))
        o = F.interpolate(o.float(), size=(540, 960), mode="bilinear", align_corners=False)
        y = torch.from_numpy(np.array(Image.open(f"{DATA}/pairs/{m['real']}").convert("RGB"))).permute(2, 0, 1)[None].float().cuda() / 255 * 2 - 1
        fg = torch.from_numpy(np.array(Image.open(f"{DATA}/pairs/{m['fg']}")) > 127)[None, None].float().cuda()
        o = o * (1 - fg) + y * fg
        ls.append(float(lp_alex(o.clamp(-1, 1), y)))
    model.set_train()
    return float(np.mean(ls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true"); ap.add_argument("--config", default="fp32m_adam8")
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--minutes", type=float, default=35)
    ap.add_argument("--out", default=os.path.join(FIX, "pilot"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import_fixer_module(); print("unwrapped no_grad on", unpatch_no_grad(), "tokenizer methods", flush=True)
    import lpips
    lp = lpips.LPIPS(net="vgg", verbose=False).cuda().eval().requires_grad_(False)
    items = load_pairs(TRAIN); rng = np.random.default_rng(0); random.seed(0); torch.manual_seed(0)
    res = json.load(open(f"{a.out}/pilot.json")) if os.path.exists(f"{a.out}/pilot.json") else {}
    if a.probe:
        for cfg in ["bf16_adamw", "bf16_adam8", "fp32m_adam8"]:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                model, params, opt = build(cfg); ts = []
                for s in range(6):
                    x, y, fg = batch(items, rng)
                    torch.cuda.synchronize(); t = time.time()
                    l2, llp = step(model, params, opt, lp, x, y, fg, rng)
                    torch.cuda.synchronize(); ts.append(time.time() - t)
                r = dict(ok=True, peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                         peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30, s_per_step=float(np.median(ts[2:])),
                         trainable_M=sum(p.numel() for p in params) / 1e6, last_l2=l2, last_lpips=llp)
            except torch.OutOfMemoryError as e:
                r = dict(ok=False, error="OOM", peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30, msg=str(e)[:300])
            print(cfg, r, flush=True); res[f"probe_{cfg}"] = r
            model = params = opt = None; torch.cuda.empty_cache()
            json.dump(res, open(f"{a.out}/pilot.json", "w"), indent=1)
        return
    # phase 2: short fine-tune with val before/after
    val = [m for s in ["scene_19"] for m in json.load(open(f"{DATA}/pairs/{s}/meta.json")) if m["variant"] == "full"]
    val = [m for k, m in enumerate(val) if m["onpath_test"] or k % 12 == 0][:60]
    lp_alex = lpips.LPIPS(net="alex", verbose=False).cuda().eval().requires_grad_(False)
    model, params, opt = build(a.config); torch.cuda.reset_peak_memory_stats()
    v0 = validate(model, val, lp_alex); print(f"VAL before: lpips {v0:.4f} ({len(val)} imgs)", flush=True)
    t0 = time.time(); run = []; log = []; s = 0
    while s < a.steps and time.time() - t0 < a.minutes * 60:
        x, y, fg = batch(items, rng); run.append(step(model, params, opt, lp, x, y, fg, rng)); s += 1
        if s % 25 == 0:
            r = np.mean(run, 0); run = []; log.append(dict(step=s, l2=float(r[0]), lpips=float(r[1])))
            print(f"step {s} l2 {r[0]:.4f} lpips {r[1]:.4f} {time.time()-t0:.0f}s", flush=True)
    v1 = validate(model, val, lp_alex); print(f"VAL after {s} steps: lpips {v1:.4f}", flush=True)
    res[f"finetune_{a.config}"] = dict(steps=s, s_per_step=(time.time() - t0) / max(s, 1), val_lpips_before=v0,
                                        val_lpips_after=v1, n_val=len(val), peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                                        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30, log=log)
    json.dump(res, open(f"{a.out}/pilot.json", "w"), indent=1)


if __name__ == "__main__":
    main()
