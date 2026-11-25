"""Full fine-tune of the released Difix (nvidia/difix) U-Net on our training pairs (same pairs/sampling as our U-Net).
fp32 master weights + 8-bit AdamW (bitsandbytes), bf16 autocast, gradient checkpointing, 512x512 crops, batch 1.
VAE and text encoder frozen (as in Difix, the VAE skip-LoRA stays at the released values).
Loss: MSE + LPIPS-VGG (movable objects masked). Cosine LR with warm-up over --total_steps; every --val_every steps the
model is scored on the VAL scene (full frame, LPIPS-Alex vs real) and the best U-Net is saved (bf16) to <out>/best_unet.pt.

usage: scripts/lib/py.sh src/fixers/difix_ft/finetune.py --pairs <root> --scenes a,b --val_scenes c --out <dir> --total_steps 5000 --minutes 58
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import lpips

from common.config import DIFIX3D_SRC
from fixers.unet.train import Pairs, val_items

sys.path.insert(0, DIFIX3D_SRC)  # released Difix pipeline code (initialization from nvidia/difix)


def difix_forward(pipe, x, emb, t):
    """Differentiable 1-step Difix: x in [-1,1] (B,3,H,W) -> output in [-1,1]."""
    sf = pipe.vae.config.scaling_factor
    lat = pipe.vae.encode(x).latent_dist.sample() * sf
    pipe.vae.decoder.incoming_skip_acts = pipe.vae.encoder.current_down_blocks  # Difix's encoder->decoder skips
    eps = pipe.unet(lat, t, encoder_hidden_states=emb, return_dict=False)[0]
    lat2 = pipe.scheduler.step(eps, t, lat, return_dict=False)[0]
    return pipe.vae.decode(lat2 / sf, return_dict=False)[0].clamp(-1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True); ap.add_argument("--scenes", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=58); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--total_steps", type=int, default=5000); ap.add_argument("--val_scenes", default="")
    ap.add_argument("--val_every", type=int, default=1000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from pipeline_difix import DifixPipeline
    from diffusers.utils import torch_utils  # noqa
    pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(timesteps=[199], device="cuda")
    t = pipe.scheduler.timesteps[0]
    with torch.no_grad():
        emb = pipe.encode_prompt("remove degradation", "cuda", 1, False)[0]

    # sanity: our differentiable forward reproduces the pipeline output
    ds = Pairs(a.pairs, a.scenes.split(","))
    from PIL import Image
    m = ds.items[0]; img = Image.open(f"{a.pairs}/{m['path']}").convert("RGB").crop((0, 0, 512, 512))
    with torch.no_grad():
        ref = torch.from_numpy(np.array(pipe("remove degradation", image=img, num_inference_steps=1, timesteps=[199],
                                             guidance_scale=0.0).images[0])).float().cuda() / 255
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1)[None].float().cuda() / 255 * 2 - 1
        mine = (difix_forward(pipe, x, emb, t)[0].permute(1, 2, 0) + 1) / 2
    print(f"sanity |ours - pipeline| mean {float((mine - ref).abs().mean()):.4f}", flush=True)

    pipe.vae.requires_grad_(False); pipe.text_encoder.requires_grad_(False)
    pipe.unet.train(); pipe.unet.enable_gradient_checkpointing()
    try:
        pipe.unet.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print("no xformers:", e)
    import bitsandbytes as bnb
    params = [p for p in pipe.unet.parameters()]
    opt = bnb.optim.AdamW8bit(params, lr=a.lr, weight_decay=0)
    lp = lpips.LPIPS(net="vgg", verbose=False).cuda().eval().requires_grad_(False)
    sampler = torch.utils.data.WeightedRandomSampler(ds.weights, num_samples=10 ** 7, replacement=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, sampler=sampler, num_workers=6, drop_last=True)
    vitems = val_items(a.pairs, a.val_scenes.split(","), every=6) if a.val_scenes else []
    lp_alex = lpips.LPIPS(net="alex", verbose=False).cuda().eval().requires_grad_(False)
    warm = 100
    lr_at = lambda s: a.lr * (s / warm if s < warm else 0.5 * (1 + np.cos(np.pi * min(1.0, (s - warm) / max(1, a.total_steps - warm)))))
    best = dict(val_lpips=1e9)

    @torch.no_grad()
    def validate():
        pipe.unet.eval(); ls = []
        for m in vitems:
            im = Image.open(f"{a.pairs}/{m['path']}").convert("RGB"); W0, H0 = im.size
            x = torch.from_numpy(np.array(im.resize((W0, H0 - H0 % 8), Image.LANCZOS))).permute(2, 0, 1)[None].float().cuda() / 255
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = (difix_forward(pipe, x * 2 - 1, emb, t) + 1) / 2
            o = F.interpolate(o.float(), size=(H0, W0), mode="bilinear", align_corners=False)
            y = torch.from_numpy(np.array(Image.open(f"{a.pairs}/{m['real']}"))).permute(2, 0, 1)[None].float().cuda() / 255
            fg = torch.from_numpy(np.array(Image.open(f"{a.pairs}/{m['fg']}")) > 127)[None, None].float().cuda()
            o = o * (1 - fg) + y * fg
            ls.append(float(lp_alex(o * 2 - 1, y * 2 - 1)))
        pipe.unet.train()
        return float(np.mean(ls))

    if vitems:
        v0 = validate(); log0 = dict(step=0, val_lpips=v0); print(f"VAL step 0 (released Difix): lpips {v0:.4f}", flush=True)
    t0 = time.time(); step = 0; run = []; log = [log0] if vitems else []
    torch.cuda.reset_peak_memory_stats()
    while time.time() - t0 < a.minutes * 60 and step < a.total_steps:
        for x, y, fg in dl:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            x, y, fg = x.cuda(), y.cuda(), fg.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = (difix_forward(pipe, x * 2 - 1, emb, t) + 1) / 2
            o = o.float(); k = 1 - fg; o_p = o * k + y * fg
            l2 = (((o - y) ** 2) * k).sum() / (k.sum() * 3 + 1)
            llp = lp(o_p * 2 - 1, y * 2 - 1).mean()
            loss = l2 + llp
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); step += 1
            run.append([l2.item(), llp.item()])
            if step % 50 == 0:
                r = np.mean(run, 0); run = []; log.append(dict(step=step, l2=float(r[0]), lpips=float(r[1])))
                print(f"step {step} l2 {r[0]:.4f} lpips {r[1]:.4f} {time.time()-t0:.0f}s peak {torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
            if vitems and (step % a.val_every == 0 or step == a.total_steps):
                vl = validate(); log.append(dict(step=step, val_lpips=vl)); tag = ""
                if vl < best["val_lpips"]:
                    best = dict(val_lpips=vl, step=step); tag = " *best*"
                    torch.save({k: v.to(torch.bfloat16) for k, v in pipe.unet.state_dict().items()}, f"{a.out}/best_unet.pt")
                print(f"VAL step {step}: lpips {vl:.4f}{tag}", flush=True)
            if time.time() - t0 > a.minutes * 60 or step >= a.total_steps:
                break
    json.dump(dict(log=log, best=best, args=vars(a), counts=ds.counts), open(f"{a.out}/ft_log.json", "w"))
    print(f"done after {step} steps; best {best}", flush=True)


if __name__ == "__main__":
    main()
