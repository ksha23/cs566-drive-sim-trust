"""Our Difix training loop (re-implementation, see model.py / losses.py / data.py).

Optimisation (paper silent -> our choices, see docs/IMPLEMENTATION_NOTES_difix.md): fp32 master weights,
bitsandbytes 8-bit AdamW (lr 2e-5, wd 1e-2), linear warm-up then constant LR, bf16 autocast, U-Net gradient
checkpointing, batch 1, grad-norm clip 1.0. Time-boxed and resumable (last.pt keeps model + optimizer + step);
every --val_every steps the model is scored on the JSON's "test" split (LPIPS-Alex, PSNR) and best.pt keeps the best.

usage: python train.py --json pairs.json --out DIR --steps 2000 [--minutes 55] [--no_ref] [--fg_mask] [--hflip]
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import lpips

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DifixRe
from losses import DifixLoss
from data import PairDataset


@torch.no_grad()
def validate(net, ds, lp, n=None):
    net.eval(); L, P = [], []
    for k in range(len(ds) if n is None else min(n, len(ds))):
        x, y, _ = ds[k]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = net(x[None].cuda()).float()
        y = y[None].cuda()
        L.append(float(lp(o, y))); P.append(float(-10 * torch.log10(((o - y) / 2) .pow(2).mean())))
    net.train()
    return float(np.mean(L)), float(np.mean(P))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--minutes", type=float, default=55)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--size", default="576x1024"); ap.add_argument("--no_ref", action="store_true")
    ap.add_argument("--fg_mask", action="store_true"); ap.add_argument("--hflip", action="store_true")
    ap.add_argument("--w_gram", type=float, default=0.5); ap.add_argument("--lora_rank", type=int, default=4)
    ap.add_argument("--val_every", type=int, default=500); ap.add_argument("--val_n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--log_every", type=int, default=50)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    H, W = map(int, a.size.split("x"))
    net = DifixRe(lora_rank=a.lora_rank).cuda(); net.set_trainable(); net.unet.enable_gradient_checkpointing(); net.decoder_ckpt = True; net.train()
    params = net.trainable_parameters()
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=1e-2)
    crit = DifixLoss(w_gram=a.w_gram).cuda()
    lp_alex = lpips.LPIPS(net="alex", verbose=False).cuda().eval().requires_grad_(False)
    ds = PairDataset(a.json, "train", (H, W), use_ref=not a.no_ref, use_fg=a.fg_mask, hflip=a.hflip)
    vds = PairDataset(a.json, "test", (H, W), use_ref=not a.no_ref, use_fg=False, hflip=False)
    step, log, best = 0, [], dict(val_lpips=1e9)
    if os.path.exists(f"{a.out}/last.pt"):
        s = torch.load(f"{a.out}/last.pt", map_location="cuda")
        net.load_trained_state_dict(s["model"]); opt.load_state_dict(s["opt"]); step, log, best = s["step"], s["log"], s["best"]
        print(f"resumed at step {step}", flush=True)
    print(f"{len(ds)} train / {len(vds)} val triplets, {sum(p.numel() for p in params)/1e6:.0f}M trainable params, "
          f"{H}x{W}, views={1 if a.no_ref else 2}", flush=True)
    g = torch.Generator().manual_seed(a.seed + step)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=a.workers, drop_last=True, generator=g,
                                     persistent_workers=True)
    t0 = time.time(); run = []; torch.cuda.reset_peak_memory_stats()
    done = False
    while not done:
        for x, y, valid in dl:
            if step >= a.steps or time.time() - t0 > a.minutes * 60:
                done = True; break
            for pg in opt.param_groups:
                pg["lr"] = a.lr * min(1.0, (step + 1) / a.warmup)
            x, y, valid = x.cuda(non_blocking=True), y.cuda(non_blocking=True), valid.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = net(x)
                loss, parts = crit(o.float(), y, valid if a.fg_mask else None)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); step += 1
            run.append([parts["l2"], parts["lpips"], parts["gram"]])
            if step % a.log_every == 0:
                r = np.mean(run, 0); run = []; log.append(dict(step=step, l2=float(r[0]), lpips=float(r[1]), gram=float(r[2])))
                print(f"step {step} l2 {r[0]:.4f} lpips {r[1]:.4f} gram {r[2]:.5f} {time.time()-t0:.0f}s "
                      f"peak {torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
            if step % a.val_every == 0 or step == a.steps:
                vl, vp = validate(net, vds, lp_alex, a.val_n); log.append(dict(step=step, val_lpips=vl, val_psnr=vp)); tag = ""
                if vl < best["val_lpips"]:
                    best = dict(val_lpips=vl, val_psnr=vp, step=step); tag = " *best*"
                    torch.save({k: v.to(torch.bfloat16) for k, v in net.trained_state_dict().items()}, f"{a.out}/best.pt")
                print(f"VAL step {step}: lpips {vl:.4f} psnr {vp:.2f}{tag}", flush=True)
    torch.save(dict(model=net.trained_state_dict(), opt=opt.state_dict(), step=step, log=log, best=best, args=vars(a)), f"{a.out}/last.pt")
    json.dump(dict(log=log, best=best, args=vars(a), peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30),
              open(f"{a.out}/train_log.json", "w"), indent=1)
    print(f"saved at step {step}{' (DONE)' if step >= a.steps else ''}; best {best}", flush=True)


if __name__ == "__main__":
    main()
