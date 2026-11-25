"""Train the compact U-Net fixer from scratch on our pairs, with validation-based checkpoint selection.

Training pairs (train scenes): cross-lane renders (full + under-fit splats), cycle-reconstruction renders, and
on-path identity renders (so the fixer learns to leave good renders alone). Movable objects are masked out of the loss.
Loss = L1 + w_lpips * LPIPS-VGG [+ w_hf * high-frequency loss] [+ w_gan * PatchGAN hinge loss].
  high-frequency loss: L1 between Sobel gradients of output and target (fights the blur of L1/LPIPS regression)
Cosine LR with warm-up; time-boxed chunks (--minutes) with resume; every --val_every steps the model is scored on the
VAL scene(s) (full-frame, LPIPS-Alex vs real) and best.pt keeps the best.

usage: scripts/lib/py.sh src/fixers/unet/train.py --pairs <root> --scenes a,b --val_scenes c --out <dir> --total_steps 6000 [--w_hf 0.5] [--w_gan 0.05] [--seed 0]
"""
import argparse, json, os, sys, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import lpips

from fixers.unet.model import UNetFixer

VARIANT_WEIGHT = {"full": 1.0, "underfit": 0.5, "cycle": 1.0, "identity": 0.6}


class Pairs(torch.utils.data.Dataset):
    def __init__(self, root, scenes, crop=512, variants=("full", "underfit", "cycle", "identity")):
        self.root, self.crop = root, crop
        self.items = [m for s in scenes for m in json.load(open(f"{root}/{s}/meta.json"))
                      if m["variant"] in variants and not m["onpath_test"]]
        # per-variant weights normalised by count, so each variant contributes in proportion to VARIANT_WEIGHT
        from collections import Counter
        cnt = Counter(m["variant"] for m in self.items)
        self.weights = [VARIANT_WEIGHT[m["variant"]] / cnt[m["variant"]] for m in self.items]
        self.counts = dict(cnt)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        m = self.items[k]
        ld = lambda p: np.array(Image.open(f"{self.root}/{p}"))
        x, y, fg = ld(m["path"]), ld(m["real"]), ld(m["fg"])
        H, W = y.shape[:2]; c = self.crop
        i, j = random.randint(0, H - c), random.randint(0, W - c)
        sl = (slice(i, i + c), slice(j, j + c))
        x, y, fg = x[sl], y[sl], fg[sl]
        if random.random() < 0.5:  # horizontal flip
            x, y, fg = x[:, ::-1], y[:, ::-1], fg[:, ::-1]
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255
        return t(x), t(y), torch.from_numpy(np.ascontiguousarray(fg) > 127)[None].float()


def val_items(root, scenes, every=3):
    it = [m for s in scenes for m in json.load(open(f"{root}/{s}/meta.json")) if m["variant"] == "full"]
    return [m for k, m in enumerate(it) if m["onpath_test"] or k % every == 0]


class PatchD(nn.Module):
    """Small spectral-norm PatchGAN discriminator."""
    def __init__(self, ch=64):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.net = nn.Sequential(sn(nn.Conv2d(3, ch, 4, 2, 1)), nn.LeakyReLU(0.2),
                                 sn(nn.Conv2d(ch, ch * 2, 4, 2, 1)), nn.LeakyReLU(0.2),
                                 sn(nn.Conv2d(ch * 2, ch * 4, 4, 2, 1)), nn.LeakyReLU(0.2),
                                 sn(nn.Conv2d(ch * 4, ch * 4, 3, 1, 1)), nn.LeakyReLU(0.2),
                                 sn(nn.Conv2d(ch * 4, 1, 3, 1, 1)))

    def forward(self, x):
        return self.net(x * 2 - 1)


def sobel(x):
    k = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device) / 4
    kx = k.view(1, 1, 3, 3).repeat(3, 1, 1, 1); ky = k.t().contiguous().view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    return torch.cat([F.conv2d(x, kx, padding=1, groups=3), F.conv2d(x, ky, padding=1, groups=3)], 1)


@torch.no_grad()
def validate(net, root, items, lp_alex):
    net.eval(); ls, ps = [], []
    for m in items:
        x = torch.from_numpy(np.array(Image.open(f"{root}/{m['path']}"))).permute(2, 0, 1)[None].float().cuda() / 255
        y = torch.from_numpy(np.array(Image.open(f"{root}/{m['real']}"))).permute(2, 0, 1)[None].float().cuda() / 255
        fg = torch.from_numpy(np.array(Image.open(f"{root}/{m['fg']}")) > 127)[None, None].float().cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = net(x)
        o = o.float() * (1 - fg) + y * fg
        ls.append(float(lp_alex(o * 2 - 1, y * 2 - 1)))
        ps.append(float(-10 * torch.log10(((o - y) ** 2).mean())))
    net.train()
    return float(np.mean(ls)), float(np.mean(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True); ap.add_argument("--scenes", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--val_scenes", default=""); ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--minutes", type=float, default=55); ap.add_argument("--total_steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=6); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--w_lpips", type=float, default=0.2); ap.add_argument("--w_hf", type=float, default=0.0)
    ap.add_argument("--w_gan", type=float, default=0.0); ap.add_argument("--gan_start", type=int, default=1000)
    ap.add_argument("--base", type=int, default=48); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    ds = Pairs(a.pairs, a.scenes.split(","))
    sampler = torch.utils.data.WeightedRandomSampler(ds.weights, num_samples=10 ** 7, replacement=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, sampler=sampler, num_workers=a.workers, drop_last=True,
                                     persistent_workers=True, prefetch_factor=4)
    vitems = val_items(a.pairs, a.val_scenes.split(",")) if a.val_scenes else []
    net = UNetFixer(base=a.base).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    warm = 200
    lr_at = lambda s: a.lr * (s / warm if s < warm else 0.5 * (1 + np.cos(np.pi * min(1.0, (s - warm) / max(1, a.total_steps - warm)))))
    D = PatchD().cuda() if a.w_gan > 0 else None
    optD = torch.optim.AdamW(D.parameters(), lr=1e-4, betas=(0.5, 0.999)) if D is not None else None
    lp = lpips.LPIPS(net="vgg", verbose=False).cuda().eval().requires_grad_(False)
    lp_alex = lpips.LPIPS(net="alex", verbose=False).cuda().eval().requires_grad_(False)
    step, log, best = 0, [], dict(val_lpips=1e9)
    ck = f"{a.out}/ckpt_last.pt"
    if os.path.exists(ck):
        s = torch.load(ck, map_location="cuda")
        net.load_state_dict(s["net"]); opt.load_state_dict(s["opt"]); step, log, best = s["step"], s["log"], s["best"]
        if D is not None and s.get("D"):
            D.load_state_dict(s["D"]); optD.load_state_dict(s["optD"])
        print(f"resumed at step {step}", flush=True)
    print(f"{len(ds)} pairs {ds.counts}, {len(vitems)} val images, {sum(p.numel() for p in net.parameters())/1e6:.1f}M params, "
          f"w_hf={a.w_hf} w_gan={a.w_gan} seed={a.seed}", flush=True)
    t0 = time.time(); run = []
    for x, y, fg in dl:
        if step >= a.total_steps or time.time() - t0 > a.minutes * 60:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y, fg = x.cuda(non_blocking=True), y.cuda(non_blocking=True), fg.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = net(x)
        o = o.float(); k = 1 - fg
        o_p = o * k + y * fg  # no loss on movable objects (they differ between passes)
        l1 = ((o - y).abs() * k).sum() / (k.sum() * 3 + 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            llp = lp(o_p * 2 - 1, y * 2 - 1).mean().float()
        loss = l1 + a.w_lpips * llp
        lhf = torch.zeros((), device="cuda"); lg = torch.zeros((), device="cuda")
        if a.w_hf > 0:
            lhf = ((sobel(o_p) - sobel(y)).abs()).mean()
            loss = loss + a.w_hf * lhf
        use_gan = D is not None and step >= a.gan_start
        if use_gan:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = -D(o_p).float().mean()
            loss = loss + a.w_gan * lg
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if D is not None:  # hinge loss for the discriminator (always trained, used by G after gan_start)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ld_ = F.relu(1 - D(y)).mean() + F.relu(1 + D(o_p.detach())).mean()
            optD.zero_grad(set_to_none=True); ld_.float().backward(); optD.step()
        step += 1
        run.append([l1.item(), llp.item(), lhf.item(), lg.item()])
        if step % 100 == 0:
            r = np.mean(run, 0); run = []
            log.append(dict(step=step, l1=float(r[0]), lpips=float(r[1]), hf=float(r[2]), gan=float(r[3]), lr=lr_at(step)))
            print(f"step {step} l1 {r[0]:.4f} lpips {r[1]:.4f} hf {r[2]:.4f} gan {r[3]:.3f} lr {lr_at(step):.2e} {time.time()-t0:.0f}s", flush=True)
        if vitems and (step % a.val_every == 0 or step == a.total_steps):
            vl, vp = validate(net, a.pairs, vitems, lp_alex)
            log.append(dict(step=step, val_lpips=vl, val_psnr=vp))
            tag = ""
            if vl < best["val_lpips"]:
                best = dict(val_lpips=vl, val_psnr=vp, step=step)
                torch.save(dict(net=net.state_dict(), base=a.base, step=step, val=best, args=vars(a)), f"{a.out}/best.pt"); tag = " *best*"
            print(f"VAL step {step}: lpips {vl:.4f} psnr {vp:.2f}{tag}", flush=True)
    torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), step=step, log=log, best=best, args=vars(a),
                    D=D.state_dict() if D is not None else None, optD=optD.state_dict() if D is not None else None), ck)
    torch.save(dict(net=net.state_dict(), base=a.base, step=step), f"{a.out}/last.pt")
    json.dump(dict(log=log, best=best, args=vars(a), counts=ds.counts), open(f"{a.out}/train_log.json", "w"))
    print(f"saved at step {step}{' (DONE)' if step >= a.total_steps else ''}; best {best}", flush=True)


if __name__ == "__main__":
    main()
