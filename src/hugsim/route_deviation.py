"""How far do the HUGSIM closed-loop runs drive from the recorded path? -> results/hugsim/route_deviation.{csv,json}

For every run (data.pkl of closed_loop.py) the ego position at each policy step is compared with the recorded
camera trajectory of the scene (ground_param.pkl, the path HUGSIM itself uses for route completion and for its
10 m "Far from preset trajectory" termination). Distance = point-to-polyline distance in the ground plane, i.e.
the lateral offset of the cameras the policy sees, in the same sense as the Para-Lane offsets of section 5.
ego_box is (z, -x, -y, ...) of the simulator's (x, z) ground coordinates (hug_sim.py: HUGSimEnv.ego_box).
usage: python route_deviation.py [--root $CS566D_DATA/hugsim/closed_loop] [--scenes $CS566D_DATA/hugsim/scenes]
                                 [--out results/hugsim]
"""
import argparse, glob, json, os, pickle
import numpy as np
from common.config import HUGSIM_DATA

BINS = [1.0, 2.0, 3.0, 4.5]  # 4.5 m separates the ~3 m and ~6 m bins of the Para-Lane evaluation


def polyline_dist(p, line):
    a, b = line[:-1], line[1:]
    ab = b - a; L2 = np.maximum((ab ** 2).sum(-1), 1e-12)
    t = np.clip(((p[:, None] - a[None]) * ab[None]).sum(-1) / L2[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.sqrt(((p[:, None] - proj) ** 2).sum(-1)).min(1)


def boot(x, n=2000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    if len(x) < 2:
        return np.nan, np.nan
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{HUGSIM_DATA}/closed_loop")
    ap.add_argument("--scenes", default=f"{HUGSIM_DATA}/scenes")
    ap.add_argument("--out", default="results/hugsim")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    paths, runs = {}, []
    for p in sorted(glob.glob(f"{a.root}/*/seed*/*/data.pkl")):
        fixer, seed, scen = p.split("/")[-4:-1]
        scene = "-".join(scen.split("-")[:2])
        if scene not in paths:
            gp = glob.glob(f"{a.scenes}/*/{scene}/ground_param.pkl")[0]
            cam_poses = pickle.load(open(gp, "rb"))[0]
            paths[scene] = np.stack([cam_poses[:, 0, 3], cam_poses[:, 2, 3]], -1)
        frames = pickle.load(open(p, "rb"))[0]["frames"]
        ego = np.array([[-f["ego_box"][1], f["ego_box"][0]] for f in frames], float)
        d = polyline_dist(ego, paths[scene]) if len(ego) else np.zeros(0)
        runs.append(dict(fixer=fixer, seed=int(seed[4:]), scen=scen, steps=len(d),
                         dist_m=[round(float(x), 3) for x in d],
                         max_m=float(d.max()) if len(d) else 0.0,
                         median_m=float(np.median(d)) if len(d) else 0.0,
                         **{f"frac_gt_{b:g}m": float((d > b).mean()) if len(d) else 0.0 for b in BINS}))
    json.dump(runs, open(f"{a.out}/route_deviation_runs.json", "w"), indent=0)

    ends = {(r["fixer"], r["seed"], r["scen"]): r for r in
            json.load(open(f"{a.out}/closed_loop_runs.json"))} if os.path.exists(f"{a.out}/closed_loop_runs.json") else {}
    fixers = sorted({r["fixer"] for r in runs}, key=lambda f: (f != "none", f))
    lines = ["per policy step (pooled over runs): fixer,n_runs,n_steps,median_m,p90_m,p99_m,max_m,"
             + ",".join(f"frac_steps_gt_{b:g}m" for b in BINS)]
    for f in fixers:
        R = [r for r in runs if r["fixer"] == f]; d = np.concatenate([r["dist_m"] for r in R])
        lines.append(f"{f},{len(R)},{len(d)},{np.median(d):.2f},{np.percentile(d, 90):.2f},{np.percentile(d, 99):.2f},"
                     f"{d.max():.2f}," + ",".join(f"{(d > b).mean():.4f}" for b in BINS))
    lines += ["", "per run: fixer,n_runs,max_m_median,max_m_ci_lo,max_m_ci_hi,"
              + ",".join(f"frac_runs_max_gt_{b:g}m" for b in BINS)]
    for f in fixers:
        m = [r["max_m"] for r in runs if r["fixer"] == f]; lo, hi = boot(m)
        lines.append(f"{f},{len(m)},{np.median(m):.2f},{lo:.2f},{hi:.2f},"
                     + ",".join(f"{np.mean(np.array(m) > b):.3f}" for b in BINS))
    if ends:
        lines += ["", "max deviation by how the run ended: fixer,end,n_runs,max_m_median,max_m_max"]
        for f in fixers:
            for e in sorted({ends[k]["end"] for k in ends if k[0] == f}):
                m = [r["max_m"] for r in runs if r["fixer"] == f and (f, r["seed"], r["scen"]) in ends
                     and ends[(f, r["seed"], r["scen"])]["end"] == e]
                if m:
                    lines.append(f"{f},{e},{len(m)},{np.median(m):.2f},{np.max(m):.2f}")
        # Does a fixer change HD-Score more in the runs that leave the path? Split by the no-fixer run's maximum.
        base = {(r["scen"], r["seed"]): r for r in runs if r["fixer"] == "none"}
        lines += ["", "paired delta HD-Score vs none, split by the no-fixer run's max deviation: "
                  "fixer,split,n_pairs,delta_mean,ci_lo,ci_hi"]
        for f in fixers:
            if f == "none":
                continue
            for name, keep in [("max<=3m", lambda x: x <= 3.0), ("max>3m", lambda x: x > 3.0)]:
                dd = [ends[(f, s, c)]["hdscore"] - ends[("none", s, c)]["hdscore"]
                      for (c, s), b in base.items() if keep(b["max_m"])
                      and (f, s, c) in ends and ("none", s, c) in ends]
                if dd:
                    lo, hi = boot(dd)
                    lines.append(f"{f},{name},{len(dd)},{np.mean(dd):.4f},{lo:.4f},{hi:.4f}")
    open(f"{a.out}/route_deviation.csv", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
