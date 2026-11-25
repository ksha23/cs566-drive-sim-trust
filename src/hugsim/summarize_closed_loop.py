"""Aggregate HUGSIM closed-loop results (eval.json per run) into results/hugsim/closed_loop.{json,csv}.

HD-Score = RC * mean-over-time(NC * DAC * (5 TTC + 2 C) / 7) as computed by HUGSIM's score_calculator (we report
its hdscore, rc, and the PDMS components nc/dac/ttc/c). Variance: std over runs, 95% bootstrap CI over runs, and a
paired comparison vs. no fixer over the (scenario, seed) runs present for both.
usage: python summarize_closed_loop.py [--root $CS566D_DATA/hugsim/closed_loop] [--out results/hugsim]
"""
import argparse, glob, json, os, re
import numpy as np
from common.config import HUGSIM_DATA

KEYS = ["hdscore", "rc", "pdms", "nc", "dac", "ttc", "c"]


def boot(x, n=2000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    if len(x) < 2:
        return np.nan, np.nan
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{HUGSIM_DATA}/closed_loop")
    ap.add_argument("--out", default="results/hugsim")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    runs = []
    for p in glob.glob(f"{a.root}/*/seed*/*/eval.json"):
        fixer, seed, scen = p.split("/")[-4:-1]
        r = json.load(open(p)); log = open(os.path.join(os.path.dirname(p), "sim.log"), errors="ignore").read()
        m = re.findall(r"Collision with background|Collision with foreground|Far from preset trajectory|Complete", log)
        r.update(fixer=fixer, seed=int(seed[4:]), scen=scen, end=(m[-1] if m else "max steps / other"),
                 difficulty=scen.split("-")[2])
        runs.append(r)
    json.dump(runs, open(f"{a.out}/closed_loop_runs.json", "w"), indent=0, default=float)
    lines = ["fixer,n_runs," + ",".join(f"{k}_mean,{k}_std,{k}_ci_lo,{k}_ci_hi" for k in KEYS) + ",collisions,off_route,completed"]
    fixers = sorted({r["fixer"] for r in runs}, key=lambda f: (f != "none", f))
    for f in fixers:
        R = [r for r in runs if r["fixer"] == f]; row = [f, str(len(R))]
        for k in KEYS:
            x = [r[k] for r in R]; lo, hi = boot(x)
            row += [f"{np.mean(x):.4f}", f"{np.std(x):.4f}", f"{lo:.4f}", f"{hi:.4f}"]
        row += [str(sum("Collision" in r["end"] for r in R)), str(sum("Far" in r["end"] for r in R)),
                str(sum(r["end"] == "Complete" for r in R))]
        lines.append(",".join(row))
    lines += ["", "paired vs none: fixer,n_pairs,delta_hdscore_mean,ci_lo,ci_hi,delta_rc_mean"]
    base = {(r["scen"], r["seed"]): r for r in runs if r["fixer"] == "none"}
    for f in fixers:
        if f == "none":
            continue
        pr = [(r, base[(r["scen"], r["seed"])]) for r in runs if r["fixer"] == f and (r["scen"], r["seed"]) in base]
        if pr:
            d = [x["hdscore"] - y["hdscore"] for x, y in pr]; lo, hi = boot(d)
            lines.append(f"{f},{len(pr)},{np.mean(d):.4f},{lo:.4f},{hi:.4f},{np.mean([x['rc'] - y['rc'] for x, y in pr]):.4f}")
    lines += ["", "per seed hdscore mean: fixer,seed,n,hdscore"]
    for f in fixers:
        for s in sorted({r["seed"] for r in runs}):
            R = [r["hdscore"] for r in runs if r["fixer"] == f and r["seed"] == s]
            if R:
                lines.append(f"{f},{s},{len(R)},{np.mean(R):.4f}")
    open(f"{a.out}/closed_loop_summary.csv", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
