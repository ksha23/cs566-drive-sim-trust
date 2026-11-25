"""HUGSIM reproduction check: our LTF closed-loop runs (no fixer) vs. the HUGSIM paper's Table 13 (LTF, nuScenes),
per difficulty level. Writes results/hugsim/reproduction_ltf.{json,md}.
Paper numbers are typed in from Table 13 of arXiv:2412.01718 (HUGSIM, TPAMI 2026 version)."""
import json
import numpy as np

PAPER_LTF_NUSC = {  # difficulty -> sub-scores (Table 13, "Nuscenes" rows, LTF columns)
    "easy":    dict(nc=0.835, dac=0.858, ttc=0.781, c=0.994, rc=0.847, hdscore=0.616),
    "medium":  dict(nc=0.587, dac=0.872, ttc=0.527, c=0.995, rc=0.542, hdscore=0.307),
    "hard":    dict(nc=0.513, dac=0.872, ttc=0.453, c=1.000, rc=0.498, hdscore=0.226),
    "extreme": dict(nc=0.401, dac=0.928, ttc=0.353, c=0.993, rc=0.357, hdscore=0.171),
}
K = ["nc", "dac", "ttc", "c", "rc", "hdscore"]


def main():
    runs = [r for r in json.load(open("results/hugsim/closed_loop_runs.json")) if r["fixer"] == "none"]
    rng = np.random.default_rng(0); out = {}
    for d in PAPER_LTF_NUSC:
        R = [r for r in runs if r["difficulty"] == d]
        if not R:
            continue
        ours = {k: float(np.mean([r[k] for r in R])) for k in K}
        hd = np.array([r["hdscore"] for r in R]); bs = hd[rng.integers(0, len(hd), (2000, len(hd)))].mean(1)
        out[d] = dict(n_runs=len(R), scenarios=sorted({r["scen"] for r in R}), ours=ours, paper=PAPER_LTF_NUSC[d],
                      hd_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    json.dump(out, open("results/hugsim/reproduction_ltf.json", "w"), indent=1)
    with open("results/hugsim/reproduction_ltf.md", "w") as f:
        f.write("| difficulty | runs (scenarios x seeds) | HD-Score ours [95% CI] | HD-Score paper | NC ours/paper | DAC | TTC | COM | Rc |\n"
                "|---|---|---|---|---|---|---|---|---|\n")
        for d, v in out.items():
            o, p = v["ours"], v["paper"]
            f.write(f"| {d} | {v['n_runs']} ({len(v['scenarios'])} x seeds) | {o['hdscore']:.3f} [{v['hd_ci'][0]:.3f}, {v['hd_ci'][1]:.3f}] | "
                    f"{p['hdscore']:.3f} | " + " | ".join(f"{o[k]:.3f} / {p[k]:.3f}" for k in ["nc", "dac", "ttc", "c", "rc"]) + " |\n")
    print(open("results/hugsim/reproduction_ltf.md").read())


if __name__ == "__main__":
    main()
