"""The headline numbers quoted in the talk and on the webpage, read from the result files in one place, so the
documents cannot drift from the tables. CPU only; run after final_report.py, midterm_checks.py, heading.py summary,
summarize_closed_loop.py and progressive/faithfusion/summary.py.
-> results/final/key_numbers.json
usage: scripts/lib/py.sh src/eval/key_numbers.py
"""
import json
import numpy as np
import pandas as pd

F = "results/final"
FIXERS = ["difix", "difix_ref", "nv_fixer", "difix_ft", "difix_reimpl_ref", "ours_hfgan_s0"]


def r(x, n=3):
    return None if x is None or pd.isna(x) else round(float(x), n)


def main():
    K = {}
    P = pd.read_csv(f"{F}/pooled_all_metrics.csv")
    g = lambda m, b, k: P[(P.method == m) & (P.bin == b)][k].iloc[0]
    K["roads"] = int(P.n_roads.max()); K["images"] = {b: int(P[P.bin == b].n_img.max()) for b in ["0m", "~3m", "~6m"]}
    K["median_offset_m"] = {b: r(P[P.bin == b].med_offset_m.median(), 1) for b in ["~3m", "~6m"]}
    K["lane_f1"] = {m: {b: r(g(m, b, "lane_f1")) for b in ["0m", "~3m", "~6m"]} for m in P.method.unique()}
    K["lane_f1_ci_3m"] = {m: [r(g(m, "~3m", "lane_f1_lo")), r(g(m, "~3m", "lane_f1_hi"))] for m in P.method.unique()}
    K["on_path_lane_f1_range"] = [r(P[P.bin == "0m"].lane_f1.min(), 2), r(P[P.bin == "0m"].lane_f1.max(), 2)]
    fx = [m for m in FIXERS if m in set(P.method)]
    K["best_fixer"] = {b: max(fx, key=lambda m: g(m, b, "lane_f1")) for b in ["~3m", "~6m"]}
    K["lpips_3m"] = {m: r(g(m, "~3m", "lpips")) for m in P.method.unique()}
    K["spurious_3m"] = {m: r(g(m, "~3m", "yolo_halluc_veh")) for m in P.method.unique()}
    K["spurious_on_path_raw"] = r(g("raw", "0m", "yolo_halluc_veh"))
    K["paint_3m"] = {m: [r(g(m, "~3m", "mark_prec"), 2), r(g(m, "~3m", "mark_rec"), 2)] for m in P.method.unique()}

    D = pd.read_csv("results/eval_more/deltas_across_scenes.csv")
    d = lambda m, b, k: D[(D.method == m) & (D.bin == b)].iloc[0][[k, k + "_lo", k + "_hi", k + "_npos"]].tolist()
    K["spurious_reimpl_vs_raw_over_roads"] = [r(x) for x in d("difix_reimpl_ref", "~3m", "yolo_halluc_veh")]
    K["spurious_ft_vs_raw_over_roads"] = [r(x) for x in d("difix_ft", "~3m", "yolo_halluc_veh")]
    K["spurious_ref_vs_raw_over_roads"] = [r(x) for x in d("difix_ref", "~3m", "yolo_halluc_veh")]
    K["progressive_lane_f1_gain_over_roads_3m"] = [r(x, 2) for x in d("prog_difix_reimpl", "~3m", "lane_f1")]

    C = pd.read_csv(f"{F}/midterm_checks.csv")
    vs = C[C.check == "vs_nv_fixer"]
    K["vs_nv_fixer"] = {f"{x.method}@{x.bin}": [r(x.value, 2), r(x.lo, 2), r(x.hi, 2), x.note] for x in vs.itertuples()}
    pdrop = C[(C.check == "paint_drop")]
    K["paint_drop_share"] = {f"{x.method}@{x.bin}": r(x.value, 4) for x in pdrop.itertuples()}
    nt = C[C.check == "lane_f1_no_traffic"]
    K["traffic_free_max_rise"] = {b: r(max(float(x.value) - float(x.note.split()[2]) for x in nt[nt.bin == b].itertuples()), 3)
                                 for b in ["~3m", "~6m"]}
    K["traffic_free_share"] = r(nt[nt.bin == "~3m"].n.iloc[0] / K["images"]["~3m"], 2)

    H = pd.read_csv("results/heading/summary.csv")

    def hv(axis, level, method, k):
        h = H[(H.axis == axis) & (H.method == method) & np.isclose(H.level.abs(), abs(level))]
        return float(np.average(h[k], weights=h.n)) if len(h) else np.nan
    K["heading_roads"] = sorted(set(",".join(H.scenes).split(",")), key=lambda s: int(s[6:]))
    K["heading_raw_lane_f1"] = {"0": r(K["lane_f1"]["raw"]["0m"], 2), "15": r(hv("yaw", 15, "raw", "lane_f1"), 2)}
    hgt = {}
    for dz in (1.0, -1.0):
        h = H[(H.axis == "height") & np.isclose(H.level, dz)].set_index("method")
        best = max((m for m in FIXERS if m in h.index), key=lambda m: h.loc[m, "lane_f1"])
        hgt[f"{dz:+g}"] = dict(raw_d_road_psnr=r(h.loc["raw", "d_road_psnr"], 1), raw_lane_f1=r(h.loc["raw", "lane_f1"], 2),
                               best_fixer=best, best_gain=r(h.loc[best, "lane_f1"] - h.loc["raw", "lane_f1"], 2))
    K["height"] = hgt

    T = pd.read_csv(f"{F}/table4_closed_loop.csv")
    T = T[T.seeds.astype(str) == T.groupby("fixer").seeds.transform(lambda s: max(s.astype(str), key=len))]
    K["closed_loop"] = {x.fixer: dict(runs=int(x.runs), hdscore=r(x.hdscore), d=r(x.d_hd), lo=r(x.d_lo), hi=r(x.d_hi))
                        for x in T.itertuples()}

    try:
        E = pd.read_csv("results/faithfusion/eig_vs_flat_roads.csv")
        o = E[E.scene == "over roads"]
        K["faithfusion_eig_minus_flat"] = {f"{x.metric}@{x.bin}": [r(x.delta, 4), r(x.lo, 4), r(x.hi, 4), int(x.n)] for x in o.itertuples()}
    except FileNotFoundError:
        pass
    json.dump(K, open(f"{F}/key_numbers.json", "w"), indent=1)
    print(json.dumps({k: K[k] for k in ("lane_f1_ci_3m", "best_fixer", "height", "closed_loop")}, indent=1)[:3000])


if __name__ == "__main__":
    main()
