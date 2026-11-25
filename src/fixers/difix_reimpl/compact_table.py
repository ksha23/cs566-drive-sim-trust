"""Compact test table (mean [95% cluster-bootstrap CI]) for the Difix variants vs. the other fixers, from summarize.py's CSV.
usage: python compact_table.py results/difix_reimpl/summary_test.csv results/difix_reimpl/summary_test_compact.md"""
import csv, sys
COLS = ["psnr", "ssim", "lpips", "road_lpips", "mark_prec", "mark_rec", "mark_f1", "ntl_iou", "lane_f1", "yolo_halluc_veh"]
KEEP = ["raw", "difix", "difix_ref", "difix_ft", "difix_reimpl_ref", "difix_reimpl448", "difix_official448", "ours_base_s0"]
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["method"] in KEEP]
rows.sort(key=lambda r: (r["bin"], KEEP.index(r["method"])))
cell = lambda r, c: ("%.3f [%.3f, %.3f]" % (float(r[c]), float(r[c + "_clo"]), float(r[c + "_chi"]))) if r.get(c) and r.get(c + "_clo") else ("%.3f" % float(r[c]) if r.get(c) else "-")
with open(sys.argv[2], "w") as f:
    f.write("| offset | method | n | " + " | ".join(COLS) + " |\n|" + "---|" * (3 + len(COLS)) + "\n")
    for r in rows:
        f.write("| %s | %s | %s | " % (r["bin"], r["method"], r["n"]) + " | ".join(cell(r, c) for c in COLS) + " |\n")
for r in rows:
    print(r["bin"], r["method"], r["n"], " ".join("%.3f" % float(r[c]) if r.get(c) else "-" for c in COLS))
