"""Build the 5-minute CS 566 talk as a self-contained PPTX (images and the video clip embedded). Run from this folder.
Numbers come from results/final/key_numbers.json (src/eval/key_numbers.py), the two talk figures are drawn here from
results/final/*.csv, and the clip is the first seconds of results/videos/paralane_straight_3m_scene16.mp4 with larger
panel labels (img/talk_clip_3m.mp4, made by make_talk_clip.py).
Speaker notes are about 600 words, about 4.5 minutes at 140 words per minute."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image

W, H = 12192000, 6858000                      # 16:9, 13.333 x 7.5 in
IN = 914400
BG, BG2 = RGBColor(0xFA, 0xF9, 0xF6), RGBColor(0x14, 0x21, 0x3D)
INK, SOFT = RGBColor(0x14, 0x21, 0x3D), RGBColor(0x4A, 0x55, 0x68)
ACC, ACC2 = RGBColor(0xD9, 0x6C, 0x1E), RGBColor(0x2B, 0x6C, 0xB0)
LIGHT, MUTED = RGBColor(0xE8, 0xEC, 0xF2), RGBColor(0xBF, 0xC8, 0xD6)
FONT = "Calibri"
URL = "ksha23.github.io/cs566-drive-sim"
CODE_URL = "github.com/ksha23/cs566-drive-sim-trust"
PARALANE = "Images: Para-Lane (Ni et al., 3DV 2025), CC BY-NC 4.0"

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "final")
K = json.load(open(f"{RES}/key_numbers.json"))   # written by src/eval/key_numbers.py; every number below comes from it
NAME = dict(raw="Raw render", difix="Released Difix", difix_ref="Released Difix-ref", nv_fixer="NVIDIA Fixer",
            difix_ft="Difix-FT", difix_reimpl_ref="Our Difix re-impl.", difix_reimpl="Our Difix re-impl.",
            none="No fixer")
COL = dict(raw="#2a78d6", difix="#eb6834", nv_fixer="#1baf7a", difix_ft="#eda100", difix_reimpl_ref="#e87ba4")  # as final_report.py
FINK, FINK2, FGRID = "#1f1f1e", "#5c5b55", "#e4e3dc"


# --- talk figures (larger type and fewer methods than the report figures) ----------------------------------------
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(FINK2)
    ax.tick_params(colors=FINK2, labelsize=15)
    ax.yaxis.grid(True, color=FGRID, lw=1); ax.set_axisbelow(True)


def fig_offset(path):
    """Lane F1 against lateral offset, five methods, labeled at the line ends."""
    P = pd.read_csv(f"{RES}/pooled_all_metrics.csv")
    xs = {"0m": 0.0, "~3m": K["median_offset_m"]["~3m"], "~6m": K["median_offset_m"]["~6m"]}
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=200); fig.patch.set_facecolor("#FAF9F6"); ax.set_facecolor("#FAF9F6")
    ends = []
    for j, m in enumerate(["raw", "difix", "nv_fixer", "difix_reimpl_ref", "difix_ft"]):
        d = P[P.method == m].set_index("bin")
        y = [d.loc[b, "lane_f1"] for b in xs]
        lo = [d.loc[b, "lane_f1"] - d.loc[b, "lane_f1_lo"] for b in xs]; hi = [d.loc[b, "lane_f1_hi"] - d.loc[b, "lane_f1"] for b in xs]
        xx = [x + (j - 2) * 0.09 for x in xs.values()]   # dodge so that overlapping lines and intervals stay visible
        ax.errorbar(xx, y, yerr=[lo, hi], color=COL[m], lw=2.5, marker="o", ms=8, mec="#FAF9F6", mew=2, capsize=3, zorder=3)
        ends.append([y[-1], m])
    ends.sort()
    ends[0][0] = max(ends[0][0], 0.06)                              # off the x axis
    for i in range(1, len(ends)):                                   # keep end labels at least 0.075 apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.075)
    for y, m in ends:
        ax.text(xs["~6m"] + 0.25, y, NAME[m], color=FINK, fontsize=14, va="center")
    ax.set_xticks(list(xs.values()), ["on the path", f"one lane over\n({xs['~3m']:.1f} m)", f"two lanes over\n({xs['~6m']:.1f} m)"])
    ax.set_xlim(-0.4, xs["~6m"] + 2.9); ax.set_ylim(0, 1.02)
    ax.set_ylabel("Lane F1 vs. real photo (95% CI)", fontsize=15, color=FINK2)
    style(ax); fig.tight_layout(); fig.savefig(path, facecolor=fig.get_facecolor()); plt.close(fig)


def fig_roads(path):
    """Lane F1 one lane over on each road: raw render, NVIDIA Fixer, Difix-FT."""
    F = pd.read_csv(f"{RES}/fig2_lane_f1_per_road_3m.csv")
    ms = ["raw", "nv_fixer", "difix_ft"]
    T = F[F.method.isin(ms)].pivot(index="road", columns="method", values="lane_f1")
    T = T.sort_values("difix_ft", ascending=False)
    lab = {"scene_14": "14\ncurve", "scene_15": "15\nunder\nhighway"}
    fig, ax = plt.subplots(figsize=(7.4, 4.9), dpi=200); fig.patch.set_facecolor("#FAF9F6"); ax.set_facecolor("#FAF9F6")
    w = 0.26
    for j, m in enumerate(ms):
        ax.bar([i + (j - 1) * (w + 0.02) for i in range(len(T))], T[m], width=w, color=COL[m], label=NAME[m], zorder=3)
    ax.set_xticks(range(len(T)), [lab.get(r, r.split("_")[1]) for r in T.index])
    ax.set_ylim(0, 1.0); ax.set_ylabel("Lane F1 one lane over", fontsize=15, color=FINK2)
    ax.set_xlabel("Para-Lane test road", fontsize=15, color=FINK2)
    ax.legend(frameon=False, fontsize=14, ncol=3, loc="upper right", bbox_to_anchor=(1.0, 1.12), labelcolor=FINK)
    style(ax); fig.tight_layout(); fig.savefig(path, facecolor=fig.get_facecolor()); plt.close(fig)
    return F.pivot(index="road", columns="method", values="lane_f1")


def fig_ff(path):
    """AUROC of three per-pixel maps for invented lane paint one lane over, per fixer (results/faithfusion/halluc_auroc.md)."""
    rows = {}
    for l in open(os.path.join(HERE, "..", "results", "faithfusion", "halluc_auroc.md")):
        c = [x.strip() for x in l.strip().strip("|").split("|")]
        if len(c) > 7 and c[0] == "paint" and c[2] == "~3m":
            e = [float(v) for v in c[5].replace("[", "").replace("]", "").replace(",", "").split()]
            rows[c[1]] = dict(eig=e[0], lo=e[1], hi=e[2], rep=float(c[6].split()[0]), cov=float(c[7]), pos=float(c[4]),
                             auprc=float(c[8].split("/")[0]))
    order = ["difix", "difix_ref", "nv_fixer", "difix_ft", "difix_reimpl_ref"]
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=200); fig.patch.set_facecolor("#FAF9F6"); ax.set_facecolor("#FAF9F6")
    ax.axvline(0.5, color=FINK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.505, -0.5, "chance", color=FINK2, fontsize=13, ha="left")
    for i, m in enumerate(order):
        r = rows[m]; y = len(order) - 1 - i
        ax.plot([r["cov"], r["rep"], r["eig"]], [y] * 3, color=FGRID, lw=2, zorder=2)
        ax.plot(r["cov"], y, "o", color="#9a988f", ms=10, mec="#FAF9F6", mew=1.5, zorder=3, label="coverage" if i == 0 else None)
        ax.plot(r["rep"], y, "s", color=FINK2, ms=9, mec="#FAF9F6", mew=1.5, zorder=3, label="reprojection" if i == 0 else None)
        ax.errorbar(r["eig"], y, xerr=[[r["eig"] - r["lo"]], [r["hi"] - r["eig"]]], fmt="D", color="#d96c1e", ms=10, mec="#FAF9F6",
                    mew=1.5, capsize=3, lw=2, zorder=4, label="information gain" if i == 0 else None)
    ax.set_yticks(range(len(order)), [NAME[m] for m in reversed(order)], fontsize=15, color=FINK)
    ax.set_xlim(0.2, 0.85); ax.set_ylim(-0.6, len(order) - 0.1)
    ax.set_xlabel("AUROC for invented lane paint (one lane over)", fontsize=15, color=FINK2)
    ax.legend(frameon=False, fontsize=14, ncol=3, loc="lower center", bbox_to_anchor=(0.42, 1.0), labelcolor=FINK,
              handletextpad=0.3, columnspacing=1.2)
    style(ax); ax.yaxis.grid(False); ax.xaxis.grid(True, color=FGRID, lw=1)
    fig.tight_layout(); fig.savefig(path, facecolor=fig.get_facecolor()); plt.close(fig)
    return rows


def fig_height(path):
    """Lane F1 against camera yaw and against camera height change: raw render and three fixers (fig3_heading_height.csv)."""
    F = pd.read_csv(f"{RES}/fig3_heading_height.csv"); F = F[F.metric == "lane_f1"]
    ms = ["raw", "nv_fixer", "difix_ft", "difix_reimpl_ref"]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 4.6), dpi=200, sharey=True); fig.patch.set_facecolor("#FAF9F6")
    for ax, axis, xl in ((axs[0], "yaw", "camera turned (°)"), (axs[1], "height", "camera raised (m)")):
        ax.set_facecolor("#FAF9F6")
        for j, m in enumerate(ms):
            g = F[(F.axis == axis) & (F.method == m)].sort_values("level")
            dx = (j - 1.5) * (0.5 if axis == "yaw" else 0.035)   # dodge so the 95% intervals stay visible
            ax.errorbar(g.level + dx, g.value, yerr=[g.value - g.lo, g.hi - g.value], color=COL[m], lw=2.2, marker="o", ms=5,
                        mec="#FAF9F6", mew=1.2, capsize=2.5, elinewidth=1.4, label=NAME[m], zorder=3)
        ax.set_xlabel(xl, fontsize=15, color=FINK2); style(ax); ax.set_ylim(0, 1.02)
    axs[0].set_ylabel("Lane F1 vs. real photo (95% CI)", fontsize=15, color=FINK2)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=13, ncol=2, loc="upper center", bbox_to_anchor=(0.55, 1.0), labelcolor=FINK)
    fig.tight_layout(rect=(0, 0, 1, 0.86)); fig.savefig(path, facecolor=fig.get_facecolor()); plt.close(fig)


def hallucination_pair(path):
    """Real image and our re-implementation, cropped from the report's 2 x 3 figure (src/viz/qualitative.py layout)."""
    im = Image.open(f"{RES}/qual_hallucination.jpg")
    a, b = im.crop((0, 78, 640, 442)), im.crop((1280, 480, 1920, 844))
    g = Image.new("RGB", (1296, 364), (250, 249, 246)); g.paste(a, (0, 0)); g.paste(b, (656, 0)); g.save(path, quality=92)


os.makedirs(f"{HERE}/img", exist_ok=True)
hallucination_pair(f"{HERE}/img/talk_hallucination.jpg")
fig_offset(f"{HERE}/img/talk_offset.png")
ROADS = fig_roads(f"{HERE}/img/talk_roads.png")
AUC = fig_ff(f"{HERE}/img/talk_faithfusion.png")
fig_height(f"{HERE}/img/talk_height.png")

# --- slide helpers ------------------------------------------------------------------------------------------------
prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]
TOTAL = 13


def bg(s, color):
    f = s.background.fill
    f.solid()
    f.fore_color.rgb = color


def text(s, x, y, w, h, runs, size=20, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """runs: str, or list of paragraphs; a paragraph is str or list of (text, dict(style)) pieces."""
    tb = s.shapes.add_textbox(Emu(x), Emu(y), Emu(w), Emu(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(size * 0.6)
        pieces = [(para, {})] if isinstance(para, str) else para
        for t, st in pieces:
            r = p.add_run()
            r.text = t
            r.font.name = FONT
            r.font.size = Pt(st.get("size", size))
            r.font.bold = st.get("bold", bold)
            r.font.color.rgb = st.get("color", color)
    return tb


def picture(s, path, x, y, w=None, h=None):
    """Place an image inside the box (x, y, w, h), keeping its aspect ratio, centered."""
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = int(iw * scale), int(ih * scale)
    return s.shapes.add_picture(path, Emu(x + (w - pw) // 2), Emu(y + (h - ph) // 2), Emu(pw), Emu(ph))


def title(s, t, eyebrow):
    text(s, int(0.6 * IN), int(0.35 * IN), int(12 * IN), int(0.4 * IN), eyebrow, size=14, color=ACC, bold=True)
    text(s, int(0.6 * IN), int(0.7 * IN), int(12.1 * IN), int(0.8 * IN), t, size=32, bold=True)


def footer(s, n, dark=False, credit=None):
    c = MUTED if dark else SOFT
    text(s, int(0.6 * IN), int(7.0 * IN), int(9 * IN), int(0.3 * IN), credit or "Kyle Sha, Allen Xia · CS 566, Fall 2026", size=12, color=c)
    text(s, int(11.9 * IN), int(7.0 * IN), int(0.8 * IN), int(0.3 * IN), f"{n} / {TOTAL}", size=12, color=c, align=PP_ALIGN.RIGHT)


def notes(s, t):
    s.notes_slide.notes_text_frame.text = t


def box(s, x, y, w, h, fill, line=None):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(x), Emu(y), Emu(w), Emu(h))
    sh.adjustments[0] = 0.06
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
    sh.shadow.inherit = False
    return sh


L = K["lane_f1"]
V = K["vs_nv_fixer"]["difix_ft@~3m"]
_n, _of = V[3].split()[0].split("/"); ROADS_POS = f"all {_of}" if _n == _of else f"{_n} of {_of}"
mn = lambda t: t.replace("-", "\u2212")   # typographic minus in numbers
SP = K["spurious_reimpl_vs_raw_over_roads"]; SPR = K["spurious_ref_vs_raw_over_roads"]
Hh = K["height"]; CL = K["closed_loop"]
FF = K.get("faithfusion_eig_minus_flat", {}).get("lane_f1@~3m")

# 1. Title ---------------------------------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK); bg(s, BG2)
text(s, int(0.8 * IN), int(1.7 * IN), int(11.5 * IN), int(0.5 * IN), "CS 566 COMPUTER VISION · FINAL PROJECT", size=16, color=ACC, bold=True)
text(s, int(0.8 * IN), int(2.3 * IN), int(11.5 * IN), int(2.0 * IN),
     "How far off the recorded path can a Gaussian-splat driving simulator be trusted?", size=44, color=BG, bold=True)
text(s, int(0.8 * IN), int(4.5 * IN), int(11.5 * IN), int(0.5 * IN), "Kyle Sha and Allen Xia · University of Wisconsin–Madison · Fall 2026",
     size=20, color=MUTED)
footer(s, 1, dark=True, credit="Data: Para-Lane (CC BY-NC 4.0), HUGSIM / nuScenes, EUVS / MARS, MV2, DL3DV")
notes(s, "Driving software is increasingly tested in simulators rebuilt from real drives. We ask how far the simulated car "
         "can leave the recorded path before such a simulator stops being trustworthy, and whether the diffusion models used "
         "to repair off-path images restore that trust or only make the images look better.")

# 2. Problem -------------------------------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK); bg(s, BG)
title(s, "A splat is accurate only where the car drove", "PROBLEM")
y0, hh = int(1.75 * IN), int(3.35 * IN)
picture(s, "img/real_3m.jpg", int(0.6 * IN), y0, int(6.0 * IN), hh)
picture(s, "img/raw_3m.jpg", int(6.75 * IN), y0, int(6.0 * IN), hh)
text(s, int(0.6 * IN), int(5.15 * IN), int(6.0 * IN), int(0.4 * IN), "Real photo, one lane over", size=16, color=SOFT, align=PP_ALIGN.CENTER)
text(s, int(6.75 * IN), int(5.15 * IN), int(6.0 * IN), int(0.4 * IN), "Splat built from the recorded lane, rendered 3 m to the side",
     size=16, color=SOFT, align=PP_ALIGN.CENTER)
text(s, int(0.6 * IN), int(5.8 * IN), int(12.1 * IN), int(1.0 * IN), [
    [("Generative fixers (e.g., NVIDIA Difix) make such renders look realistic. ", {"color": SOFT}),
     ("We measure whether they also make them correct.", {"color": ACC})]], size=21)
footer(s, 2, credit=PARALANE)
notes(s, "The recorded video becomes a 3D Gaussian splat, millions of small colored ellipsoids that can be rendered from "
         "any camera position. Here it is rendered three meters to the side: "
         "the road texture smears and the lane lines break up. Fixers such as NVIDIA's Difix clean up such renders. Looking "
         "better is not the same as being right.")

# 3. Setup ---------------------------------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK); bg(s, BG)
title(s, "Real ground truth from three parallel lanes", "SETUP")
lx, ly, lw = int(0.7 * IN), int(1.9 * IN), int(5.6 * IN)
lanes = [("Lane A: recorded, splat built here", ACC2), ("Lane B (≈3 m): real photos exist", ACC), ("Lane C (≈6 m): real photos exist", ACC)]
for i, (lab, col) in enumerate(lanes):
    yy = ly + i * int(1.25 * IN)
    box(s, lx, yy, lw, int(0.95 * IN), LIGHT)
    arrow = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Emu(lx + int(0.25 * IN)), Emu(yy + int(0.3 * IN)), Emu(int(1.0 * IN)), Emu(int(0.36 * IN)))
    arrow.fill.solid(); arrow.fill.fore_color.rgb = col; arrow.line.fill.background()
    text(s, lx + int(1.45 * IN), yy, lw - int(1.6 * IN), int(0.95 * IN), lab, size=19, anchor=MSO_ANCHOR.MIDDLE)
text(s, int(6.8 * IN), int(1.9 * IN), int(5.9 * IN), int(4.6 * IN), [
    "Para-Lane scans the same roads in three parallel lanes. We build the splat from one lane and render it at the other lanes' recorded poses.",
    "Trust: does a lane detector (CLRNet) find the same lanes in the render as in the real photo? (lane F1)",
    "Hallucination: lane paint or vehicles in the render that the real photo does not have.",
    [(f"{K['roads']} held-out roads · {K['images']['~3m']:,} images at ≈3 m · {K['images']['~6m']:,} at ≈6 m · 95% bootstrap intervals", {"color": SOFT})]],
    size=19)
footer(s, 3)
notes(s, "The Para-Lane dataset makes this measurable: it drives the same roads in three parallel lanes. We build the splat "
         "from one lane and render it at the recorded poses of the other two, so there is a real photo at exactly the "
         "rendered pose. Trust is whether a lane detector finds the same lanes in both. We also count invented lane paint "
         "and vehicles.")

# 4. Methods -------------------------------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK); bg(s, BG)
title(s, "What we compare, and what we built", "METHODS")
cols = [("Image fixers", ["Released Difix, Difix-ref and NVIDIA Fixer (trained on driving data)",
                          "Our re-implementation of Difix training", "Released Difix fine-tuned on our pairs (Difix-FT)",
                          "Small U-Net from scratch"]),
        ("Fix the 3D scene", ["Progressive update: repaired views added back into the splat (Difix3D+), with flat or FaithFusion weights",
                              "Road-surface prior from LiDAR (no generation)"]),
        ("Built and validated", ["Our Difix trainer scores better than the official one at an equal short budget (DL3DV, 3 scenes, single view: LPIPS 0.218 vs. 0.237)",
                                 "Our own splat projection matches gsplat to 90.8 dB (results use gsplat)"])]
cx, cw = int(0.6 * IN), int(3.95 * IN)
for i, (h, items) in enumerate(cols):
    x = cx + i * (cw + int(0.15 * IN))
    box(s, x, int(1.75 * IN), cw, int(4.95 * IN), RGBColor(0xFF, 0xFF, 0xFF), line=RGBColor(0xD8, 0xDD, 0xE5))
    text(s, x + int(0.25 * IN), int(1.95 * IN), cw - int(0.5 * IN), int(0.5 * IN), h, size=22, bold=True, color=ACC if i < 2 else ACC2)
    text(s, x + int(0.25 * IN), int(2.6 * IN), cw - int(0.5 * IN), int(4.0 * IN), ["• " + t for t in items], size=17, color=INK)
footer(s, 4)
notes(s, "We compare NVIDIA's released Difix and its driving-trained Fixer with our re-implementation of the Difix training "
         "procedure, the released Difix fine-tuned on our lane pairs, and a small U-Net. Other methods repair the 3D scene "
         "itself instead of each image. We validated our trainer against NVIDIA's code and our splat projection against gsplat.")

# 5. Trust vs offset -----------------------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK); bg(s, BG)
title(s, "Trust collapses one lane over; trained fixers recover part", "RESULT 1")
picture(s, "img/talk_offset.png", int(0.4 * IN), int(1.6 * IN), int(7.6 * IN), int(5.2 * IN))
c3 = K["lane_f1_ci_3m"]
text(s, int(8.3 * IN), int(1.9 * IN), int(4.4 * IN), int(4.9 * IN), [
    f"Raw render: {L['raw']['0m']:.2f} on the path, {L['raw']['~3m']:.2f} one lane over, {L['raw']['~6m']:.2f} two lanes over.",
    f"Best one lane over: Difix-FT {L['difix_ft']['~3m']:.2f} [{c3['difix_ft'][0]:.2f}, {c3['difix_ft'][1]:.2f}].",
    [(f"Difix-FT is ahead of NVIDIA Fixer on {ROADS_POS} roads (+{V[0]:.2f} [+{V[1]:.2f}, +{V[2]:.2f}] on the same frames).", {"color": ACC})],
    f"Two lanes over, the best fixer reaches {max(L[m]['~6m'] for m in ('difix_ft', 'difix_reimpl_ref')):.2f}."], size=19)
footer(s, 5)
notes(s, f"Lane agreement with the real photo falls from {L['raw']['0m']:.2f} on the path to {L['raw']['~3m']:.2f} one lane "
         f"over and almost zero two lanes over. Released Difix recovers to {L['difix']['~3m']:.2f}, NVIDIA Fixer to "
         f"{L['nv_fixer']['~3m']:.2f}, and the two fixers trained on our lane pairs to {L['difix_reimpl_ref']['~3m']:.2f} and {L['difix_ft']['~3m']:.2f}. On the "
         f"same frames Difix-FT is ahead of NVIDIA Fixer on {ROADS_POS} roads. Two lanes over, even the best fixer reaches "
         f"only about {L['difix_ft']['~6m']:.1f}.")

# 6. Hallucination -------------------------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK); bg(s, BG)
title(s, "Better-looking is not the same as correct", "RESULT 2")
picture(s, "img/talk_hallucination.jpg", int(0.6 * IN), int(1.55 * IN), int(12.1 * IN), int(3.45 * IN))
nsp = int(min(SP[3], SPR[3])); spw = "every road" if nsp == K["roads"] else f"at least {nsp} of {K['roads']} roads"
text(s, int(0.6 * IN), int(5.1 * IN), int(5.9 * IN), int(1.7 * IN), [
    "Left: real photo, one lane over. Right: our re-implementation paints a stripe that is not on the road (orange; "
    "blue box: erased paint): "
    "paint precision 0.13 (raw 0.95), lane F1 0.86. Selected frame; such drops occur on "
    f"{K['paint_drop_share']['difix_reimpl_ref@~3m']:.0%} of its frames."], size=16)
text(s, int(6.8 * IN), int(5.1 * IN), int(5.9 * IN), int(1.7 * IN), [
    f"Both fixers that copy from a reference view add vehicles on {spw} (mean per-road change: ours +{SP[0]:.3f}, "
    f"released Difix-ref +{SPR[0]:.3f} per image); Difix-FT does not.",
    f"Repairing the 3D scene invents the fewest vehicles ({K['spurious_3m']['prog_difix_reimpl']:.2f} per image)."], size=16)
footer(s, 6, credit=PARALANE)
notes(s, "Lane agreement alone hides errors. Orange marks invented lane paint, blue erased paint. Our re-implementation "
         "restores the lanes but here paints a stripe that is not on the road; over all frames, both fixers that copy from a "
         f"reference view add vehicles on {spw}. Repairing the 3D scene invents the fewest vehicles, but helps less two lanes over.")

# 7. Clip -------------------------------------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK); bg(s, BG)
title(s, "One lane over, frame by frame", "EXAMPLE")
cw_, ch_ = int(11.5 * IN), int(11.5 * IN * 482 / 1280)   # the clip's own aspect (1280 x 482)
s.shapes.add_movie("img/talk_clip_3m.mp4", Emu((prs.slide_width - cw_) // 2), Emu(int(1.6 * IN)), Emu(cw_), Emu(ch_),
                   poster_frame_image="img/talk_clip_3m_poster.jpg", mime_type="video/mp4")
text(s, (prs.slide_width - cw_) // 2, int(1.72 * IN) + ch_, cw_, int(0.8 * IN),
     "Video, 10 s (one frame shown when not playing). Marks follow the metrics' rules, frame by frame. Orange: invented paint · blue: erased paint · "
     "red box: vehicle with no counterpart in the real photo.", size=16, color=SOFT)
footer(s, 7, credit=PARALANE)
notes(s, "This clip shows the real photo one lane over, the raw render and four fixers, marked with the same rules as the "
         "metrics. The fixers restore the lane lines to different degrees, and some invent paint or vehicles along the way.")

# 8. Roads -------------------------------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK); bg(s, BG)
title(s, "Results vary strongly by road", "RESULT 3")
picture(s, "img/talk_roads.png", int(0.4 * IN), int(1.55 * IN), int(7.6 * IN), int(5.3 * IN))
FT = ROADS["difix_ft"]; RW = ROADS["raw"]
ORDER_OK = bool(((RW < ROADS["nv_fixer"]) & (ROADS["nv_fixer"] < FT)).all())
print("road order raw < NVIDIA Fixer < Difix-FT on all roads:", ORDER_OK)
hi_road = FT.idxmax().split("_")[1]
lo_road = "under an elevated highway" if FT.idxmin() == "scene_15" else f"on road {FT.idxmin().split('_')[1]}"
text(s, int(8.3 * IN), int(1.9 * IN), int(4.4 * IN), int(4.9 * IN), [
    f"Difix-FT one lane over: {FT.max():.2f} on road {hi_road}, but {FT.min():.2f} {lo_road}.",
    f"The raw render stays between {RW.min():.2f} and {RW.max():.2f} on every road.",
    [(f"On every road the point estimates order raw render < NVIDIA Fixer < Difix-FT.", {"color": ACC})] if ORDER_OK else ""], size=19)
footer(s, 8)
notes(s, f"The level depends strongly on the road. One lane over, Difix-FT reaches {FT.max():.2f} on one boulevard but only "
         f"{FT.min():.2f} {lo_road}, while the raw render stays between {RW.min():.2f} and {RW.max():.2f} everywhere.")

# 9. Heading and height --------------------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK); bg(s, BG)
title(s, "Camera heading barely matters; height does", "RESULT 4")
picture(s, "img/talk_height.png", int(0.4 * IN), int(1.6 * IN), int(7.6 * IN), int(4.73 * IN))
nh = len(K["heading_roads"])
HS = pd.read_csv(os.path.join(HERE, "..", "results", "heading", "summary.csv"))
HS = HS[(HS.axis == "height") & (HS.level.abs() == 1.0)]
hraw = {lv: HS[(HS.level == lv) & (HS.method == "raw")].lane_f1.iloc[0] for lv in (1.0, -1.0)}
hbest = {lv: HS[(HS.level == lv) & (HS.method != "raw")].lane_f1.max() for lv in (1.0, -1.0)}
hci = lambda lv, best: (lambda r: f"[{r.lane_f1_clo:.2f}, {r.lane_f1_chi:.2f}]")(
    HS[(HS.level == lv) & (HS.method != "raw")].sort_values("lane_f1").iloc[-1] if best else HS[(HS.level == lv) & (HS.method == "raw")].iloc[0])
hname = {lv: NAME[HS[(HS.level == lv) & (HS.method != "raw")].sort_values("lane_f1").method.iloc[-1]] for lv in (1.0, -1.0)}
text(s, int(8.3 * IN), int(1.9 * IN), int(4.4 * IN), int(4.9 * IN), [
    f"Turning ±15° ({nh} roads): raw lane F1 {K['heading_raw_lane_f1']['0']:.2f} → {K['heading_raw_lane_f1']['15']:.2f}.",
    f"Raising it 1 m: raw {hraw[1.0]:.2f} {hci(1.0, False)}, best {hbest[1.0]:.2f} {hci(1.0, True)} ({hname[1.0][0].lower() + hname[1.0][1:]}): the intervals overlap.",
    [(f"Lowering it 1 m: raw {hraw[-1.0]:.2f}, best {hbest[-1.0]:.2f}; only a thin strip of road is comparable there.", {"color": ACC})]], size=19)
footer(s, 9)
notes(s, "Turning the camera by fifteen degrees barely changes lane agreement. Changing its height does, and fixers help "
         f"little there: one meter up, lane agreement is {hraw[1.0]:.2f} raw and {hbest[1.0]:.2f} at best; one meter down, it "
         "nearly vanishes.")

# 10. FaithFusion ---------------------------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK); bg(s, BG)
title(s, "FaithFusion: a modest map, unhelpful weights", "RESULT 5")
picture(s, "img/talk_faithfusion.png", int(0.4 * IN), int(1.6 * IN), int(7.6 * IN), int(4.73 * IN))
eig = [r["eig"] for r in AUC.values()]; cov = [r["cov"] for r in AUC.values()]; pos = [r["pos"] for r in AUC.values()]
text(s, int(8.3 * IN), int(1.9 * IN), int(4.4 * IN), int(4.9 * IN), [
    f"Its information-gain map partly flags invented lane paint one lane over: AUROC {min(eig):.2f}–{max(eig):.2f} across five "
    f"fixers, against {min(cov):.2f}–{max(cov):.2f} for splat coverage alone (invented paint: {min(pos):.0%}–{max(pos):.0%} of road pixels).",
    [("Used as weights in the 3D update, it slightly lowers lane F1 one lane over: " + mn(f"{FF[0]:+.3f} [{FF[1]:+.3f}, {FF[2]:+.3f}]")
      + f" (mean over {FF[3]} roads).", {"color": ACC})],
    [(f"Precision stays low: AUPRC {min(r['auprc'] for r in AUC.values()):.3f}–{max(r['auprc'] for r in AUC.values()):.3f}.", {"color": SOFT})]], size=19)
footer(s, 10)
notes(s, "FaithFusion weights each repaired pixel by how much new information it adds to the splat. Its map does flag "
         "where fixers invent lane paint, better than splat coverage alone, though far from reliably: its precision stays low. But used as weights in the 3D update, it "
         f"slightly lowers lane agreement one lane over, on {FF[3]} roads.")

def table(s, rows, x, y, widths, rh, size=15):
    tb = s.shapes.add_table(len(rows), len(widths), Emu(x), Emu(y), Emu(int(sum(widths) * IN)), Emu(rh * len(rows))).table
    for j, wd in enumerate(widths):
        tb.columns[j].width = Emu(int(wd * IN))
    for i, row in enumerate(rows):
        tb.rows[i].height = Emu(rh)
        for j, v in enumerate(row):
            c = tb.cell(i, j); c.text = v
            c.fill.solid(); c.fill.fore_color.rgb = LIGHT if i == 0 else RGBColor(0xFF, 0xFF, 0xFF)
            p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
            for r in p.runs:
                r.font.name = FONT; r.font.size = Pt(size); r.font.color.rgb = INK; r.font.bold = (i == 0)


# 11. Closed loop --------------------------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK); bg(s, BG)
title(s, "No fixer raises closed-loop driving scores", "RESULT 6")
fx = [f for f in ("difix", "difix_ft", "nv_fixer", "difix_reimpl") if f in CL]
rows = [("HUGSIM, closed loop", "HD-Score", "Change vs. no fixer")] + [("No fixer", f"{CL['none']['hdscore']:.3f}", "–")] + \
       [(NAME[f], f"{CL[f]['hdscore']:.3f}", mn(f"{CL[f]['d']:+.3f} [{CL[f]['lo']:+.3f}, {CL[f]['hi']:+.3f}]")) for f in fx]
RH = int(0.5 * IN)
table(s, rows, int(0.6 * IN), int(1.8 * IN), (2.8, 1.4, 3.3), RH, size=16)
text(s, int(0.6 * IN), int(2.05 * IN) + RH * len(rows), int(7.5 * IN), int(0.9 * IN),
     f"Released LTF policy, {CL['none']['runs'] // 3} scenarios × 3 seeds on nuScenes scenes; paired 95% intervals.", size=16, color=SOFT)
text(s, int(8.5 * IN), int(1.9 * IN), int(4.2 * IN), int(4.9 * IN), [
    "No fixer raises the driving score, our re-implementation included.",
    "NVIDIA Fixer lowers it, mostly through one of the three policy seeds.",
    [("The spread between seeds (0.31–0.46) exceeds every fixer effect.", {"color": ACC})]], size=19)
footer(s, 11)
notes(s, f"In closed-loop driving in HUGSIM, with a released policy over {CL['none']['runs'] // 3} scenarios and three seeds, no "
         "fixer raises the driving score, our re-implementation included; NVIDIA Fixer lowers it slightly, mostly through one "
         "seed. The spread between seeds exceeds any fixer effect.")

# 12. Other cities -------------------------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK); bg(s, BG)
title(s, "Gains shrink in another city", "RESULT 7")
T3 = pd.read_csv(f"{RES}/table3_cross_dataset.csv")
eu = {k: T3[T3.condition.str.startswith(k)].set_index("method") for k in ("EUVS L3", "EUVS rot60")}
ci = lambda d, m: f"{d.loc[m, 'lpips']:.2f} [{d.loc[m, 'lpips_clo']:.2f}, {d.loc[m, 'lpips_chi']:.2f}]"
erows = [("EUVS (US), LPIPS ↓", "new route", "turned 60°")] + \
        [(NAME[m], ci(eu["EUVS L3"], m), ci(eu["EUVS rot60"], m)) for m in ("raw", "difix_ref", "difix_ft")]
table(s, erows, int(0.6 * IN), int(1.8 * IN), (2.8, 2.35, 2.35), RH, size=16)
text(s, int(0.6 * IN), int(2.05 * IN) + RH * len(erows), int(7.5 * IN), int(0.9 * IN),
     "Fixers applied without retraining; 95% intervals over (scene, camera) sequences.", size=16, color=SOFT)
text(s, int(8.5 * IN), int(1.9 * IN), int(4.2 * IN), int(4.9 * IN), [
    "The generic released Difix-ref helps most in both settings.",
    [("Difix-FT, trained on our roads, is worse on the new route and no better than the raw render at the 60° turn.", {"color": ACC})]], size=19)
footer(s, 12)
notes(s, "In a US city the generic Difix-ref helps most, while Difix-FT, trained on our roads, is worse on a new route and no "
         "better at a sixty-degree turn. Fixers trained on matching data do not carry their advantage to other places.")

# 13. Takeaways ------------------------------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK); bg(s, BG2)
text(s, int(0.8 * IN), int(0.7 * IN), int(11.5 * IN), int(0.5 * IN), "TAKEAWAYS", size=16, color=ACC, bold=True)
text(s, int(0.8 * IN), int(1.3 * IN), int(11.7 * IN), int(5.0 * IN), [
    f"1. A splat simulator is trustworthy only near the recorded path: lane F1 falls from {L['raw']['0m']:.2f} to {L['raw']['~3m']:.2f} one lane over.",
    f"2. Fixers trained on matching data recover most of it at 3 m ({L['difix_reimpl_ref']['~3m']:.2f}–{L['difix_ft']['~3m']:.2f}), much less at 6 m (≈{L['difix_ft']['~6m']:.1f}).",
    "3. Realism and correctness diverge: fixers invent lane paint and vehicles; repairing the 3D scene invents the fewest vehicles.",
    "4. Gains depend on the road, shrink with camera height and other cities, and do not raise closed-loop driving scores."],
    size=24, color=BG)
text(s, int(0.8 * IN), int(5.8 * IN), int(11.7 * IN), int(0.9 * IN), [f"Project page, videos and results: {URL}", f"Code: {CODE_URL}"], size=20, color=MUTED)
footer(s, 13, dark=True)
notes(s, "To summarize: trust drops quickly off the recorded path; trained fixers recover much of it one lane over but far "
         "less beyond; realism and correctness diverge; and the gains depend on the road and do not show up in closed-loop "
         "driving. Code, videos and results are linked from our webpage. Thank you.")

prs.save("cs566_drive_sim_talk.pptx")
words = sum(len(sl.notes_slide.notes_text_frame.text.split()) for sl in prs.slides)
print(f"saved: {TOTAL} slides, {words} words of notes (about {words / 140:.1f} min at 140 wpm)")
