"""Demo videos + teaser stills for the project webpage (results/videos/, index.json). Every clip is built from outputs
that already exist on disk (renders, fixer outputs, sweeps from src/viz/sweeps.py, HUGSIM closed-loop logs); nothing
here is used for any metric. Labels are burned in; where no real frame exists the tile says so.
usage: scripts/lib/py.sh src/viz/make_videos.py <recipe|all> ...     (recipes: see RECIPES at the bottom)
"""
import glob, json, os, sys
import numpy as np
from PIL import Image

from common.config import DATA, REPO
from viz.compose import write_video, still

OUT = f"{REPO}/results/videos"
F = f"{DATA}/fixed"; P = f"{DATA}/pairs"; V = f"{DATA}/viz"
ATTR = {
    "paralane": "Data: Para-Lane (Ni et al., 3DV 2025), CC BY-NC 4.0. Renders/fixes: CS 566 course project, non-commercial.",
    "hugsim": "Data: HUGSIM released nuScenes scenes (MIT); nuScenes by Motional, CC BY-NC-SA 4.0. CS 566, non-commercial.",
    "euvs": "Data: EUVS benchmark (ICCV 2025) MARS scenes (MARS, CVPR 2024; CC BY-NC-SA 4.0). CS 566, non-commercial.",
    "mv2": "Data: MV2 multi-vehicle dataset (arXiv 2608.12442), academic research use only. CS 566, non-commercial.",
    "dl3dv": "Data: DL3DV-10K sample (Ling et al., CVPR 2024), DL3DV terms of use (non-commercial). CS 566, non-commercial.",
}
NAMES = {"raw": "raw splat", "difix": "released Difix", "difix_ref": "released Difix-ref", "nv_fixer": "NVIDIA Fixer",
         "difix_ft": "Difix-FT (ours, fine-tuned)", "difix_reimpl_ref": "our Difix (reimpl., ref)",
         "ours_hfgan_s0": "our small U-Net", "prog_difix_reimpl": "progressive (our Difix)",
         "road_prior": "road-prior splat (no fixer)", "prog_difix_ft": "progressive (Difix-FT)",
         "prog_none": "refined without fixer (control)"}
INDEX = f"{OUT}/index.json"


def add_index(entry):
    if entry["file"].startswith(("mv2", "dl3dv")):  # local-only clips (licences), never listed
        return
    idx = json.load(open(INDEX)) if os.path.exists(INDEX) else {"videos": [], "stills": []}
    idx["about"] = ("Demo clips of the CS 566 (UW-Madison, Fall 2026) course project 'How far off the recorded path can a "
                    "Gaussian-splat driving simulator be trusted?' (Kyle Sha, Allen Xia). Non-commercial, educational use. "
                    "Each entry carries its dataset attribution; no raw datasets or model weights are distributed. "
                    "Labels are burned into every frame; tiles without a real frame say so.")
    idx["format"] = "H.264 (libx264, yuv420p, faststart) MP4, 1280 px wide, <= 20 s, <= 4.5 MB; stills JPEG <= 500 KB"
    key = "stills" if entry["file"].endswith(".jpg") else "videos"
    idx[key] = [e for e in idx[key] if e["file"] != entry["file"]] + [entry]
    idx[key].sort(key=lambda e: e["file"])
    json.dump(idx, open(INDEX, "w"), indent=1, ensure_ascii=False)


def side(off):
    return f"{abs(off):.1f} m {'left' if off > 0 else 'right'}" if abs(off) > 0.05 else "0 m (recorded lane)"


def cross_items(scene, src, dst):
    from fixers.apply import eval_items  # the stride-2 eval subset the fixers were run on
    M = [m for m in eval_items(P, scene, 2) if m["src"] == src and m["dst"] == dst]
    return sorted(M, key=lambda m: int(m["ts"]))


MARKS = "Marks (same rules as the metrics): orange = invented lane paint, blue = erased lane paint, red box = spurious vehicle."
LEGEND = [((255, 140, 0), "invented paint"), ((40, 120, 255), "erased paint"), ((230, 30, 30), "spurious vehicle")]


def paralane_cross(name, scene, src, dst, methods, title, scenario, what, caption, frames=None, fps=5, extra_note=None,
                   cols=3, tile_w=426, marks=False):
    items = cross_items(scene, src, dst)
    if frames:
        items = items[frames[0]:frames[1]]
    items = [m for m in items if all(os.path.exists(f"{F}/{me}/{m['path']}") for me in methods if me != "raw")]
    assert items, (scene, src, dst)
    off = float(np.median([m["offset_m"] for m in items]))
    title = title.format(off=f"median {abs(off):.1f} m")  # on-screen offset = median |offset| of the frames shown (= offset_m below)
    fr = []
    if marks:  # invented / erased paint and spurious vehicles drawn on every output tile (src/viz/annotate.py)
        from viz.annotate import annotate
        extra_note = MARKS
    for m in items:
        tiles = [(f"real frame, lane {dst[-1]}", f"{P}/{m['real']}")]
        if marks:
            real = np.array(Image.open(f"{P}/{m['real']}").convert("RGB")); fg = np.array(Image.open(f"{P}/{m['fg']}")) > 127
        for me in methods:
            p = f"{P}/{m['path']}" if me == "raw" else f"{F}/{me}/{m['path']}"
            if marks:
                p = annotate(np.array(Image.open(p).convert("RGB").resize(real.shape[1::-1])), real, fg)[0]
            tiles.append((NAMES.get(me, me), p))
        fr.append(dict(tiles=tiles, status=f"Para-Lane {scene}: splat of lane {src[-1]} rendered at lane {dst[-1]}'s real "
                                            f"poses, lateral offset in this frame {side(m['offset_m'])}", note=extra_note,
                       legend=LEGEND if marks else None))
    info = write_video(f"{OUT}/{name}.mp4", fr, title, ATTR["paralane"], fps=fps, cols=cols, tile_w=tile_w)
    add_index(dict(file=f"{name}.mp4", title=title, scenario=scenario, dataset="Para-Lane", scene=scene,
                   shown=what, caption=caption + (" " + MARKS if marks else ""), offset_m=round(abs(off), 2), real_ground_truth=True,
                   attribution=ATTR["paralane"], **info))
    return fr


def r_straight():
    paralane_cross("paralane_straight_3m_scene16", "scene_16", "clip_0", "clip_1",
                   ["raw", "difix", "difix_reimpl_ref", "ours_hfgan_s0", "nv_fixer"],
                   "Para-Lane scene_16: straight road, one lane over ({off})", "lateral offset, straight road",
                   "real adjacent-lane frame | raw splat | released Difix | our Difix reimplementation (ref) | our U-Net | NVIDIA Fixer",
                   "The splat was fitted on lane 0 only and is rendered where lane 1 actually drove. Released Difix sharpens "
                   "but does not restore the missing lane paint; our driving-trained Difix restores most lane lines; the "
                   "small U-Net changes little. Cars differ between passes (the lanes were driven at different times).",
                   marks=True)


def r_far():
    paralane_cross("paralane_far_lane_7m_scene17", "scene_17", "clip_0", "clip_2",
                   ["raw", "difix", "difix_reimpl_ref", "ours_hfgan_s0", "nv_fixer"],
                   "Para-Lane scene_17: straight road, two lanes over ({off})", "lateral offset, far lane",
                   "real far-lane frame | raw splat | released Difix | our Difix reimplementation (ref) | our U-Net | NVIDIA Fixer",
                   "At 7.7 m the splat has never seen most of the road surface. Every fixer produces a cleaner-looking image, "
                   "but none recovers the real lane geometry reliably; lane agreement stays low (RESULTS section 5).", marks=True)


def r_progressive():
    paralane_cross("progressive_before_after_scene16", "scene_16", "clip_0", "clip_1",
                   ["raw", "prog_none", "prog_difix_ft", "prog_difix_reimpl"],
                   "Para-Lane scene_16: progressive 3D update, before / after ({off})", "progressive update",
                   "real | original splat | refined without fixer (control) | progressive with Difix-FT | progressive with our Difix",
                   "The splat itself is refined with repaired pseudo-views at ±3 m and ±6 m (a simplified Difix3D+ progressive "
                   "update) and then rendered without any fixer. On scene_16 the lane lines return in 3D (lane F1 at ~3 m: 0.18 "
                   "for the raw render, 0.91 with our re-implementation, 0.90 with Difix-FT); on scene_17 the same procedure "
                   "helps far less, so this gain depends on the road.",
                   extra_note="All four renders are plain splat renders (no 2D fixer at test time). The control runs the same "
                              "refinement with raw renders as pseudo-views.")


def r_road():
    paralane_cross("road_prior_before_after_scene16", "scene_16", "clip_0", "clip_1", ["raw", "road_prior"],
                   "Para-Lane scene_16: road-surface prior, before / after ({off})", "road prior",
                   "real | raw splat | splat fitted with the LiDAR road-surface prior",
                   "A height/slope field fitted to LiDAR pulls road Gaussians onto the road plane and flattens them. No "
                   "generative model is involved, so nothing can be invented; lane lines are sharper off-path.",
                   extra_note="Road prior (ViSE-style, our formulation): road Gaussians snapped to a LiDAR height field and "
                              "flattened along its normal. It cannot hallucinate content; it only regularises geometry.",
                   cols=2, tile_w=560)


def r_euvs(scene="MARS_location_41_scene_02", proto="L3", n=90):
    E = f"{DATA}/euvs/out"; EF = f"{DATA}/euvs/fixed"
    M = [m for m in json.load(open(f"{E}/{proto}/meta_all.json")) if m["scene"] == scene and m["channel"] == 2]
    M.sort(key=lambda m: m["name"]); M = M[:n]
    methods = ["raw", "difix_ref", "nv_fixer", "difix_reimpl_ref", "ours_hfgan_s0"]
    fr = []
    for m in M:
        tiles = [("real frame (EUVS test)", f"{E}/{m['real']}")]
        for me in methods:
            tiles.append((NAMES[me], f"{E}/{m['path']}" if me == "raw" else f"{EF}/{me}/{m['path']}"))
        fr.append(dict(tiles=tiles, status=f"EUVS {proto} (MARS {scene.replace('MARS_', '')}): held-out traversal, nearest "
                                            f"training camera {m['dist_m']:.1f} m / {m['rot_deg']:.0f} deg away"))
    title = "EUVS / MARS location 41: different route through the same place (transfer)"
    info = write_video(f"{OUT}/euvs_mars_L3.mp4", fr, title, ATTR["euvs"], fps=5)
    add_index(dict(file="euvs_mars_L3.mp4", title=title, scenario="translation + rotation (EUVS level 3), transfer",
                   dataset="EUVS (MARS)", scene=scene, real_ground_truth=True,
                   shown="real test frame | raw splat | released Difix-ref | NVIDIA Fixer | our Difix (ref) | our U-Net",
                   caption="The test traversal takes another route through the same place (median 5.5 m and 89 deg from "
                           "the nearest training camera). Only the released, generically trained fixers help here; our "
                           "Para-Lane-trained models do not transfer (RESULTS section 13).",
                   attribution=ATTR["euvs"], **info))


def r_hugsim(scene="scene-0064"):
    R = f"{DATA}/hugsim/renders"; HF = f"{DATA}/hugsim/fixed"
    idx = json.load(open(f"{R}/{scene}/index.json")); frames = sorted({v["idx"] for v in idx})
    sched = np.interp(np.arange(len(frames)), [0, 5, 12, 19, 26, len(frames) - 1], [0, 0, 1, 2, 3, 3]).round().astype(int)
    methods = ["difix", "nv_fixer", "difix_ft", "ours"]
    names = dict(NAMES, ours="our small U-Net (base)")
    fr = []
    for k, i in zip(sched, frames):
        stem = f"off+{k}/{i:05d}.png"
        tiles = [("real nuScenes frame", None, "HUGSIM's released scenes ship no\nreal images; there is no real\nframe at any offset"),
                 ("raw HUGSIM render", f"{R}/{scene}/{stem}")]
        for me in methods:
            tiles.append((names[me], f"{HF}/{me}/{scene}/{stem}"))
        fr.append(dict(tiles=tiles, status=f"HUGSIM nuScenes {scene}: ego shifted {k} m to the right of the logged path"))
    title = "HUGSIM nuScenes scene-0064: ego car shifted 0 -> 3 m (no real frame)"
    info = write_video(f"{OUT}/hugsim_nuscenes_offsets.mp4", fr, title, ATTR["hugsim"], fps=4)
    add_index(dict(file="hugsim_nuscenes_offsets.mp4", title=title, scenario="lateral offset in a released simulator scene",
                   dataset="nuScenes via HUGSIM", scene=scene, real_ground_truth=False,
                   shown="(no real frame) | raw HUGSIM render | released Difix | NVIDIA Fixer | Difix-FT | our U-Net",
                   caption="HUGSIM's released reconstruction rendered 0-3 m off the logged path. Detector agreement with the "
                           "0 m render falls ~16% by 3 m; NVIDIA Fixer recovers most, our Para-Lane-trained models do not "
                           "transfer (our U-Net washes out vehicles; RESULTS section 8).",
                   attribution=ATTR["hugsim"], **info))


def r_closed_loop(run="none/seed0/scene-0167-hard-00"):
    import imageio, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from viz.compose import font
    from PIL import ImageDraw
    import pickle
    D = f"{DATA}/hugsim/closed_loop/{run}"
    ev = json.load(open(f"{D}/eval.json")); frames = pickle.load(open(f"{D}/data.pkl", "rb"))[0]["frames"]
    pv = [im for im in imageio.get_reader(f"{D}/policy_view.mp4")]
    n = min(len(pv), len(frames)); ego = np.array([f["ego_box"][:2] for f in frames[:n]])
    out = []
    for t in range(n):
        f = frames[t]
        fig = plt.figure(figsize=(6.4, 2.9), dpi=100); ax = fig.add_axes([0.08, 0.12, 0.9, 0.84])
        ax.plot(ego[:, 1] * -1, ego[:, 0], color="#bbbbbb", lw=1, ls="--", label="ego path (whole run)")
        ax.plot(ego[: t + 1, 1] * -1, ego[: t + 1, 0], color="#1f77b4", lw=2, label="ego path so far")
        trw = np.array([[float(v) for v in p[:2]] for p in f["planned_traj"]["traj"]])  # already in the scenario frame
        x0, y0 = f["ego_box"][0], f["ego_box"][1]
        ax.plot(-trw[:, 1], trw[:, 0], color="#2ca02c", lw=3.5, zorder=5, label="LTF planned path")
        for b, nm in zip(f["obj_boxes"], f["obj_names"]):
            ax.add_patch(plt.Rectangle((-b[1] - b[3] / 2, b[0] - b[4] / 2), b[3], b[4], color="#d62728", alpha=0.7))  # w across, l along
        ax.add_patch(plt.Rectangle((-y0 - 0.8, x0 - 1.5), 1.6, 3.0, color="#1f77b4"))
        ax.set_xlim(-y0 - 25, -y0 + 25); ax.set_ylim(x0 - 8, x0 + 32); ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=7); ax.legend(fontsize=7, loc="upper left"); ax.set_xlabel("m (bird's-eye view, ego frame of the scenario)", fontsize=7)
        fig.canvas.draw(); bev = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)
        top = Image.fromarray(pv[t]).resize((1280, 256), Image.LANCZOS)
        canvas = Image.new("RGB", (1280, 50 + 256 + 290 + 24), (16, 16, 20)); canvas.paste(top, (0, 50))
        canvas.paste(Image.fromarray(bev), (0, 306))
        d = ImageDraw.Draw(canvas)
        d.text((10, 5), "HUGSIM closed loop: the LTF driving policy drives inside the splat", fill=(255, 255, 255), font=font(18, True))
        d.text((10, 28), f"nuScenes {run.split('/')[-1]} (no fixer), step {t + 1}/{n} at 4 Hz, collision: {f['collision']}",
               fill=(255, 220, 120), font=font(15))
        txt = (f"Top: the three front cameras the policy sees (rendered by HUGSIM).\nBottom: bird's-eye view; blue = ego, "
               f"red = other vehicles, green = the policy's planned path.\n\nFinal scores of this run: HD-Score "
               f"{ev['hdscore']:.2f}, route completion {ev['rc']:.2f}, no-collision {ev['nc']:.0f}, drivable-area "
               f"compliance {ev['dac']:.2f}.\nNo real camera exists off the logged path, so this shows the simulator loop, "
               f"not an accuracy check.")
        from viz.compose import _wrap
        _wrap(d, txt, 660, 316, 600, font(14), (220, 220, 220))
        d.text((10, canvas.size[1] - 20), ATTR["hugsim"], fill=(160, 160, 170), font=font(12))
        out.append(np.asarray(canvas))
    import subprocess, imageio_ffmpeg
    path = f"{OUT}/hugsim_closed_loop_ltf.mp4"; Hh, Ww = out[0].shape[:2]
    for crf in (23, 27, 31):
        p = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                              "-s", f"{Ww}x{Hh}", "-r", "5", "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                              "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
        for im in out[:100]:
            p.stdin.write(np.ascontiguousarray(im).tobytes())
        p.stdin.close(); p.wait()
        if os.path.getsize(path) / 2 ** 20 <= 9.5:
            break
    info = dict(frames=min(n, 100), fps=5, seconds=min(n, 100) / 5, width=Ww, height=Hh, mb=round(os.path.getsize(path) / 2 ** 20, 2), crf=crf)
    print("wrote", path, info)
    title = "HUGSIM nuScenes scene-0167: closed loop, LTF policy, no fixer"
    add_index(dict(file="hugsim_closed_loop_ltf.mp4", title=title, scenario="closed-loop driving in the simulator",
                   dataset="nuScenes via HUGSIM", scene=run, real_ground_truth=False,
                   shown="policy's three front camera views + bird's-eye view (ego, other vehicles, planned path)",
                   caption=f"One closed-loop rollout (hard scenario, seed 0): HD-Score {ev['hdscore']:.2f}, route completion "
                           f"{ev['rc']:.2f}. Played at 5 fps (sim runs at 4 Hz). Across 25 scenarios x 3 seeds the no-fixer "
                           f"HD-Score is 0.41 (RESULTS section 8).",
                   attribution=ATTR["hugsim"], **info))


def sweep(kind, name, title, scenario, what, caption, note, fps=6):
    M = json.load(open(f"{V}/{kind}_scene_16/meta.json")); VF = f"{V}/fixed"
    fr = []
    for m in M:
        if kind == "lane":
            gt = (("nearest real frame, " + m["gt_kind"].split("of ")[1].split(" (")[0].replace("clip_", "lane ")) if m["gt"] else "real frame",
                  f"{V}/{m['gt']}" if m["gt"] else None, "no real frame at this offset\n(real lanes only at ~3.3 m and ~6.2 m)")
            status = f"Para-Lane scene_16, lane 0's splat, camera shifted {side(-m['off'])} of the recorded path"  # sweep off: + = right
            tiles = [gt, ("raw splat", f"{V}/{m['path']}"), ("released Difix", f"{VF}/difix/{m['path']}"),
                     ("our Difix (reimpl., ref)", f"{VF}/difix_reimpl_ref/{m['path']}"),
                     ("our small U-Net", f"{VF}/ours_hfgan_s0/{m['path']}"),
                     ("road-prior splat (no fixer)", f"{V}/{m['path'].replace('/raw/', '/alt_road/')}"),
                     ("progressive (our Difix) splat", f"{V}/{m['path'].replace('/raw/', '/alt_prog/')}")]
        elif kind == "yaw":
            status = f"Para-Lane scene_16, lane 0, camera turned {abs(m['yaw']):.0f} deg {'left' if m['yaw'] > 0 else 'right'} about its centre"
            tiles = [("real frame rotated (exact GT)", f"{V}/{m['gt']}"), ("raw splat", f"{V}/{m['path']}"),
                     ("released Difix", f"{VF}/difix/{m['path']}"), ("our Difix (reimpl., ref)", f"{VF}/difix_reimpl_ref/{m['path']}"),
                     ("our small U-Net", f"{VF}/ours_hfgan_s0/{m['path']}"), ("NVIDIA Fixer", f"{VF}/nv_fixer/{m['path']}")]
        else:
            status = f"Para-Lane scene_16, lane 0, camera {abs(m['dz']):.2f} m {'higher' if m['dz'] > 0 else 'lower'} than recorded"
            tiles = [("real frame", None, "no real frame exists at a\ndifferent camera height"),
                     ("recorded real frame (0 m)", f"{V}/{m['ref']}"), ("raw splat", f"{V}/{m['path']}"),
                     ("released Difix", f"{VF}/difix/{m['path']}"), ("our Difix (reimpl., ref)", f"{VF}/difix_reimpl_ref/{m['path']}"),
                     ("our small U-Net", f"{VF}/ours_hfgan_s0/{m['path']}"), ("NVIDIA Fixer", f"{VF}/nv_fixer/{m['path']}")]
        fr.append(dict(tiles=tiles, status=status, note=note))
    cols, tw = (4, 318) if len(fr[0]["tiles"]) > 6 else (3, 426)
    info = write_video(f"{OUT}/{name}.mp4", fr, title, ATTR["paralane"], cols=cols, tile_w=tw, fps=fps)
    add_index(dict(file=f"{name}.mp4", title=title, scenario=scenario, dataset="Para-Lane", scene="scene_16",
                   shown=what, caption=caption, attribution=ATTR["paralane"], **info,
                   real_ground_truth={"lane": "partial: only at the two real adjacent lanes (another pass, traffic differs)",
                                      "yaw": "exact inside the region the recorded camera saw (black outside)",
                                      "height": False}[kind]))


def r_lane_sweep():
    sweep("lane", "lane_change_sweep_0_3_6m", "Para-Lane scene_16: lane-change sweep, 0 -> 3 -> 6 m",
          "continuous lateral sweep",
          "nearest real frame (only on the two real lanes) | raw | released Difix | our Difix (ref) | our U-Net | road prior | progressive splat",
          "The camera slides from the recorded lane to the next lane (3.3 m) and the one after (6.2 m). Real frames exist "
          "only where Para-Lane actually drove (shown as the nearest real frame of that lane, driven at another time, so "
          "traffic differs); in between there is no ground truth.",
          "Continuous sweep for illustration. Where no real lane was driven, no ground truth exists; the metrics in the "
          "report use only real adjacent-lane frames.")


def r_yaw_sweep():
    sweep("yaw", "heading_sweep_pm15deg", "Para-Lane scene_16: heading sweep, -15 to +15 deg",
          "heading (yaw) about the camera centre",
          "real frame rotated by the exact homography (black = never seen by the real camera) | raw | released Difix | our Difix (ref) | our U-Net | NVIDIA Fixer",
          "For a pure rotation the real frame can be warped exactly; black regions are parts of the scene the recorded "
          "camera never saw. Inside the seen region every method stays close to the real frame (heading is not what "
          "breaks the splat); the newly revealed regions at the image border are where splats and fixers improvise.", None)


def r_height_sweep():
    sweep("height", "camera_height_sweep_pm1m", "Para-Lane scene_16: camera-height sweep, -1 to +1 m (no real frame)", "camera height",
          "(no real frame) | recorded real frame at 0 m | raw | released Difix | our Difix (ref) | our U-Net | NVIDIA Fixer",
          "Lower or higher cameras (a sports car, a truck) see the road at a different angle. There is no real frame at "
          "another height, so this is a visual check only; the report scores the road region against a road-plane warp "
          "(RESULTS section 12) and against real car/scooter pairs on MV2.",
          "No ground truth at other heights. The recorded frame (0 m) is shown for reference only.")


def r_dl3dv(level=60, n=40):
    D = f"{DATA}/dl3dv"; T = json.load(open(f"{D}/dl3dv_test_s{level}.json"))["test"]
    keys = sorted(T); scenes = sorted({T[k]["scene"] for k in keys})
    sc = scenes[1]; keys = [k for k in keys if T[k]["scene"] == sc][:n]
    models = [("released Difix", "released"), ("released Difix-ref", "released_ref"), ("official trainer (matched)", "official448"),
              ("our trainer (matched)", "ours448"), ("our trainer, ref view", "oursref")]
    fr = []
    for k in keys:
        it = T[k]
        tiles = [("real frame (held out)", it["target_image"]), ("raw sparse splat", it["image"])]
        tiles += [(lab, f"{D}/out/{m}/{k}.png") for lab, m in models]
        fr.append(dict(tiles=tiles, status=f"DL3DV-10K sample scene {sc[:8]} (held out): splat from every {level}th frame, "
                                            f"rendered at a held-out frame", note=f"All Difix models from SD-Turbo; matched runs: "
                                            f"1,801 steps, no reference view, 448x768, same pairs. The released models were "
                                            f"trained by NVIDIA on DL3DV at a much larger budget."))
    title = "DL3DV: our Difix reimplementation vs. the official trainer and the released weights"
    info = write_video(f"{OUT}/dl3dv_difix_validation.mp4", fr, title, ATTR["dl3dv"], cols=4, tile_w=318, fps=4)
    add_index(dict(file="dl3dv_difix_validation.mp4", title=title, scenario="sparse-view reconstruction (fixer validation)",
                   dataset="DL3DV-10K (sample)", scene=sc, real_ground_truth=True,
                   shown="real held-out frame | raw sparse-view splat | released Difix | released Difix-ref | official trainer | our trainer | our trainer with reference view",
                   caption="Validation of our Difix training code on the paper's own data type (docs/difix_reimpl_validation.md).",
                   attribution=ATTR["dl3dv"], license_note="DL3DV is gated (terms of use); check its terms before publishing frames.", **info))


def r_curve():
    paralane_cross("paralane_curve_scene14", "scene_14", "clip_0", "clip_2",
                   ["raw", "difix", "difix_reimpl_ref", "ours_hfgan_s0", "nv_fixer"],
                   "Para-Lane scene_14 (new test road): curve, two lanes over ({off})", "lateral offset, curved road (new test scene)",
                   "real far-lane frame | raw splat | released Difix | our Difix reimplementation (ref) | our U-Net | NVIDIA Fixer",
                   "scene_14 was added as a held-out test scene (never used for training or selection): the road bends by 30-50 deg over ~110 m (per pass). "
                   "Lane 0's splat is rendered where lane 2 drove. Our driving-trained Difix draws crisp lane lines, "
                   "but not always where the real ones are (compare the markings with the real frame): plausible is not "
                   "the same as faithful. Per-scene numbers: RESULTS section 15.", marks=True)


def r_highway():
    paralane_cross("paralane_under_highway_traffic_scene15", "scene_15", "clip_0", "clip_1",
                   ["raw", "difix", "difix_reimpl_ref", "ours_hfgan_s0", "nv_fixer"],
                   "Para-Lane scene_15 (new test road): under a highway, one lane over ({off})", "lateral offset, shadows + traffic (new test scene)",
                   "real adjacent-lane frame | raw splat | released Difix | our Difix reimplementation (ref) | our U-Net | NVIDIA Fixer",
                   "scene_15 (held-out test scene) runs under an elevated highway: hard shadows, pillars and the most traffic "
                   "of all Para-Lane scenes we used. Vehicles are pasted from the real frame only for the metrics, not here: "
                   "cars in the renders are whatever the splat and fixer produce (the lanes were driven at different times). "
                   "Per-scene numbers: RESULTS section 15.", marks=True)


def r_mv2(seq=None):
    O = f"{DATA}/mv2/out"; MF = f"{DATA}/mv2/fixed"
    M = json.load(open(f"{O}/car2scooty/meta_all.json"))
    seqs = sorted({m["scene"] for m in M}); seq = seq or seqs[0]
    M = sorted([m for m in M if m["scene"] == seq and m["kind"] == "cross"], key=lambda m: m["frame"])
    methods = ["raw", "difix", "difix_ref", "nv_fixer", "difix_reimpl_ref", "ours_hfgan_s0"]
    methods = methods[:5]  # tiles: real + 5
    M = [m for m in M if all(os.path.exists(f"{MF}/{me}/{m['path']}") for me in methods if me != "raw")]
    fr = []
    for m in M:
        tiles = [("real scooter frame", f"{O}/{m['real']}")]
        tiles += [(NAMES[me], f"{O}/{m['path']}" if me == "raw" else f"{MF}/{me}/{m['path']}") for me in methods[:5]]
        fr.append(dict(tiles=tiles, status=f"MV2 {seq}: splat from the CAR's camera rendered at the SCOOTER's real pose "
                                            f"({abs(m['lat_m']):.1f} m lateral, {abs(m['dh_m']):.2f} m lower camera, simultaneous)"))
    title = f"MV2 {seq}: car -> scooter, other lane and lower camera"
    info = write_video(f"{OUT}/mv2_car_to_scooter.mp4", fr, title, ATTR["mv2"], fps=4)
    add_index(dict(file="mv2_car_to_scooter.mp4", title=title, scenario="lateral offset + camera height (real, simultaneous)",
                   dataset="MV2", scene=seq, real_ground_truth=True,
                   shown="real scooter frame | raw splat (car views only) | released Difix | released Difix-ref | NVIDIA Fixer | our Difix (ref)",
                   caption="A car and a scooter drove the same street at the same time in different lanes, with the "
                           "scooter's GoPro mounted lower. The splat is built from the car only and checked against what "
                           "the scooter really saw (Indian roads; all fixers are used without retraining; RESULTS section 14).",
                   attribution=ATTR["mv2"], **info))


def r_stills():
    """Teaser JPGs for the site header (<= 500 KB): one Para-Lane cross-lane frame, one heading-sweep frame."""
    from fixers.apply import eval_items
    m = [x for x in eval_items(P, "scene_16", 2) if x["src"] == "clip_0" and x["dst"] == "clip_1"]
    m = sorted(m, key=lambda x: int(x["ts"]))[20]
    fr = dict(tiles=[("real frame, lane 1", f"{P}/{m['real']}"), ("raw splat (from lane 0)", f"{P}/{m['path']}"),
                     ("released Difix", f"{F}/difix/{m['path']}"), ("our Difix (reimpl., ref)", f"{F}/difix_reimpl_ref/{m['path']}"),
                     ("our small U-Net", f"{F}/ours_hfgan_s0/{m['path']}"), ("NVIDIA Fixer", f"{F}/nv_fixer/{m['path']}")],
              status=f"Para-Lane scene_16: splat of lane 0 rendered where lane 1 really drove ({side(m['offset_m'])})")
    title = f"Para-Lane scene_16: one lane over ({abs(m['offset_m']):.1f} m)"
    still(f"{OUT}/teaser_cross_lane.jpg", fr, title, ATTR["paralane"])
    add_index(dict(file="teaser_cross_lane.jpg", title=title, dataset="Para-Lane", scene="scene_16",
                   shown="real | raw | released Difix | our Difix | our U-Net | NVIDIA Fixer",
                   caption="One lane over: fixers make the render prettier; only the driving-trained one restores the real lane paint.",
                   attribution=ATTR["paralane"]))
    Ms = json.load(open(f"{V}/lane_scene_16/meta.json")); k = len(Ms) - 5; mm = Ms[k]
    fr = dict(tiles=[("nearest real frame, far lane", f"{V}/{mm['gt']}" if mm["gt"] else None, "no real frame"),
                     ("raw splat", f"{V}/{mm['path']}"), ("our Difix (reimpl., ref)", f"{V}/fixed/difix_reimpl_ref/{mm['path']}")],
              status=f"Para-Lane scene_16: lane 0's splat, camera {side(-mm['off'])} of the recorded path")
    still(f"{OUT}/teaser_far_lane.jpg", fr, f"Para-Lane scene_16: lane-change sweep, {abs(mm['off']):.1f} m off the recorded lane", ATTR["paralane"], cols=3)
    add_index(dict(file="teaser_far_lane.jpg", title=f"Para-Lane scene_16: lane-change sweep, {abs(mm['off']):.1f} m off the recorded lane", dataset="Para-Lane", scene="scene_16",
                   shown="nearest real far-lane frame | raw splat | our Difix", attribution=ATTR["paralane"],
                   caption="At ~6 m the splat has little to go on; the fixer produces a clean road, but not always the real one."))


# r_dl3dv and r_mv2 are not in RECIPES and their outputs are git-ignored: DL3DV terms restrict redistribution of its
# frames and MV2 is licensed for academic research use only (run them by hand for local viewing: make_videos.py mv2)
LOCAL_ONLY = {"mv2": r_mv2, "dl3dv": r_dl3dv}
RECIPES = {"straight": r_straight, "curve": r_curve, "highway": r_highway, "stills": r_stills, "lane_sweep": r_lane_sweep, "yaw_sweep": r_yaw_sweep, "height_sweep": r_height_sweep,
           "far": r_far, "progressive": r_progressive, "road": r_road, "euvs": r_euvs,
           "hugsim": r_hugsim, "closed_loop": r_closed_loop}

if __name__ == "__main__":
    todo = list(RECIPES) if sys.argv[1] == "all" else sys.argv[1:]
    os.makedirs(OUT, exist_ok=True)
    for r in todo:
        (RECIPES.get(r) or LOCAL_ONLY[r])()
