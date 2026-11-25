# More Para-Lane test scenes (stage 18)

Entry point: `scripts/18_more_scenes.sh` (stages `screen | fit | fix | eval | prog | road | summary`, one GPU job per
stage and scene, resumable). Summary: `src/eval/more_scenes_summary.py` -> `results/eval_more/` (`tables.md`,
`summary_pooled.csv`, `summary_<scene>.csv`, `deltas.csv`, `deltas_across_scenes.csv`, `screening.json`).

## 1. Screening and choice
Seven not-yet-used Para-Lane 0315 scenes were downloaded with the existing byte-range method
(`scripts/01_download_paralane.sh scene_18 scene_3 scene_4 scene_14 scene_15 scene_11 scene_10`) and screened with
`src/eval/screen_scenes.py`: path length, heading change and curvature from the COLMAP poses, the lateral gaps between
the three passes, and a traffic proxy (movable-object mask area from Para-Lane's foreground labels), plus a contact
sheet of 6 frames per scene (`$DATA/screen/<scene>.jpg`, looked at by eye). None of these scenes is in train
(scene_20–24), val (scene_19) or the original test set (scene_16, 17).

| scene | length (m, mean of 3 passes) | largest heading change (°) | max turn (°/10 m) | lane gaps 0–1 / 1–2 / 0–2 (m) | movable-object area | frames with movable objects | road (contact sheet) |
|---|---|---|---|---|---|---|---|
| scene_16 (test) | 61 | 2.5 | 3.5 | 3.3 / 2.9 / 6.3 | 0.018 | 0.95 | straight, crosswalk / intersection |
| scene_17 (test) | 97 | 8.7 | 1.5 | 3.8 / 3.9 / 7.7 | 0.016 | 0.48 | straight, wide |
| scene_18 | 96 | 6.5 | 2.7 | 3.1 / 0.1 / 3.2 | 0.005 | 0.30 | two passes in the same lane |
| scene_3 | 105 | 1.3 | 0.6 | 2.8 / 3.0 / 5.8 | 0.009 | 0.54 | straight boulevard |
| scene_4 | 102 | 0.6 | 0.8 | 2.6 / 3.2 / 5.7 | 0.011 | 0.70 | straight boulevard |
| **scene_14** | 112 | −50.7 | 30.7 | 2.3 / 3.1 / 5.6 | 0.010 | 0.67 | **curve** |
| **scene_15** | 126 | −0.5 | 1.4 | 4.1 / 3.3 / 7.4 | 0.060 | 1.00 | **under an elevated highway, traffic** |
| scene_11 | 133 | −5.1 | 2.4 | 2.6 / 3.0 / 5.7 | 0.010 | 0.86 | straight boulevard |
| **scene_10** | 131 | −1.6 | 2.3 | 3.0 / 2.5 / 5.6 | 0.014 | 0.91 | **long straight, bus stops** |

Chosen new TEST scenes (never used for training, checkpoint or config selection):
* **scene_14**: curved suburban boulevard, the only real curve in the screened set (heading change 28–51° per pass over
  ~110 m, up to 31° per 10 m); lane gaps 2.3 / 3.1 / 5.6 m.
* **scene_15**: under an elevated highway (pillars, hard shadows) with the most traffic of all scenes (movable-object
  area 0.06, 6× the others; movable objects in every frame); widest gaps (4.1 / 3.3 / 7.4 m).
* **scene_10**: long straight boulevard (131 m) with bus stops and painted arrows; gaps 3.0 / 2.5 / 5.6 m.

Round 2 (Sep 30): **scene_3, scene_11 and scene_4** were added as further TEST roads with the same pipeline
(straight suburban boulevards; 105 / 133 / 102 m; lane gaps 2.8 / 3.0 / 5.8, 2.6 / 3.0 / 5.7 and 2.6 / 3.2 / 5.7 m).
Not chosen: scene_18 (two passes only 0.1 m apart, so it has no second offset). No screened scene contains an intersection
crossing; the crosswalk / intersection case is scene_16 (original test set).

## 2. What ran per scene
As for scene_16/17 (stages 2, 6, 7d, 8), on the stride-2 eval subset (676 / 692 / 706 images for scene_14 / 15 / 10;
scene_3 / 11 / 4 likewise). Round 2 took about 1.5 GPU-h per road (fit 22–27 min, fixers ~40 min, eval 5 min,
progressive 15 min, road prior 26–30 min):
* 3 lane splats (12k iterations) + eval pairs;
* fixers: released Difix, NVIDIA Fixer, Difix-FT, our Difix-ref reimplementation (`difix_reimpl_ref`), our U-Net
  (`hfgan_s0`); released Difix-ref on scene_14 and scene_15 only (1.4 s/img; not one of the key comparisons, skipped
  on scene_10, scene_3, scene_11 and scene_4 for GPU time);
* progressive update with our Difix-ref reimplementation (`prog_difix_reimpl`, same schedule as stage 7d);
* road-surface prior refit (`road_prior`), on the new scenes and on scene_17 (scene_16 had it already);
* metrics: image / paint / YOLO (`src/eval/evaluate.py`) and CLRNet lanes.

## 3. Results (full tables: `results/eval_more/tables.md`; discussion: RESULTS §15; final figures: `results/final/`)
* Per-frame fixers trained on driving pairs (Difix-FT, our Difix-ref reimpl.) raise lane F1 at ~3 m on all eight
  test roads (+0.30 to +0.80 vs. raw); the ranking of §5 holds on every road. The absolute level depends strongly on
  the road: scene_15 (highway shadows, traffic) stays hard for all arms; the round-2 boulevards are the easiest
  (0.93–0.99 for both).
* The progressive update's gain is scene-dependent: large on scene_16, scene_3 and scene_4 (+0.52 to +0.73 lane F1
  at ~3 m), moderate on scene_11 (+0.31), small on the other four (+0.06 to +0.20). Across 8 roads: +0.33
  [+0.19, +0.49] at ~3 m, +0.12 at ~6 m. It costs about +0.05 LPIPS on-path.
* The road prior helps on all eight roads (lane F1 +0.08 to +0.35 at ~3 m), invents nothing, adds little at ~6 m.
* Our Difix-ref reimpl. adds fake vehicles on every road (+0.089 per image at ~3 m [+0.062, +0.117]); Difix-FT does not.
* The paint-mask metric misfires on hard shadows and bright curb strips (scene_15, scene_3/4/11); lane F1 does not.

## Problems met
* `src/splats/render_method.py` reloaded all images of the target clip for every rendered frame
  (`dict.setdefault(k, load_clip(...))` evaluates `load_clip` each time): 7 s per frame. Fixed; scene_14's road-prior
  renders were redone completely after the fix.
* Editing `scripts/18_more_scenes.sh` while a stage was running made that bash process read a shifted file
  ("syntax error at line 53") after its GPU job had finished; the stage's outputs were checked complete (4,732 rows each).
