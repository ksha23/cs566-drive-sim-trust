# FaithFusion [2] arm: EIG weighting and "does the map predict hallucination?"

FaithFusion (Wang et al., CVPR 2026, arXiv 2511.21113, github.com/wangyuanbiubiubiu/FaithFusion, MIT; tested at
commit `dfd8c5d`) fuses a 3DGS street reconstruction with a diffusion restorer. A per-pixel **Expected Information Gain
(EIG)** map decides how strongly generated content may overwrite the reconstruction. Our code is in
`src/progressive/faithfusion/`, and the entry point is `scripts/14_faithfusion.sh`.

## 1. How EIG is computed (their code, reused)
We read `models/trainers/base.py:render_uncertainly_gaussians`, `models/video_utils.py` and `diff/` (their modified
3DGS rasterizer). **We build and call that rasterizer unchanged** (`scripts/14_faithfusion.sh setup`, a venv over our
project env). Its backward pass uses `atomicAddSquare`, so parameter "gradients" are sums of *squared* per-pixel
Jacobians: a diagonal Fisher information, as in FisherRF.

Our reimplementation of the driver logic is `eig.py`, a line-for-line port of their procedure:
1. **Accumulate the training-view Fisher information.** For each training view of the source lane, render with SH colors and
   back-propagate `non_sky / hit_count` per pixel. Sum the squared gradients over means, rotations, scales,
   opacities and SH into one scalar per Gaussian, `H_train`. Then set `I_train = 1/(H_train + 1e-6)`.
2. **Accumulate the novel-view information.** For a novel view, back-propagate ones to get `H_view`, then set `I_acq = H_view * I_train` (per Gaussian).
3. **Render the pixel-wise map.** Render `I_acq` as a color (alpha-blended), mask pixels with depth ≤ 0.1, and divide by the per-pixel hit count and by 1000.
   The map is `log(1 + ·)`. Uncovered pixels (opacity < 0.1) get 100, meaning "generate freely".

The paper's bound is EIG ≤ ½ Σ tr(H[novel] H[train]⁻¹); the code uses the diagonal, per-Gaussian version above.

Our deviations from their pipeline:
- **Gaussians.** They come from our gsplat splats. The parameterization is identical (wxyz quaternions, log-scales,
  logit-opacities, degree-3 SH, evaluated at degree 1 in their EIG rasterizer as in their code).
- **Sky mask** for step 1: we use splat alpha < 0.5; they use Depth-Anything-V2.
- **Coverage alpha** for the "uncovered = 100" rule: taken from our gsplat render.

Sanity check, scene_16 lane 0 splat (`H_train` takes 3.7 s over 95 views; one map takes about 0.05 s):
- **On-path held-out view:** median log-EIG 0.036.
- **Lane-2 pose, about 6 m off:** median 0.48, 75th percentile 2.9.
- **Where it is high:** mostly on the smeared near road, exactly where the raw render is worst. See `results/faithfusion/ff_examples_scene_16_difix.jpg`.

## 2. How EIG weights the progressive update
- **Their code** (`base.py`, EIGent branch) buckets the log-EIG map into per-pixel **L1** weights:
  - < 0.05 → 0.1
  - [0.05, 0.2) → 0.3
  - [0.2, 1.0) → 0.7
  - ≥ 1.0 → 1.0
  SSIM is scaled by 0.7.
- **Their paper** writes λ_EIG ⊙ (λ_r L1 + (1−λ_r) L_SSIM). We follow the code.
- **Their repo turns the weighting on only for their own restorer "EIGent"**, whose weights are not released. With
  `fix_model: difix` the gain is set to `None`, and every pseudo-view pixel gets a flat weight of 0.7
  (`datasets/base/split_wrapper.py`, `base.py`).
- **Their Difix path** uses `difix_ref` with the on-path real frame as reference, at 1024×576.

**Our two arms** (`prog_ff.py`). They are identical except for the weights:
- **flat:** FaithFusion's Difix path, weight 0.7 everywhere and SSIM × 0.7.
- **eig:** their EIG buckets, computed on the *current* splat at each pseudo pose (recomputed every round), and SSIM × 0.7.

**Shared settings for both arms:**
- **Restorer:** released Difix-ref (`nvidia/difix_ref`), with the lane's own real image at the source pose as reference, run at our 960×540.
- **Mask:** our DriveRestorer-style validity mask (no sky, no far region).
- **Schedule:** rounds at ±2, ±4 and ±6 m, every 3rd training pose, 2k refinement iterations per round, pseudo-view probability 0.4.
- **Hook:** the per-pixel weights reach the splat optimizer through an optional `weight` / `ssim_w` entry in `pseudo.pt` (hook in `src/splats/fit_splat.py`).

**Schedule deviation.** FaithFusion shifts 1 m every 2k iterations for up to 6 m. We use three 2 m steps to fit the shared GPU budget.

### Results (scene_16, 3 lanes; refined splats rendered at the other lanes' real poses, no post-fix)
Source files: `results/faithfusion/summary_ff.{csv,md}`, which merges `eval_scene_16_ff.json` and `ntl_scene_16_ff.json` with the
stage-6/7 rows. The paired test is in `paired_eig_vs_flat.json` and the per-round logs in `prog_ff_round_logs.json`.

| offset | method | PSNR ↑ | LPIPS ↓ | lane-paint F1 ↑ | CLRNet NTL-IoU ↑ | CLRNet lane F1 ↑ |
|---|---|---|---|---|---|---|
| ~3 m | raw splat | 19.63 | 0.493 | 0.473 | 0.177 | 0.178 |
| ~3 m | refine, no fixer (control, 3/6 m schedule) | 19.85 | 0.495 | 0.608 | 0.161 | 0.159 |
| ~3 m | **FF flat** (Difix-ref, w = 0.7) | 20.04 | 0.456 | 0.652 | 0.482 | 0.508 |
| ~3 m | **FF EIG** (Difix-ref, EIG weights) | 20.04 | 0.462 | 0.658 | 0.466 | 0.497 |
| ~3 m | progressive Difix-FT (stage 7, 3/6 m schedule) | 20.28 | 0.445 | 0.787 | 0.657 | 0.896 |
| ~6 m | raw splat | 17.61 | 0.554 | 0.417 | 0.093 | 0.121 |
| ~6 m | refine, no fixer | 17.88 | 0.549 | 0.400 | 0.093 | 0.109 |
| ~6 m | **FF flat** | 17.86 | 0.510 | 0.384 | 0.167 | 0.228 |
| ~6 m | **FF EIG** | 17.89 | 0.512 | 0.391 | 0.180 | 0.242 |
| ~6 m | progressive Difix-FT | 18.21 | 0.495 | 0.471 | 0.461 | 0.575 |

Paired EIG − flat differences (same frames; 95 % bootstrap CI over frames) are all tiny:
- **~3 m:** LPIPS +0.005 [0.005, 0.006], paint F1 +0.006 [0.002, 0.011], NTL-IoU −0.016 [−0.036, 0.001].
- **~6 m:** PSNR +0.03 dB, LPIPS +0.003, NTL-IoU +0.013 [0.002, 0.032].
- **On-path:** unchanged. PSNR is 25.38 for both arms against 25.63 raw, so neither arm damages the recorded path.

Mean EIG weight inside the validity mask rises with offset: about 0.31 at ±2 m, 0.45 at ±4 m and 0.66 at ±6 m, against a flat 0.7.
So the EIG arm trusts Difix less near the path and about as much far from it.

**Verdict.** With Difix-ref as restorer, FaithFusion's EIG weighting is **indistinguishable from the flat weighting of
their Difix path** on real adjacent-lane frames. Both arms clearly beat the no-fixer control on lane agreement (NTL-IoU
0.47–0.48 vs 0.16 at ~3 m). Both are below our progressive loop with the fine-tuned Difix. The restorer matters far more
than the per-pixel weighting.

**Update (final round, all eight roads; RESULTS §19).** The scene_16 verdict above does not hold with more roads: over
eight roads the EIG weighting slightly lowers lane F1 at ~3 m (−0.026 [−0.038, −0.015], lower on 7 of 8 roads) and raises
LPIPS (+0.0035). With all eight roads and five fixers (`results/faithfusion/halluc_auroc.md`), the EIG map's AUROC for
invented paint at ~3 m is 0.67–0.75 (released Difix-ref 0.70); section 3 below is the earlier analysis of scene_16 and
scene_17 only.

Caveats:
- **Different schedule.** The stage-7 rows use a ±3/±6 m schedule, not ±2/±4/±6.
- **One scene.** Only scene_16 was run, because of GPU time.
- **Restorer.** EIG is paired with Difix rather than their EIGent, which also *consumes* the EIG map as a spatial prior; EIGent weights are not released.

## 3. Does the map predict where fixers hallucinate?
`maps.py` and `halluc_auroc.py` work on the test scenes' cross-lane evaluation renders (scene_16 and scene_17; 513 images
at about 3 m and 256 at about 6 m). The fixer outputs are the ones already scored in `results/eval/`.

**Per-pixel labels** (half resolution), measured against the **real** target-lane frame:
- **PAINT** (primary): lane paint in the fixer output that is not within about 10 px (full resolution) of paint in the real image.
  Masks come from `eval.evaluate.marking_mask`. The region is the road (lower 40 %) minus movable objects.
- **CHANGE** (secondary): pixels the fixer changed and got wrong, i.e. ΔE76(fixed, raw) > 10 and ΔE76(fixed, real) > 20
  (CIELAB, 7×7 blur). The region is everything except movable objects and sky.

**Scores** (higher means less trustworthy):
- FaithFusion log-EIG.
- A HAD-style **reprojection-inconsistency** map (our simple version): the fraction of the 3 nearest source-lane training cameras
  in which the pixel's 3-D point is outside the frustum or occluded (by more than 10 % of depth).
- 1 − splat alpha.

**Statistics.** AUROC and AUPRC are pooled over up to 1,500 random region pixels per image, with 95 % bootstrap CIs over images (100 resamples).
Positive rates are low, so AUPRC should be read against the positive rate.

Full table: `results/faithfusion/halluc_auroc.md` and `.json`. Chart: `results/faithfusion/ff_auroc.png`.

AUROC [95% CI]:

| label | fixer | offset | positive rate | EIG | reprojection | 1−alpha |
|---|---|---|---|---|---|---|
| PAINT | Difix | ~3 m | 0.006 | **0.694** [0.669, 0.719] | 0.603 [0.570, 0.627] | 0.309 |
| PAINT | Difix | ~6 m | 0.012 | 0.507 [0.468, 0.549] | 0.517 [0.502, 0.531] | 0.529 |
| PAINT | Difix-FT | ~3 m | 0.004 | **0.757** [0.738, 0.777] | 0.638 [0.617, 0.661] | 0.276 |
| PAINT | Difix-FT | ~6 m | 0.003 | 0.589 [0.544, 0.638] | 0.528 [0.511, 0.543] | 0.464 |
| PAINT | Difix-ref | ~3 m | 0.007 | 0.522 [0.467, 0.572] | 0.439 [0.401, 0.487] | 0.428 |
| PAINT | Difix-ref | ~6 m | 0.024 | 0.439 [0.409, 0.477] | 0.501 [0.484, 0.516] | 0.592 |
| PAINT | our U-Net | ~3 m | 0.002 | 0.668 [0.636, 0.701] | 0.538 [0.495, 0.578] | 0.137 |
| PAINT | our U-Net | ~6 m | 0.003 | 0.523 [0.465, 0.572] | 0.505 [0.475, 0.535] | 0.220 |
| CHANGE | Difix | ~3 m | 0.007 | 0.673 [0.658, 0.690] | 0.700 [0.689, 0.713] | **0.774** |
| CHANGE | Difix | ~6 m | 0.017 | 0.597 [0.578, 0.617] | 0.581 [0.570, 0.592] | **0.701** |
| CHANGE | Difix-FT | ~3 m | 0.033 | 0.654 [0.633, 0.671] | 0.651 [0.641, 0.663] | **0.700** |
| CHANGE | Difix-FT | ~6 m | 0.084 | 0.639 [0.625, 0.654] | 0.611 [0.600, 0.622] | **0.660** |
| CHANGE | Difix-ref | ~3 m | 0.020 | 0.575 [0.557, 0.598] | 0.593 [0.574, 0.615] | **0.693** |
| CHANGE | Difix-ref | ~6 m | 0.060 | 0.561 [0.550, 0.574] | 0.556 [0.548, 0.566] | **0.646** |
| CHANGE | our U-Net | ~3 m | 0.287 | 0.528 [0.515, 0.541] | 0.523 [0.512, 0.535] | **0.643** |
| CHANGE | our U-Net | ~6 m | 0.227 | 0.550 [0.532, 0.566] | 0.546 [0.536, 0.556] | **0.614** |

### Findings
- **EIG does carry signal for hallucinated lane paint at about 3 m.**
  - AUROC is 0.69 (Difix), 0.76 (Difix-FT) and 0.67 (our U-Net), all with CIs above 0.5.
  - It beats both the reprojection map and coverage.
  - Lift over chance is modest: AUPRC is about 1.5–2× the positive rate.
- **At about 6 m the EIG signal fades to near chance** for paint (0.44–0.59). So the map is least informative exactly
  where hallucination matters most.
- **Difix-ref's paint hallucinations are not predicted by EIG** (0.52 at ~3 m, 0.44 at ~6 m). Its hallucinations are
  steered by the reference frame, not by where the splat is uncertain.
- **For the broader "changed and wrong" label, plain coverage (1 − alpha) is the best predictor** for every fixer
  (0.61–0.77). EIG and reprojection are about equal (0.53–0.70). Most gross changes happen where the splat has
  nothing: the horizon, and far or unobserved regions.
- **Our U-Net's CHANGE positive rate is high (0.23–0.29).** It shifts color and exposure globally, which no geometric
  map can predict (AUROC about 0.53).
- **1 − alpha scores < 0.5 for PAINT.** Paint hallucinations sit on well-covered road, where the splat has smeared
  paint to "restore", not in holes.

Caveats:
- **Labels are heuristic** (color-threshold paint masks; ΔE thresholds), and the positive rates are small.
- **The reprojection map is a simple stand-in.** It is not the HAD model [11].
- **The EIG maps come from the splat alone.** They do not depend on the fixer, so one map scores all fixers.

## Re-run
```
scripts/14_faithfusion.sh setup   # clone + build the rasterizer (CPU)
scripts/14_faithfusion.sh maps    # EIG / reprojection / alpha maps for scene_16 + scene_17 (1 GPU job, ~3 min)
scripts/14_faithfusion.sh auroc   # table + JSON (CPU, ~10 min); then: scripts/lib/ff_py.sh -m progressive.faithfusion.figures
scripts/14_faithfusion.sh prog    # flat and EIG arms on scene_16 (1 GPU job per arm, ~17 min each)
scripts/14_faithfusion.sh eval    # image metrics + CLRNet lanes for prog_ff_flat / prog_ff_eig
```
