#!/bin/bash
# Stage 18: more Para-Lane TEST scenes (diversity of the lateral results; docs/more_scenes.md).
# Every stage is resumable and, except summary/screen, one GPU job per (stage, scene) through $GJ (each < 60 min).
#   screen <scene...>   CPU: length, curvature, lane gaps per scene -> results/eval_more/screening.json
#   fit <scene>         3 lane splats + eval pairs (stage 2, 'eval' role)
#   fix <scene>         difix, difix_ref, nv_fixer, difix_ft, difix_reimpl_ref, ours_hfgan_s0 on the eval renders
#   eval <scene>        image / hallucination / YOLO metrics + CLRNet lanes for raw + the fixers
#   prog <scene>        progressive update with our Difix-ref reimplementation (as stage 7d) + its metrics
#   road <scene>        road-surface-prior refit of the 3 lanes (as stage 8) + its metrics
#   summary             per-scene and pooled bootstrap tables, scene-level deltas (CPU)
# Rows: results/eval_more/eval_<scene>_{fix,prog,road}.json and ntl_<scene>_{fix,prog,road}.json.
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
R=results/eval_more; mkdir -p $R
FIXERS="raw difix difix_ref nv_fixer difix_ft difix_reimpl_ref ours_hfgan_s0"
CK_DX=$DATA/difix_reimpl/ours_matched/best.pt
st=$1; s=$2
REF=difix_ref,   # released Difix-ref: skipped on scene_10, scene_3, scene_4, scene_11 (GPU time; not one of the key comparisons), see docs/more_scenes.md; added later by stage 'ref'
case $s in scene_10|scene_3|scene_4|scene_11) SKIPREF=1;; esac
if [ -n "$SKIPREF" ]; then FIXERS="raw difix nv_fixer difix_ft difix_reimpl_ref ours_hfgan_s0"; REF=""; fi
case $st in
  screen) shift; $PY src/eval/screen_scenes.py $R/screening.json "$@" ;;
  fit) $GJ bash scripts/02_scene.sh $s eval > logs/more_fit_$s.log 2>&1 ;;
  fix)
    A="$DATA/pairs $s $DATA/fixed --skip_existing"
    $GJ bash -c "
      $PY src/fixers/apply.py $A --methods difix,${REF}nv_fixer
      $PY src/fixers/apply.py $A --methods difix_ft --ckpt $DATA/difix_ft/best_unet.pt
      $PY src/fixers/apply.py $A --methods ours --ckpt $DATA/fixers/hfgan_s0/best.pt --tag ours_hfgan_s0
      HF_HOME=$SMOKE/hf TORCH_HOME=$SMOKE/cache/torch XDG_CACHE_HOME=$SMOKE/cache PYTHONPATH=$REPO/src \
        $DATA/difix_reimpl/venv/bin/python src/fixers/apply.py $A --methods difix_reimpl --ckpt $CK_DX --tag difix_reimpl_ref
    " > logs/more_fix_$s.log 2>&1 ;;
  eval)
    $GJ bash -c "
      $PY src/eval/evaluate.py $DATA/pairs $s $DATA/fixed $R/eval_${s}_fix.json --methods $(echo $FIXERS | tr ' ' ',')
      scripts/lib/lanes_scene.sh $s $DATA/fixed $R/ntl_${s}_fix.json $FIXERS
    " > logs/more_eval_$s.log 2>&1 ;;
  ref)  # released Difix-ref on a road where 'fix' skipped it; rows in separate files, so nothing else is recomputed
    $GJ bash -c "set -e
      $PY src/fixers/apply.py $DATA/pairs $s $DATA/fixed --skip_existing --methods difix_ref
      $PY src/eval/evaluate.py $DATA/pairs $s $DATA/fixed $R/eval_${s}_ref.json --methods difix_ref
      scripts/lib/lanes_scene.sh $s $DATA/fixed $R/ntl_${s}_ref.json difix_ref" > logs/more_ref_$s.log 2>&1 ;;
  prog)
    $GJ bash -c "for c in clip_0 clip_1 clip_2; do
        [ -f $DATA/splats/$s/\$c/prog_difix_reimpl/log.json ] || $PY src/progressive/progressive.py $s \$c difix_reimpl $CK_DX; done
      $PY src/eval/evaluate.py $DATA/pairs $s $DATA/fixed $R/eval_${s}_prog.json --methods prog_difix_reimpl
      scripts/lib/lanes_scene.sh $s $DATA/fixed $R/ntl_${s}_prog.json prog_difix_reimpl" > logs/more_prog_$s.log 2>&1 ;;
  road)
    $GJ bash -c "for c in clip_0 clip_1 clip_2; do d=$DATA/splats/$s/\$c
        [ -f \$d/splat_road.pt ] || $PY src/splats/fit_splat.py $s \$c \$d --iters 12000 --name splat_road --road_prior 0.1 > logs/road_${s}_\$c.log 2>&1
        $PY src/splats/render_method.py $s \$c \$d/splat_road.pt road_prior; done
      $PY src/eval/evaluate.py $DATA/pairs $s $DATA/fixed $R/eval_${s}_road.json --methods road_prior
      scripts/lib/lanes_scene.sh $s $DATA/fixed $R/ntl_${s}_road.json road_prior" > logs/more_road_$s.log 2>&1 ;;
  summary)
    CUDA_VISIBLE_DEVICES= $PY src/eval/more_scenes_summary.py ;;
esac
