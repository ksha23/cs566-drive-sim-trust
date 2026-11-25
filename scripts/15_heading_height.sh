#!/bin/bash
# Stage 15: trust vs. heading (yaw) and camera height on the Para-Lane TEST scenes (consistency + pseudo-ground truth;
# see src/eval/heading.py and src/eval/heading_geom.py for the conventions and the warps).
#   render                 $GJ-wrapped: yaw +-5/10/15 deg and dz +-0.5/1 m on-path, lateral + yaw +-10 deg (both scenes)
#   fix <scene> <method>   one fixer on one scene's deviated renders (one GPU job, resumable: --skip_existing)
#   fixall                 every (scene, fixer) as its own GPU job
#   lanes <scene>          CLRNet on every deviated + 0-deviation output of every method (GPU job)
#   eval <scene>           per-image metrics (GPU job; one rows file per method = resume unit)
#   summary                bootstrap tables, sanity checks (results/heading/checks.json), figures + panels (CPU)
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
HR=$DATA/heading; METHODS="difix difix_ref nv_fixer difix_ft difix_reimpl_ref ours_hfgan_s0"
link_zero() {  # 0-deviation outputs = the existing lateral-eval outputs (symlinked, never recomputed)
  s=$1; m=$2
  for d in $DATA/fixed/$m/$s/render/*; do
    t=$HR/fixed/$m/$s/render/$(basename $d)/y+0_z+0.0; mkdir -p $(dirname $t); [ -e $t ] || ln -s $d/full $t
  done
}
fix() {
  s=$1; m=$2; link_zero $s $m
  A="$HR $s $HR/fixed --meta $HR/$s/meta.json --skip_existing"
  case $m in
    difix|difix_ref|nv_fixer) $PY src/fixers/apply.py $A --methods $m ;;
    difix_ft) $PY src/fixers/apply.py $A --methods difix_ft --ckpt $DATA/difix_ft/best_unet.pt ;;
    ours_hfgan_s0) $PY src/fixers/apply.py $A --methods ours --ckpt $DATA/fixers/hfgan_s0/best.pt --tag ours_hfgan_s0 ;;
    difix_reimpl_ref)  # our Difix reimplementation runs in its venv (docs/IMPLEMENTATION_NOTES_difix.md)
      HF_HOME=$SMOKE/hf TORCH_HOME=$SMOKE/cache/torch XDG_CACHE_HOME=$SMOKE/cache PYTHONPATH=$REPO/src \
        $DATA/difix_reimpl/venv/bin/python src/fixers/apply.py $A --methods difix_reimpl \
        --ckpt $DATA/difix_reimpl/ours_matched/best.pt --tag difix_reimpl_ref ;;
  esac
}
lanes() {
  s=$1; L=$HR/lanes/$s; mkdir -p $L
  export XDG_CACHE_HOME=$SMOKE/cache TORCH_HOME=$SMOKE/cache/torch
  $PY -c "import json; M=json.load(open('$HR/$s/meta.json')); print('\n'.join(sorted({m['path'] for m in M} | {m['zero'] for m in M})))" > $L/paths.txt
  for m in raw $METHODS; do
    R=$HR; [ $m != raw ] && R=$HR/fixed/$m
    [ -f $L/$m.json ] && continue
    sed "s|^|$R/|" $L/paths.txt > $L/list_$m.txt
    $DATA/lanes/env/bin/python src/eval/lanes/clrnet_infer.py --list $L/list_$m.txt --root $R --out $L/$m.json
  done
}
case $1 in
  render) $GJ bash -c "for s in scene_16 scene_17; do $PY src/splats/render_deviation.py \$s; done" > logs/heading_render.log 2>&1 ;;
  fix) fix $2 $3 ;;
  fixall) for s in scene_16 scene_17; do for m in $METHODS; do
            $GJ bash scripts/15_heading_height.sh fix $s $m > logs/heading_fix_${s}_$m.log 2>&1; done; done ;;
  lanes) lanes $2 ;;
  eval) $PY src/eval/heading.py eval $2 --methods raw,${METHODS// /,} ;;
  summary)
    CUDA_VISIBLE_DEVICES= $PY src/eval/heading.py summary > /dev/null
    CUDA_VISIBLE_DEVICES= $PY src/eval/heading_checks.py > /dev/null
    CUDA_VISIBLE_DEVICES= $PY src/eval/heading_figures.py ;;
esac
