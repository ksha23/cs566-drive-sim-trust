#!/bin/bash
# Stage 21: our own (hybrid) Gaussian-splat renderer, src/splats/own_render.py; validation vs gsplat (docs/own_renderer.md).
# usage (each step is one GPU job, <= ~1 h):
#   $GJ scripts/21_own_renderer.sh check              # own vs gsplat vs pure-PyTorch reference on real splats
#   $GJ scripts/21_own_renderer.sh shortfit           # 3k-iteration fits (densification on) with both renderers
#   $GJ scripts/21_own_renderer.sh refit              # full 12k refit of scene_16/clip_0 with our renderer
#                                                     #   (+ a gsplat refit with seed 1 = run-to-run noise reference)
#   $GJ scripts/21_own_renderer.sh render             # render them at the other lanes' real poses
#   $GJ scripts/21_own_renderer.sh eval               # existing evaluation (evaluate.py + CLRNet lanes) on those renders
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
O=$DATA/own_renderer; S=scene_16; C=clip_0; mkdir -p $O results/own_renderer
case $1 in
check)
  $PY src/splats/own_render_check.py $O/check --scenes scene_16,scene_17 --per_bin 4 > logs/own_check.log 2>&1
  cp $O/check/check.json $O/check/summary.json results/own_renderer/ ;;
shortfit)
  for r in gsplat own; do
    $PY src/splats/fit_splat.py $S $C $O/shortfit/$r --iters 3000 --renderer $r > logs/own_shortfit_$r.log 2>&1
    grep "^it \|TRAIN\|ON-PATH" logs/own_shortfit_$r.log
  done
  $PY src/splats/own_render_summary.py --fitlogs results/own_renderer/shortfit.json logs/own_shortfit_gsplat.log logs/own_shortfit_own.log ;;
refit)
  [ -f $O/splats/$S/$C/splat.pt ] || $PY src/splats/fit_splat.py $S $C $O/splats/$S/$C --iters 12000 --renderer own > logs/own_refit_${S}_$C.log 2>&1
  [ -f $O/splats_gsplat_s1/$S/$C/splat.pt ] || $PY src/splats/fit_splat.py $S $C $O/splats_gsplat_s1/$S/$C --iters 12000 --seed 1 > logs/own_refit_gsplat_s1_${S}_$C.log 2>&1
  grep -h "TRAIN\|ON-PATH" logs/own_refit_${S}_$C.log logs/own_refit_gsplat_s1_${S}_$C.log
  $PY src/splats/own_render_summary.py --fitlogs results/own_renderer/refit_fits.json logs/fit_${S}_$C.log logs/own_refit_${S}_$C.log logs/own_refit_gsplat_s1_${S}_$C.log ;;
render)
  $PY src/splats/own_render_eval.py $S $O --splat own_refit=$O/splats/$S/$C/splat.pt:own \
      --splat gsplat_refit_s1=$O/splats_gsplat_s1/$S/$C/splat.pt:gsplat \
      --splat own_render_orig=$DATA/splats/$S/$C/splat.pt:own > logs/own_render_eval_$S.log 2>&1
  cp $O/render_timing_$S.json results/own_renderer/ ;;
eval)
  M=raw,own_refit,gsplat_refit_s1,own_render_orig
  $PY src/eval/evaluate.py $O/pairs $S $O/fixed results/own_renderer/eval_$S.json --methods $M --stride 1 > logs/own_eval_$S.log 2>&1
  # CLRNet lanes on the same subset (own output dir: the shared $DATA/lanes/out/<scene> cache is not touched)
  L=$O/lanes/$S; mkdir -p $L; LP=$DATA/lanes/env/bin/python
  export XDG_CACHE_HOME=$SMOKE/cache TORCH_HOME=$SMOKE/cache/torch
  $PY -c "import json,sys; it=json.load(open('$O/pairs/$S/meta.json')); open('$L/real.txt','w').write('\n'.join(sorted({'$O/pairs/'+m['real'] for m in it}))); open('$L/renders.txt','w').write('\n'.join(m['path'] for m in it))"
  $LP src/eval/lanes/clrnet_infer.py --list $L/real.txt --root $O/pairs --out $L/real.json >> logs/own_eval_$S.log 2>&1
  ARGS=""
  for m in ${M//,/ }; do
    if [ $m = raw ]; then R=$O/pairs; else R=$O/fixed/$m; fi
    sed "s|^|$R/|" $L/renders.txt > $L/list_$m.txt
    $LP src/eval/lanes/clrnet_infer.py --list $L/list_$m.txt --root $R --out $L/$m.json >> logs/own_eval_$S.log 2>&1
    ARGS="$ARGS --method $m=$L/$m.json"
  done
  $PY src/eval/lanes/ntl_iou.py --pairs $O/pairs --scene $S --real $L/real.json $ARGS --out results/own_renderer/ntl_$S.json --stride 1 | tee -a logs/own_eval_$S.log
  $PY src/splats/own_render_summary.py $S > results/own_renderer/summary_$S.md ;;
*) echo "usage: $0 check|shortfit|refit|render|eval"; exit 1 ;;
esac
