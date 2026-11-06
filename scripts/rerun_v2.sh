#!/bin/bash
# Re-calibration with the model-intent label (DESIGN 4.2b), 2026-08-11.
#
# Everything lands in artifacts/v2 so the 2026-08-10 run stays intact as the record of what the
# forced-letter label produced. Runs to G2 and STOPS: G3/G4 need the severity-window and
# L_session-band decisions that are parked for Davide, and re-running closed-loop sessions before
# those are settled would burn GPU on a metric we already know is broken.
#
# Waits on a PID rather than a pgrep pattern (a previous version matched its own argv forever).
set -u
cd /network/scratch/b/baldelld/MATS/neural-contraints-learn
P=/network/scratch/b/baldelld/MATS/venv-vllm/bin/python
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
TAG=${MODEL##*/}
V2=artifacts/v2
mkdir -p "$V2" logs
STATUS=logs/rerun_v2.status

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

if [ "${WAIT_PID:-}" != "" ]; then
  say "waiting for PID $WAIT_PID to release the GPU"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 15; done
fi

say "STAGE 1/4  calibrate with letter distributions  (GPU, ~25 min)"
$P -u scripts/calibrate.py --model "$MODEL" --n-fit 1200 --n-val 600 --n-depths 8 \
   --out "$V2" > logs/v2_calibrate.log 2>&1
if [ ! -f "$V2/reads_${TAG}.npz" ]; then
  say "FAILED: no reads artifact. tail of log:"; tail -25 logs/v2_calibrate.log | tee -a "$STATUS"
  exit 1
fi
say "calibrate done: $(du -h "$V2/reads_${TAG}.npz" | cut -f1)"

say "STAGE 2/4  fit draws on the intent label  (CPU, ~35 min)"
$P -u scripts/fit_draws.py --reads "$V2/reads_${TAG}.npz" --draws 20 --label intent \
   > logs/v2_fit.log 2>&1
DJ="$V2/draws_${TAG}.json"
if [ ! -f "$DJ" ]; then
  say "FAILED: no draws artifact. tail:"; tail -25 logs/v2_fit.log | tee -a "$STATUS"; exit 1
fi
G1=$($P -c "import json;print(json.load(open('$DJ')).get('G1_PASS'))")
BEATS=$($P -c "import json;print(json.load(open('$DJ'))['gates'].get('beats_prompt_only'))")
say "fit done: G1_PASS=$G1  beats_prompt_only=$BEATS"
$P -c "
import json; r=json.load(open('$DJ')); b=r['prompt_only_baselines']
print('  prompt-only best %.4f (%s)' % (b['best_prompt_only'], b['best_prompt_only_source']))
rows=[x for x in r['rows'] if 'r2' in x]
import statistics as st
for L in sorted({x['layer'] for x in rows}):
    g=[x for x in rows if x['layer']==L]
    print('  layer %3d  R2 %+.3f  gap %+.3f  match %.3f' % (
        L, st.median(x['r2'] for x in g), st.median(x['gap_over_prompt'] for x in g),
        st.median(x.get('match_acc', float('nan')) for x in g)))
" | tee -a "$STATUS"

if [ "$BEATS" != "True" ]; then
  say "STOPPING: the best layer does not beat the prompt-only baseline even with the intent label."
  say "That is a real result, not a bug. G2 is pointless until it is resolved. See DESIGN 4.2b."
  exit 2
fi

say "STAGE 3/4  G2 perturbation screen at the rule's layer  (CPU, ~20 min)"
$P -u scripts/screen.py --reads "$V2/reads_${TAG}.npz" --draws 20 --scout-draws 3 --n-cand 40 \
   --fit-report "$DJ" > logs/v2_screen.log 2>&1
SJ="$V2/screen_${TAG}.json"
if [ ! -f "$SJ" ]; then
  say "FAILED: no screen artifact. tail:"; tail -25 logs/v2_screen.log | tee -a "$STATUS"; exit 1
fi
$P -c "
import json; r=json.load(open('$SJ'))
print('  layer %s | G2_PASS %s | usable %s | matched %s/%s' % (
  r['layer'], r['G2_PASS'], r['usable_levels'], r['n_draws_severity_matched'], r['draws']))
print('  rule: %s' % r['layer_selection_rule'])" | tee -a "$STATUS"

say "STAGE 4/4  figures"
$P -u scripts/figures.py > logs/v2_figures.log 2>&1 || {
  say "figures failed:"; tail -20 logs/v2_figures.log | tee -a "$STATUS"; }
say "DONE. G3/G4 deliberately not run: parked decisions (severity window, L_session band)."
