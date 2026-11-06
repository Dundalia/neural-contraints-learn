#!/bin/bash
# Closed-loop stages on the v2 (intent-label) decoder: G3 baseline, G4 matched impairment, and
# the no-perturbation control. Queued after rerun_v2.sh.
#
# PARAMETERS ARE UNCHANGED FROM THE FROZEN DESIGN. The severity window, the 36-move budget and the
# block lengths are the ones already in the doc. Those are exactly the knobs that made G4's metric
# unusable on 2026-08-10 (severity floors the outcome, so moves-to-acquire is censored), and
# changing them is a methodology decision that is Davide's, not mine. So this runs as designed and
# the flooring may well recur. That is a finding to report, not a bug to work around.
#
# Cheap: the 2026-08-10 G4 batch of 8 lockstep sessions x 56 trials took ~4.5 min of wall clock.
set -u
cd /network/scratch/b/baldelld/MATS/neural-contraints-learn
P=/network/scratch/b/baldelld/MATS/venv-vllm/bin/python
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
TAG=${MODEL##*/}
V2=artifacts/v2
OUT=outputs/v2
STATUS=logs/rerun_v2.status
mkdir -p "$OUT"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

for PID in "$@"; do while kill -0 "$PID" 2>/dev/null; do sleep 15; done; done

R="$V2/reads_${TAG}.npz"; S="$V2/screen_${TAG}.json"
if [ ! -f "$S" ]; then
  say "CLOSED LOOP SKIPPED: no $S, so the upstream pipeline stopped before G2. Nothing to run."
  exit 0
fi
LAYER=$($P -c "import json;print(json.load(open('$S'))['layer'])")
say "CLOSED LOOP on the intent-label decoder, layer $LAYER"

say "G3  baseline closed loop, 4 sessions x 20 trials"
$P -u scripts/run.py --reads "$R" --screen "$S" --mode baseline --sessions 4 --trials 20 \
   --tag g3 --out "$OUT" > logs/v2_g3.log 2>&1
$P scripts/analyze.py --dir "$OUT/g3" --mode baseline 2>&1 | tail -8 | tee -a "$STATUS"

say "G4  matched impairment, 8 sessions (4 per alignment arm)"
$P -u scripts/run.py --reads "$R" --screen "$S" --mode blocks --sessions 8 \
   --tag g4 --out "$OUT" > logs/v2_g4.log 2>&1
$P scripts/analyze.py --dir "$OUT/g4" --mode blocks --bin 4 2>&1 | tail -12 | tee -a "$STATUS"

say "CONTROL  no perturbation, same block structure, 6 sessions"
$P -u scripts/run.py --reads "$R" --screen "$S" --mode blocks --sessions 6 --sham \
   --tag sham --out "$OUT" > logs/v2_sham.log 2>&1
$P scripts/analyze.py --dir "$OUT/sham" --mode blocks --bin 4 2>&1 | tail -12 | tee -a "$STATUS"

say "figures"
$P -u scripts/figures.py > logs/v2_figures2.log 2>&1 || {
  say "figures failed:"; tail -20 logs/v2_figures2.log | tee -a "$STATUS"; }
say "CLOSED LOOP DONE"
