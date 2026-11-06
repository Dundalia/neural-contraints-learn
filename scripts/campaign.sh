#!/bin/bash
# Unattended Step 1 -> G3 campaign (docs/DESIGN.md §7.0). Stops at a failed gate rather than
# relaxing a threshold: a failing gate is a methodology decision, not something to work around.
#
#   scripts/campaign.sh <model> <artifacts_dir> <outputs_dir>
set -u
cd /network/scratch/b/baldelld/MATS/neural-contraints-learn
MODEL="${1:-Qwen/Qwen3.5-9B}"
ART="${2:-artifacts}"
OUTD="${3:-outputs}"
TAG="$(basename "$MODEL")"
P=/network/scratch/b/baldelld/MATS/venv-vllm/bin/python
export HF_HOME=/network/scratch/b/baldelld/.cache/huggingface HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
mkdir -p "$ART" "$OUTD"

say() { echo "===== $* ($(date '+%H:%M:%S')) ====="; }
gate() {  # gate <json> <key>
  $P -c "import json,sys;print(json.load(open('$1')).get('$2'))" 2>/dev/null
}

say "STAGE 1/4  G1 reads"
$P -u scripts/calibrate.py --model "$MODEL" --n-fit 1200 --n-val 600 --n-depths 8 \
   --chunk 128 --gpu-frac 0.85 --max-model-len 24000 --out "$ART" 2>&1 \
   | grep --line-buffered -vE "^(Rendering|Processed|Adding|INFO|WARNING)" || true
READS="$ART/reads_${TAG}.npz"
[ -s "$READS" ] || { say "ABORT: no reads artifact"; exit 1; }

say "STAGE 2/4  G1 draws + diagnostics"
$P -u scripts/fit_draws.py --reads "$READS" --draws 20
DJ="$ART/draws_${TAG}.json"
G1=$(gate "$DJ" G1_PASS)
say "G1_PASS=$G1"

say "STAGE 3/4  G2 screen (the kill gate)"
# LAYER is optional. Left empty, screen.py applies the DESIGN 4.1 rule (max R2 among layers
# admitting matched candidates). Set LAYER=<n> to override, which is what you want if the
# operating layer is a decision rather than an outcome. The 2026-08-10 sessions ran at layer 21,
# selected by a within-bin floor that has since been withdrawn (DESIGN 4.5).
$P -u scripts/screen.py --reads "$READS" --draws 20 --scout-draws 3 --n-cand 40 \
   --fit-report "$DJ" ${LAYER:+--layer "$LAYER"}
SJ="$ART/screen_${TAG}.json"
G2=$(gate "$SJ" G2_PASS)
LAYER=$(gate "$SJ" layer)
say "G2_PASS=$G2  layer=$LAYER"

if [ "$G2" != "True" ]; then
  say "STOP: G2 did not pass. Per the design, truncate the alignment grid rather than relax"
  say "the windows -- that is a methodology decision, so stopping for review."
  exit 2
fi

say "STAGE 4/4  G3 baseline closed loop (4 sessions, 20 trials)"
$P -u scripts/run.py --reads "$READS" --screen "$SJ" --model "$MODEL" \
   --mode baseline --sessions 4 --batch 4 --trials 20 --tag g3 --out "$OUTD" 2>&1 \
   | grep --line-buffered -vE "^(Rendering|Processed|Adding|INFO|WARNING)" || true

say "STAGE 4b  G3 summary"
$P -u scripts/analyze.py --dir "$OUTD/g3" --mode baseline || true
say "CAMPAIGN COMPLETE"
