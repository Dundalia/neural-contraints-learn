#!/bin/bash
#SBATCH --job-name=nc-v7
#SBATCH --partition=short-unkillable
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=3:00:00
#SBATCH --output=logs/v7_%j.out
#
# RE-RUN ON THE CORRECTED DECODER (2026-08-18).
#
# WHY THIS EXISTS. The 2026-08-11 label change (DESIGN 7.2c) made the readout fit the model's own
# intent distribution and made the G1 gate test that. It landed in nc/labels.py, fit_draws.py and the
# gate, but NOT in scripts/run.py or scripts/screen.py, so every session in v5, v6q, v6g_d9 and
# v6g_d17 -- 570 sessions, ~73 h of wall time -- ran a decoder fit to the superseded forced-letter
# label, which scores BELOW the prompt-only baseline on its own label (-0.124 for the 9B, -0.218 for
# the 31B). The gate certified an object the loop never used. Both call sites now route through
# nc.labels.readout_label and tests/test_live_label.py fails if they diverge again.
#
# QWEN FIRST, ON PURPOSE. The 9B is ~3.5x cheaper than the 31B, and it answers the question that
# decides whether the 31B is worth two more days: does the model ADAPT once the decoder is one it can
# measurably steer? The intent-fit readout beats the best prompt-only predictor by +0.099 here, so
# for the first time the model has leverage over the thing being perturbed. The previous null on
# adaptation was close to guaranteed by the bug and is not evidence about the model.
#
# MACHINERY IS UNCHANGED and all of it is verified in production: an afterany successor queued up
# front so a wall kill cannot break the chain; per-trial checkpoints so an interrupted session resumes
# where it stopped; a watchdog for the vLLM livelock; the wall guard off because checkpointing made it
# pure waste; both populations advancing together, two GPUs each, so the population question is
# answered at equal seed counts rather than one population finishing first.
set -u
cd /network/scratch/b/baldelld/MATS/neural-contraints-learn
P=/network/scratch/b/baldelld/MATS/venv-vllm/bin/python
CHAIN=${CHAIN:-1}
CHAIN_MAX=${CHAIN_MAX:-40}
STATUS=logs/v7_campaign.status
mkdir -p logs
export TOKENIZERS_PARALLELISM=false
say() { echo "[$(date '+%m-%d %H:%M:%S')] chain $CHAIN: $*" | tee -a "$STATUS"; }

MODEL=Qwen/Qwen3.5-9B ; TAG=Qwen3.5-9B ; V=artifacts/v7q
SCREEN="$V/screen_${TAG}.json"
# TOTAL is DERIVED from the screen, never assumed. The corrected decoder dropped alignment 0.1,
# so the triangular grid is 21 feasible cells plus one control, not the 24 the old runs used, and a
# hardcoded 144 would have made the chain hunt forever for sessions the grid does not contain.
# alignments and thetas come from the SCREEN, not from this file: the corrected decoder may not
# admit every level, and the rule is to truncate the grid rather than relax a window
LEVELS=$($P -c "
import json; sc=json.load(open('$SCREEN')); print(' '.join(str(x) for x in sc['usable_levels']))")
THETAS="50 60 70 78 84 87"
LEN_ARGS="--trials-baseline 8 --trials-perturb 80 --trials-washout 8 --move-budget 48 --max-turns 560"
GRID_ARGS="--grid-thetas $THETAS --grid-alignments $LEVELS --grid-control"
REPEATS=${REPEATS:-6}
TOTAL=$($P -c "
import sys, math; sys.path.insert(0,'.')
import nc.manifold as mf
levels=[float(x) for x in '$LEVELS'.split()]
n=0
for al in levels:
    adm={round(math.degrees(x),3) for x in mf.admissible_thetas(al)}
    n+=sum(1 for th in [$(echo $THETAS | tr ' ' ',')] if round(float(th),3) in adm)
print((n+1)*$REPEATS)")

DRAWS=$($P -c "
import json
sc=json.load(open('$SCREEN')); usable=[float(x) for x in sc['usable_levels']]
c=[]
for x in sc['rows']:
    ch=x.get('chosen') or {}
    if all(str(l) in ch and ch[str(l)] and ch[str(l)].get('pass') for l in usable):
        c.append((x.get('severity_spread_rel',9), x.get('mag_spread_rel',9), x['draw']))
c.sort(); print(' '.join(str(d) for _,_,d in c[:2]))")
set -- $DRAWS ; D1=${1:-} ; D2=${2:-}
if [ -z "$D1" ]; then say "no draw is severity-matched on every usable level; STOP"; exit 2; fi
OUT1=outputs/v7q_d$D1 ; OUT2=outputs/v7q_d$D2

missing_list() {
  $P - "$1" "$TOTAL" <<'PYEOF2'
import sys
from pathlib import Path
base, total = Path(sys.argv[1]), int(sys.argv[2])
def done(i):
    for pat in (f"sess_{i:03d}_trials.json", f"sess_{i:03d}_skipped.json"):
        if (base / pat).exists() or any(base.glob(f"*/{pat}")):
            return True
    return False
print(" ".join(str(i) for i in range(total) if not done(i)))
PYEOF2
}
progress_count() { ls -1 outputs/v7q_d*/*/sess_*_trials.json outputs/v7q_d*/*/sess_*_skipped.json \
  2>/dev/null | wc -l; }
shard_of() { local k=$1 n=$2; shift 2; local i=0 out=""; for x in "$@"; do
  [ $((i % n)) -eq "$k" ] && out="$out $x"; i=$((i+1)); done; echo "$out"; }

SHARD_PID=(); SHARD_LOG=()
run_sweep() {   # draw outdir gpucsv
  local DRAW="$1" OUT="$2" GPUS="$3"
  local MISS; MISS=($(missing_list "$OUT")); local N=${#MISS[@]}
  if [ "$N" -eq 0 ]; then say "  $OUT: complete"; return 0; fi
  IFS=',' read -r -a GA <<< "$GPUS"; local NG=${#GA[@]}
  say "  $OUT: $N of $TOTAL missing across $NG GPU(s) [$GPUS]"
  local k=0
  for g in "${GA[@]}"; do
    local LIST; LIST=$(shard_of "$k" "$NG" "${MISS[@]}"); k=$((k+1))
    [ -z "$LIST" ] && continue
    local LG="logs/v7_c${CHAIN}_$(basename "$OUT")_gpu${g}.log"
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=5 $P -u scripts/run.py \
        --model "$MODEL" --reads "$V/reads_${TAG}.npz" --screen "$SCREEN" --mode blocks \
        --draws "$DRAW" --repeats $REPEATS $GRID_ARGS $LEN_ARGS \
        --session-list $LIST --skip-existing --wall-limit-min 999 \
        --batch 4 --max-model-len 32768 --gpu-frac 0.90 \
        --tag "c${CHAIN}g${g}" --out "$OUT" > "$LG" 2>&1 &
    SHARD_PID+=("$!"); SHARD_LOG+=("$LG")
  done
}
kill_tree() { local p=$1 c; for c in $(pgrep -P "$p" 2>/dev/null); do kill_tree "$c"; done
  kill -9 "$p" 2>/dev/null; }
watchdog() { local stall=$1 i p f age any
  while true; do sleep 120; any=0
    for i in "${!SHARD_PID[@]}"; do p=${SHARD_PID[$i]}; f=${SHARD_LOG[$i]}
      kill -0 "$p" 2>/dev/null || continue; any=1
      age=$(( ($(date +%s) - $(stat -c %Y "$f" 2>/dev/null || echo 0)) / 60 ))
      [ "$age" -ge "$stall" ] && { say "WATCHDOG: $f silent ${age}min; killing $p"; kill_tree "$p"; }
    done; [ "$any" -eq 0 ] && return; done; }
start_watchdog() { [ ${#SHARD_PID[@]} -eq 0 ] && return; watchdog "${1:-30}" & WD=$!; }
stop_watchdog() { [ -n "${WD:-}" ] && kill -9 "$WD" 2>/dev/null; WD=""; }

PROG0=$(progress_count)
N1=($(missing_list "$OUT1")); N1=${#N1[@]}
N2=0; [ -n "$D2" ] && { n=($(missing_list "$OUT2")); N2=${#n[@]}; }
say "corrected-decoder re-run | layer $($P -c "import json;print(json.load(open('$SCREEN'))['layer'])") | levels [$LEVELS] | populations [$DRAWS]"
say "state: pop $D1 $((TOTAL-N1))/$TOTAL, pop $D2 $((TOTAL-N2))/$TOTAL"

NEXT=""
if [ $((N1+N2)) -gt 0 ] && [ "$CHAIN" -lt "$CHAIN_MAX" ]; then
  NEXT=$(sbatch --parsable --dependency=afterany:"${SLURM_JOB_ID:-0}" \
         --export=ALL,CHAIN=$((CHAIN+1)),CHAIN_MAX=$CHAIN_MAX scripts/v7_campaign.sh 2>/dev/null)
  say "queued chain $((CHAIN+1)) as job ${NEXT:-FAILED} (afterany, survives a wall kill)"
fi

if [ "$N1" -gt 0 ] && [ "$N2" -gt 0 ]; then
  say "both populations, two GPUs each"
  run_sweep "$D1" "$OUT1" 0,1 ; run_sweep "$D2" "$OUT2" 2,3
elif [ "$N1" -gt 0 ]; then run_sweep "$D1" "$OUT1" 0,1,2,3
elif [ "$N2" -gt 0 ]; then run_sweep "$D2" "$OUT2" 0,1,2,3
fi
start_watchdog 30; wait; stop_watchdog

M1=($(missing_list "$OUT1")); M1=${#M1[@]}
M2=0; [ -n "$D2" ] && { n=($(missing_list "$OUT2")); M2=${#n[@]}; }
PROG1=$(progress_count)
say "written this slot: $((PROG1-PROG0)) | remaining: pop $D1 $M1, pop $D2 $M2"
$P scripts/compare_draws.py 2>&1 | tail -20 | tee -a "$STATUS" || say "compare_draws failed"

if [ $((M1+M2)) -eq 0 ]; then
  say "QWEN RE-RUN COMPLETE"
  [ -n "$NEXT" ] && { scancel "$NEXT" 2>/dev/null; say "cancelled successor $NEXT"; }
  for d in "$OUT1" "$OUT2"; do say "--- $d ---"
    $P scripts/analyze_grid.py --dir "$d" 2>&1 | tail -18 | tee -a "$STATUS"; done
  say "DONE"
elif [ "$PROG1" -le "$PROG0" ]; then
  [ -n "$NEXT" ] && scancel "$NEXT" 2>/dev/null
  say "STOPPING: zero sessions written this slot; check logs/v7_c${CHAIN}_*.log"
elif [ -n "$NEXT" ]; then say "chain $((CHAIN+1)) (job $NEXT) will continue"
else say "resubmitting from the tail"
  sbatch --export=ALL,CHAIN=$((CHAIN+1)),CHAIN_MAX=$CHAIN_MAX scripts/v7_campaign.sh; fi
