#!/bin/bash
#SBATCH --job-name=nc-v5chain
#SBATCH --partition=short-unkillable
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=3:00:00
#SBATCH --output=logs/v5_chain_%j.out
#
# SELF-CHAINING, RESUMABLE grid sweep. Designed to be launched once and left overnight.
#
# WHY A CHAIN AND NOT PARALLEL JOBS: only one short-unkillable job runs at a time (QOS caps GPUs
# per user), so three parallel jobs bought failure tolerance but no wall-clock saving. This runs
# whatever is still missing, stops cleanly before the 3h wall, and resubmits itself.
#
# HOW IT IS SAFE TO REPEAT:
#   * run.py --skip-existing treats a session as done if its trials file OR a skip marker exists
#     anywhere in the sweep, so a job never redoes finished work.
#   * --deadline-min stops STARTING new batches near the wall. A batch that starts always finishes
#     and writes, so a wall-clock kill costs nothing.
#   * infeasible cells (alignment 0.1 admits no perturbation) write a skip marker, so the chain
#     cannot loop forever retrying them.
#   * if a job makes ZERO progress the chain stops rather than spinning.
set -u
cd /network/scratch/b/baldelld/MATS/neural-contraints-learn
P=/network/scratch/b/baldelld/MATS/venv-vllm/bin/python
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
TAG=${MODEL##*/}
V=artifacts/v4
OUT=outputs/v5
TOTAL=144
CHAIN=${CHAIN:-1}
CHAIN_MAX=${CHAIN_MAX:-8}
DEADLINE=${DEADLINE:-95}          # minutes; a batch runs 55-75 min, wall is 180, load is ~6
STATUS=logs/v5_chain.status
mkdir -p "$OUT" logs
export TOKENIZERS_PARALLELISM=false
say() { echo "[$(date +%H:%M:%S)] chain $CHAIN: $*" | tee -a "$STATUS"; }

missing_list() {
  $P - <<PYEOF
from pathlib import Path
base = Path("$OUT")
def done(i):
    for pat in (f"sess_{i:03d}_trials.json", f"sess_{i:03d}_skipped.json"):
        if (base / pat).exists() or any(base.glob(f"*/{pat}")):
            return True
    return False
print(" ".join(str(i) for i in range($TOTAL) if not done(i)))
PYEOF
}

MISSING=($(missing_list))
BEFORE=${#MISSING[@]}
say "$BEFORE of $TOTAL sessions still missing"
if [ "$BEFORE" -eq 0 ]; then
  say "sweep COMPLETE; running analysis and figures"
  $P scripts/analyze_grid.py --dir "$OUT" 2>&1 | tail -40 | tee -a "$STATUS"
  $P scripts/figures.py > logs/v5_figures.log 2>&1 || say "figures failed"
  $P scripts/plot_trajectories.py --dir "$OUT" --arm 1.0 > /dev/null 2>&1 || true
  say "DONE"
  exit 0
fi

DRAW=$($P -c "
import json
sc = json.load(open('$V/screen_${TAG}.json'))
usable = [float(x) for x in sc['usable_levels']]
c = []
for x in sc['rows']:
    ch = x.get('chosen') or {}
    if all(str(l) in ch and ch[str(l)] and ch[str(l)].get('pass') for l in usable):
        c.append((x.get('severity_spread_rel', 9), x.get('mag_spread_rel', 9), x['draw']))
c.sort(); print(c[0][2])")

# deal the missing sessions round-robin across the 4 GPUs so each shard gets a comparable mix of
# easy and hard cells; a contiguous split would put all the hard cells on one GPU
for k in 0 1 2 3; do
  LIST=""
  for idx in "${!MISSING[@]}"; do
    if [ $((idx % 4)) -eq $k ]; then LIST="$LIST ${MISSING[$idx]}"; fi
  done
  if [ -z "$LIST" ]; then say "  shard $k: nothing to do"; continue; fi
  say "  shard $k: $(echo $LIST | wc -w) sessions"
  CUDA_VISIBLE_DEVICES=$k OMP_NUM_THREADS=5 $P -u scripts/run.py \
      --reads "$V/reads_${TAG}.npz" --screen "$V/screen_${TAG}.json" --mode blocks \
      --draws "$DRAW" --repeats 6 \
      --grid-thetas 50 60 70 78 84 87 --grid-alignments 1.0 0.7 0.4 0.2 0.1 --grid-control \
      --session-list $LIST --skip-existing --deadline-min "$DEADLINE" --batch 4 \
      --trials-baseline 8 --trials-perturb 80 --trials-washout 8 --move-budget 48 \
      --max-turns 560 --max-model-len 81920 --gpu-frac 0.90 \
      --tag "c${CHAIN}s${k}" --out "$OUT" > "logs/v5_chain${CHAIN}_shard${k}.log" 2>&1 &
done
wait

AFTER=($(missing_list)); AFTER=${#AFTER[@]}
say "progress: $BEFORE -> $AFTER missing"
if [ "$AFTER" -eq 0 ]; then
  say "sweep COMPLETE; running analysis and figures"
  $P scripts/analyze_grid.py --dir "$OUT" 2>&1 | tail -40 | tee -a "$STATUS"
  $P scripts/figures.py > logs/v5_figures.log 2>&1 || say "figures failed"
  $P scripts/plot_trajectories.py --dir "$OUT" --arm 1.0 > /dev/null 2>&1 || true
  say "DONE"
elif [ "$AFTER" -ge "$BEFORE" ]; then
  say "STOPPING THE CHAIN: no progress this job ($BEFORE -> $AFTER). Something is wrong; look at"
  say "  logs/v5_chain${CHAIN}_shard*.log before resubmitting."
elif [ "$CHAIN" -ge "$CHAIN_MAX" ]; then
  say "chain cap $CHAIN_MAX reached with $AFTER still missing; resubmit by hand if wanted"
else
  NEXT=$((CHAIN + 1))
  say "queuing chain $NEXT for the remaining $AFTER"
  sbatch --export=ALL,CHAIN=$NEXT,CHAIN_MAX=$CHAIN_MAX,DEADLINE=$DEADLINE scripts/v5_chain.sh
fi
