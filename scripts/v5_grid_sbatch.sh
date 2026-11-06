#!/bin/bash
#SBATCH --job-name=nc-v5grid
#SBATCH --partition=short-unkillable
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=3:00:00
#SBATCH --output=logs/v5_slurm_%j.out
#
# ROTATION x ALIGNMENT GRID. One job = two target seeds across the WHOLE 24-cell grid.
# Submit with SEED_BLOCK=0,1,2 to cover seeds {0,1}, {2,3}, {4,5}.
#
# WHY SPLIT BY SEED AND NOT BY CELL: a job that dies or hits the 3h wall then costs seeds, not
# cells, so what survives is a complete grid measured less precisely rather than a grid with holes.
# run.py writes session files after each batch, so a wall-clock kill loses at most the batch in
# flight.
#
# LENGTH PARAMS ARE FIXED HERE, NOT SWEPT: 80 perturbation trials and a 48-move budget, the values
# we settled on. Cost is quadratic in turns because prefix caching is off and every turn re-prefills
# the whole conversation, so batch is 4 and max_model_len 81920 (KV = 4 x 81920 x 128KB = 42 GB).
#
# SEVERITY IS RECORDED, NOT GATED (--grid-* path in run.py). The window exists to hold difficulty
# in a band; spanning that band is the entire point of this sweep, and severity has anyway measured
# -0.27 against impairment where effective rotation measures -0.80.
set -u
cd /network/scratch/b/baldelld/MATS/neural-contraints-learn
P=/network/scratch/b/baldelld/MATS/venv-vllm/bin/python
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
TAG=${MODEL##*/}
V=artifacts/v4                      # reuse the corrected calibration and screen
OUT=outputs/v5
SEED_BLOCK=${SEED_BLOCK:-0}
CELLS=24                            # 23 feasible (theta, alignment) pairs + 1 control
PER_JOB=$((2 * CELLS))              # two seeds per job
START=$((SEED_BLOCK * PER_JOB))
STATUS=logs/v5_block${SEED_BLOCK}.status
mkdir -p "$OUT" logs
export TOKENIZERS_PARALLELISM=false
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

DRAW=$($P -c "
import json, numpy as np
sc = json.load(open('$V/screen_${TAG}.json'))
usable = [float(x) for x in sc['usable_levels']]
c = []
for x in sc['rows']:
    ch = x.get('chosen') or {}
    if all(str(l) in ch and ch[str(l)] and ch[str(l)].get('pass') for l in usable):
        c.append((x.get('severity_spread_rel', 9), x.get('mag_spread_rel', 9), x['draw']))
c.sort(); print(c[0][2])")
say "seed block $SEED_BLOCK: sessions ${START}..$((START + PER_JOB - 1)), draw $DRAW"

for k in 0 1 2 3; do
  S=$((START + 12 * k))
  CUDA_VISIBLE_DEVICES=$k OMP_NUM_THREADS=5 $P -u scripts/run.py \
      --reads "$V/reads_${TAG}.npz" --screen "$V/screen_${TAG}.json" --mode blocks \
      --draws "$DRAW" --repeats 6 \
      --grid-thetas 50 60 70 78 84 87 --grid-alignments 1.0 0.7 0.4 0.2 0.1 --grid-control \
      --session-start "$S" --sessions 12 --batch 4 \
      --trials-baseline 8 --trials-perturb 80 --trials-washout 8 --move-budget 48 \
      --max-turns 560 --max-model-len 81920 --gpu-frac 0.90 \
      --tag "b${SEED_BLOCK}s${k}" --out "$OUT" > "logs/v5_b${SEED_BLOCK}_shard${k}.log" 2>&1 &
  say "  launched shard $k on GPU $k: sessions ${S}..$((S + 11))"
done
wait
say "sessions written this block: $(ls -1 $OUT/b${SEED_BLOCK}s*/sess_*_trials.json 2>/dev/null | wc -l) of $PER_JOB"
say "DONE block $SEED_BLOCK"
