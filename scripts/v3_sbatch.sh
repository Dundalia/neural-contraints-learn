#!/bin/bash
#SBATCH --job-name=nc-v3
#SBATCH --partition=short-unkillable
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=3:00:00
#SBATCH --output=logs/v3_slurm_%j.out
# 2 draws x 4 arms x 12 repeats = 96 closed-loop sessions, sharded 24 per GPU.
#
# One vLLM process per GPU. The (draw, arm, repeat) mapping is a pure function of the global
# session index, so the shards stay mutually paired without talking to each other: every repeat
# is 8 consecutive sessions holding all 8 conditions on one target sequence.
#
# MOVE BUDGET STAYS AT 36. Raising it would cut the 20% right-censoring of moves-to-acquire, but
# it also redefines the task and breaks comparability with both earlier runs. Instead the primary
# outcome is success per trial, which is uncensored BY CONSTRUCTION (it is exactly the indicator
# of beating the budget) and becomes well resolved at 24 sessions per arm. Moves stay secondary,
# censored at a single known point, which is the easiest case to handle in analysis.
set -u
cd /network/scratch/b/baldelld/MATS/neural-contraints-learn
P=/network/scratch/b/baldelld/MATS/venv-vllm/bin/python
R=artifacts/v2/reads_Qwen3.5-9B.npz
S=artifacts/v2/screen_Qwen3.5-9B.json
OUT=outputs/v3
mkdir -p "$OUT" logs
# keep each process off its neighbours' cores; 4 procs x 6 threads of the 24 requested
export OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false

for k in 0 1 2 3; do
  START=$((24 * k))
  CUDA_VISIBLE_DEVICES=$k $P -u scripts/run.py \
      --reads "$R" --screen "$S" --mode blocks \
      --draws 5 9 --alignments 1.0 0.7 0.4 --control-arm --repeats 12 \
      --session-start "$START" --sessions 24 --batch 8 \
      --trials-baseline 8 --trials-perturb 40 --trials-washout 8 \
      --max-turns 290 --max-model-len 40960 --gpu-frac 0.90 \
      --tag shard$k --out "$OUT" > "logs/v3_shard${k}.log" 2>&1 &
  echo "launched shard $k on GPU $k: sessions ${START}..$((START+23))"
done
wait
echo "===== ALL SHARDS DONE ====="
ls -1 "$OUT"/shard*/sess_*_trials.json 2>/dev/null | wc -l | xargs echo "session files written:"
