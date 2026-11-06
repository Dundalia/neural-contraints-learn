#!/bin/bash
#SBATCH --job-name=nc-bench
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=46G
#SBATCH --time=1:30:00
#SBATCH --output=logs/bench_%j.out
#
# Read-path benchmark. Deliberately NOT on short-unkillable: that slot belongs to the campaign, and
# this is tooling measurement, not the experiment. One a100l is enough (gemma-4-31B is 58.25 GiB, so
# it must be the 80 GB card, never the 40 GB a100).
set -u
cd /network/scratch/b/baldelld/MATS/neural-contraints-learn
P=/network/scratch/b/baldelld/MATS/venv-vllm/bin/python
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6
echo "=== $(date) on $(hostname) ==="
$P -u scripts/bench_read.py --model "${MODEL:-google/gemma-4-31B-it}" \
   --layer "${LAYER:-53}" --turns "${TURNS:-120}" --reps "${REPS:-3}" \
   --batches 1 2 4 --gpu-frac 0.92 --max-model-len 32768
echo "=== done $(date) ==="
