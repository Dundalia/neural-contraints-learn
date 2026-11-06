#!/bin/bash
#SBATCH --job-name=nc-bench2
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=46G
#SBATCH --time=2:00:00
#SBATCH --output=logs/bench2_%j.out
set -u
cd /network/scratch/b/baldelld/MATS/neural-contraints-learn
P=/network/scratch/b/baldelld/MATS/venv-vllm/bin/python
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6
echo "===== A. equivalence of the cached read, at production depths, through the REAL readout ====="
$P -u scripts/bench_equiv.py --depths 60 150 300 450 --samples 16
echo "===== B. read-path benchmark incl. generation+span_prefix_cache (missed last run) ====="
$P -u scripts/bench_read.py --turns 300 --reps 3 --batches 1 2
