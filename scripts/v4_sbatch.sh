#!/bin/bash
#SBATCH --job-name=nc-v4
#SBATCH --partition=short-unkillable
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=3:00:00
#SBATCH --output=logs/v4_slurm_%j.out
#
# THE WHOLE PIPELINE under the corrected read convention (DESIGN 4.2, re-indexed 2026-08-12):
# one activation per emitted letter, namely the one that generated it. Everything has to be
# re-run because a decoder must be applied under the convention it was fit on.
#
# Stage 1 calibrate    1 GPU,  ~25 min   deliberately NOT sharded: sharding would mean merging
#                                        activations and labels from four processes, and a merge
#                                        bug misaligns them silently. That exact class of bug
#                                        already cost this project a whole run. 75 idle GPU-min
#                                        is a cheap price for keeping alignment in one process
#                                        with one assertion.
# Stage 2 fit draws    4 CPU procs       rows are tagged with their draw index, so splitting by
#                                        draw and concatenating row lists carries no alignment
#                                        risk. ~35 min sequential becomes ~10.
# Stage 3 screen       1 CPU proc, ~20 min
# Stage 4 closed loop  4 GPUs, ~60 min   96 sessions, 24 per shard, the verified v3 sharding.
# Stage 5 analysis + figures
#
# Total ~2 h against the 3 h limit. Four GPUs are load-bearing for stage 4: at two the closed
# loop alone would run past the wall.
set -u
cd /network/scratch/b/baldelld/MATS/neural-contraints-learn
P=/network/scratch/b/baldelld/MATS/venv-vllm/bin/python
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
TAG=${MODEL##*/}
V=artifacts/v4
OUT=outputs/v4
STATUS=logs/v4.status
mkdir -p "$V" "$OUT" logs
export TOKENIZERS_PARALLELISM=false
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

say "STAGE 1/5  calibrate, one read per emitted letter  (1 GPU)"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 $P -u scripts/calibrate.py \
    --model "$MODEL" --n-fit 1200 --n-val 600 --n-depths 8 --out "$V" \
    > logs/v4_calibrate.log 2>&1
R="$V/reads_${TAG}.npz"
[ -f "$R" ] || { say "FAILED at calibrate:"; tail -25 logs/v4_calibrate.log | tee -a "$STATUS"; exit 1; }
$P - <<PYEOF | tee -a "$STATUS"
import numpy as np, json
z = np.load("$R", allow_pickle=True)
r = json.load(open("$V/reads_${TAG}.json"))
print(f"  fit samples {z['U_fit'].shape[1]} (was 5076 at k-1 per reply; expect ~15% more)")
print(f"  letter-dist missing {r['letter_dist_missing_frac_fit']:.4f}")
PYEOF

say "STAGE 2/5  fit draws on the intent label, 4 CPU shards"
for k in 0 1 2 3; do
  OMP_NUM_THREADS=6 $P -u scripts/fit_draws.py --reads "$R" --label intent \
      --draws 20 --draw-start $((5 * k)) --draw-count 5 \
      --out "$V/draws_shard${k}.json" > "logs/v4_fit${k}.log" 2>&1 &
done
wait
DJ="$V/draws_${TAG}.json"
$P scripts/merge_draw_shards.py --shards "$V"/draws_shard*.json --out "$DJ" 2>&1 | tee -a "$STATUS"
[ -f "$DJ" ] || { say "FAILED at fit/merge:"; tail -20 logs/v4_fit0.log | tee -a "$STATUS"; exit 1; }
BEATS=$($P -c "import json;print(json.load(open('$DJ'))['gates'].get('beats_prompt_only'))")
say "fit done: beats_prompt_only=$BEATS"
$P -c "
import json, statistics as st
r=json.load(open('$DJ')); b=r['prompt_only_baselines']
print('  prompt-only best %.4f (%s)' % (b['best_prompt_only'], b['best_prompt_only_source']))
rows=[x for x in r['rows'] if 'r2' in x]
for L in sorted({x['layer'] for x in rows}):
    g=[x for x in rows if x['layer']==L]
    print('  layer %3d  R2 %+.3f  vs prompt %+.3f  match %.3f' % (L,
      st.median(x['r2'] for x in g), st.median(x['gap_over_prompt'] for x in g),
      st.median(x.get('match_acc', float('nan')) for x in g)))" | tee -a "$STATUS"
if [ "$BEATS" != "True" ]; then
  say "STOP: no layer beats the prompt-only baseline. That is a result, not a crash."
  exit 2
fi

say "STAGE 3/5  G2 screen"
OMP_NUM_THREADS=20 $P -u scripts/screen.py --reads "$R" --draws 20 --scout-draws 3 \
    --n-cand 40 --fit-report "$DJ" > logs/v4_screen.log 2>&1
S="$V/screen_${TAG}.json"
[ -f "$S" ] || { say "FAILED at screen:"; tail -20 logs/v4_screen.log | tee -a "$STATUS"; exit 1; }
$P -c "
import json; r=json.load(open('$S'))
print('  layer %s | G2_PASS %s | usable %s | matched %s/%s' % (r['layer'], r['G2_PASS'],
      r['usable_levels'], r['n_draws_severity_matched'], r['draws']))" | tee -a "$STATUS"

say "STAGE 4/5  closed loop, 2 draws x 4 arms x 12 target seeds, 4 GPUs"
DRAWS=$($P -c "
import json, numpy as np
sc=json.load(open('$S')); dr=json.load(open('$DJ')); L=sc['layer']
usable=[float(x) for x in sc['usable_levels']]
# pick the two draws whose arms are most tightly severity-matched and least speed-confounded,
# the criteria declared before any behaviour is run
c=[]
for x in sc['rows']:
    ch=x.get('chosen') or {}
    if all(str(l) in ch and ch[str(l)] and ch[str(l)].get('pass') for l in usable):
        c.append((x.get('severity_spread_rel',9), x.get('mag_spread_rel',9), x['draw']))
c.sort()
print(' '.join(str(d) for _,_,d in c[:2]))")
say "  draws chosen by pre-declared criteria: $DRAWS"
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k OMP_NUM_THREADS=5 $P -u scripts/run.py \
      --reads "$R" --screen "$S" --mode blocks \
      --draws $DRAWS --alignments 1.0 0.7 0.4 --control-arm --repeats 12 \
      --session-start $((24 * k)) --sessions 24 --batch 8 \
      --trials-baseline 8 --trials-perturb 40 --trials-washout 8 \
      --max-turns 290 --max-model-len 40960 --gpu-frac 0.90 \
      --tag shard$k --out "$OUT" > "logs/v4_shard${k}.log" 2>&1 &
done
wait
say "sessions written: $(ls -1 $OUT/shard*/sess_*_trials.json 2>/dev/null | wc -l) of 96"

say "STAGE 5/5  analysis and figures"
$P scripts/analyze_v3.py --dir "$OUT" 2>&1 | grep -v RuntimeWarning | tee -a "$STATUS"
$P scripts/check_context_confound.py --dir "$OUT" 2>&1 | grep -v RuntimeWarning | tee -a "$STATUS"
$P scripts/figures.py > logs/v4_figures.log 2>&1 || say "figures failed, see logs/v4_figures.log"
say "DONE"
