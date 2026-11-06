#!/bin/bash
#SBATCH --job-name=nc-v6
#SBATCH --partition=short-unkillable
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=3:00:00
#SBATCH --output=logs/v6_%j.out
#
# CROSS-DRAW AND CROSS-MODEL REPLICATION of the v5 rotation x alignment grid.
#
# WHAT IT DELIVERS, in priority order (each job resumes exactly where the last one stopped):
#   Q   Qwen3.5-9B draw 14        144 sessions   outputs/v6q       the second POPULATION. v5's
#                                                                  alignment effect is n=1 in the
#                                                                  unit that matters, because one
#                                                                  draw fixes the 100 coordinates,
#                                                                  and v4 showed the population
#                                                                  effect can exceed the arm effect.
#   G0  gemma-4-31B-it calibrate + fit + screen   artifacts/v6g
#   G1  gemma draw A              144 sessions   outputs/v6g_dA     the second MODEL.
#   G2  gemma draw B              144 sessions   outputs/v6g_dB
#
# WHY ONE SELF-CHAINING JOB AND NOT FOUR: only one short-unkillable runs at a time
# (QOSMaxGRESPerUser), so parallel submissions buy nothing and just fight for the slot. One job that
# always does the highest-priority remaining work fills every slot and keeps exactly one thing queued.
#
# WHY THE FIRST SLOT SPLITS THE NODE: gemma calibration is deliberately UNSHARDED (v4 stage 1 -- a
# merge across four processes can misalign activations from labels silently, which already cost this
# project a whole run), so it uses one GPU. The other three run Qwen sessions rather than idle ~2 h.
#
# WHY PARTIAL RESULTS ARE VALID, WHICH IS THE POINT: session indices cycle grid cells fastest and the
# target seed slowest, and every shard works its list in ascending order, so finished sessions always
# form a COMPLETE GRID at however many seeds got done. Stopping this campaign at any moment costs
# precision, never coverage -- so gemma's cost, which is unmeasured, cannot produce an unusable dir.
#
# GEMMA 4 NOTES (all verified 2026-08-14, scripts/check_gemma_template.py):
#   * MODEL ID MUST BE THE UPPERCASE B, google/gemma-4-31B-it. The lowercase cache entry is
#     configs-only (31 MB) and would fail under HF_HUB_OFFLINE. There is no Gemma 4 27B; the dense
#     Gemma 4 sizes are 12B and 31B (27B was Gemma 3).
#   * the template needs no system role -- the board protocol already folds instructions into the
#     first user message and turns alternate strictly, so it renders unchanged.
#   * the DEFAULT template prefills a closed empty thought channel (`<|channel>thought\n<channel|>`),
#     so the answer follows directly and extra_left=1 lands on the last prompt token, as on Qwen.
#     enable_thinking=True would instead inject a system turn and let the model think: not wanted.
#   * one token per move symbol holds ('r r u' -> 3 tokens), which is what the whole positional
#     label mapping rests on, and session reads force the response to the RE-JOINED parsed moves,
#     so stray model whitespace cannot shift the span.
#   * generation_config eos is [1, 106, 50] and 106 is `<turn|>`, so vLLM stops at the turn
#     delimiter on its own; no stop_tokens entry is needed (unlike olmo, which overruns).
#   * 262k vocab: the letter-distribution read uses prompt_logprobs, whose fp32 [tokens x vocab]
#     buffer OOMs at vLLM's default prefill batch. Hence --max-num-batched-tokens 2560 (vLLM rejects
#     below 2496 for a multimodal model) at gmu 0.87 for slack, and --chunk 32 to bound the 8-layer
#     activation capture. Sessions use no logprobs, so they get gmu 0.92.
#   * weights are 58.25 GiB of an 80 GB card, and 50 of 60 layers are sliding-window (1024), so KV
#     is only ~3.4 GiB per 32k sequence -> batch 2 fits with headroom. Sessions reach ~26k tokens
#     (39 tok/turn x 560 turns, measured in BOTH tokenizers), so max_model_len 32768 is sized.
#   * THE WALL GUARD IS NOW EFFECTIVELY OFF (999) ON PURPOSE, 2026-08-15. It existed because an
#     interrupted batch lost everything, so refusing to start a batch that would overrun was the
#     lesser evil. Per-trial checkpointing removed that reason: an interrupted session now loses
#     ~2 min. With the guard at 130 each GPU completed ONE ~90 min batch per 170 min slot and then
#     sat idle, which is where most of the missing throughput was going. Jobs will now normally end
#     in TIMEOUT rather than cleanly; that is fine, because the successor is queued with afterany
#     and the analysis is run outside the job.
set -u
cd /network/scratch/b/baldelld/MATS/neural-contraints-learn
P=/network/scratch/b/baldelld/MATS/venv-vllm/bin/python
CHAIN=${CHAIN:-1}
CHAIN_MAX=${CHAIN_MAX:-40}
STATUS=logs/v6_campaign.status
mkdir -p logs artifacts/v6g
export TOKENIZERS_PARALLELISM=false
say() { echo "[$(date '+%m-%d %H:%M:%S')] chain $CHAIN: $*" | tee -a "$STATUS"; }

QMODEL=Qwen/Qwen3.5-9B ; QTAG=Qwen3.5-9B ; QV=artifacts/v4 ; QDRAW=14 ; QOUT=outputs/v6q
GMODEL=google/gemma-4-31B-it ; GTAG=gemma-4-31B-it ; GV=artifacts/v6g
GSCREEN="$GV/screen_${GTAG}.json" ; GDRAWS_J="$GV/draws_${GTAG}.json" ; GREADS="$GV/reads_${GTAG}.npz"
GBLOCKED="$GV/GEMMA_BLOCKED" ; GTRIES="$GV/calib_attempts"
TOTAL=144
GRID_ARGS="--grid-thetas 50 60 70 78 84 87 --grid-alignments 1.0 0.7 0.4 0.2 0.1 --grid-control"
# byte-identical to v5 so draw 14 is a clean REPLICATION rather than a new condition
LEN_ARGS="--trials-baseline 8 --trials-perturb 80 --trials-washout 8 --move-budget 48 --max-turns 560"

missing_list() {   # $1 = sweep dir -> ascending indices with neither trials nor a skip marker
  $P - "$1" "$TOTAL" <<'PYEOF'
import sys
from pathlib import Path
base, total = Path(sys.argv[1]), int(sys.argv[2])
def done(i):
    for pat in (f"sess_{i:03d}_trials.json", f"sess_{i:03d}_skipped.json"):
        if (base / pat).exists() or any(base.glob(f"*/{pat}")):
            return True
    return False
print(" ".join(str(i) for i in range(total) if not done(i)))
PYEOF
}

gemma_draws() {    # the two draws the pre-declared criteria pick, or empty if none qualifies
  [ -f "$GSCREEN" ] || return 0
  $P -c "
import json
sc=json.load(open('$GSCREEN')); usable=[float(x) for x in sc['usable_levels']]
c=[]
for x in sc['rows']:
    ch=x.get('chosen') or {}
    if all(str(l) in ch and ch[str(l)] and ch[str(l)].get('pass') for l in usable):
        c.append((x.get('severity_spread_rel',9), x.get('mag_spread_rel',9), x['draw']))
c.sort(); print(' '.join(str(d) for _,_,d in c[:2]))" 2>/dev/null
}

progress_count() { # every written session plus skip marker, across every sweep dir; the guard metric
  ls -1 outputs/v6q/*/sess_*_trials.json outputs/v6q/*/sess_*_skipped.json \
        outputs/v6g_d*/*/sess_*_trials.json outputs/v6g_d*/*/sess_*_skipped.json 2>/dev/null | wc -l
}

# deal an ascending index list round-robin across GPUs, each shard ascending, so the LOWEST indices
# finish first everywhere -> a complete grid at fewer seeds, never a grid with holes
shard_of() { local k=$1 n=$2; shift 2; local i=0 out=""; for x in "$@"; do
  [ $((i % n)) -eq "$k" ] && out="$out $x"; i=$((i+1)); done; echo "$out"; }

SHARD_PID=(); SHARD_LOG=()

run_sweep() {   # model reads screen draw outdir gpucsv batch maxlen gmu stopflag
  local MODEL="$1" R="$2" S="$3" DRAW="$4" OUT="$5" GPUS="$6" B="$7" ML="$8" GMU="$9" STOP="${10}"
  local MISS; MISS=($(missing_list "$OUT")); local N=${#MISS[@]}
  if [ "$N" -eq 0 ]; then say "  $OUT: complete"; return 0; fi
  IFS=',' read -r -a GA <<< "$GPUS"; local NG=${#GA[@]}
  say "  $OUT: $N of $TOTAL missing, dealt across $NG GPU(s) [$GPUS], batch $B, $STOP"
  local k=0
  for g in "${GA[@]}"; do
    local LIST; LIST=$(shard_of "$k" "$NG" "${MISS[@]}")
    k=$((k+1)); [ -z "$LIST" ] && continue
    local LG="logs/v6_c${CHAIN}_$(basename "$OUT")_gpu${g}.log"
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=5 $P -u scripts/run.py \
        --model "$MODEL" --reads "$R" --screen "$S" --mode blocks \
        --draws "$DRAW" --repeats 6 $GRID_ARGS $LEN_ARGS \
        --session-list $LIST --skip-existing $STOP \
        --batch "$B" --max-model-len "$ML" --gpu-frac "$GMU" \
        --tag "c${CHAIN}g${g}" --out "$OUT" > "$LG" 2>&1 &
    SHARD_PID+=("$!"); SHARD_LOG+=("$LG")
  done
}

# kill a shard and its vLLM children; a kill -9 on the python parent alone leaves EngineCore
# holding the GPU
kill_tree() { local p=$1 c; for c in $(pgrep -P "$p" 2>/dev/null); do kill_tree "$c"; done
  kill -9 "$p" 2>/dev/null; }

# WATCHDOG (added 2026-08-15 after chain 5). A shard can livelock inside vLLM: process alive, GPU
# busy, log frozen mid-generate at 0 tok/s, no error ever raised. The bare `wait` below then blocks
# until the 3 h wall kills the whole job, which costs the surviving shards' in-flight batches AND the
# analysis tail. Every shard writes progress continuously (vLLM's progress bar touches the file), so
# a log silent for $1 minutes is a hang, not slow work. Killing it lets the other shards finish and
# the tail run; its unwritten sessions are simply retried by the next chain.
watchdog() {
  local stall=$1 i p f age any
  while true; do
    sleep 120
    any=0
    for i in "${!SHARD_PID[@]}"; do
      p=${SHARD_PID[$i]}; f=${SHARD_LOG[$i]}
      kill -0 "$p" 2>/dev/null || continue
      any=1
      age=$(( ($(date +%s) - $(stat -c %Y "$f" 2>/dev/null || echo 0)) / 60 ))
      if [ "$age" -ge "$stall" ]; then
        say "WATCHDOG: $f silent for ${age} min -> livelock; killing pid $p so the rest can finish"
        kill_tree "$p"
      fi
    done
    [ "$any" -eq 0 ] && return
  done
}
start_watchdog() { [ ${#SHARD_PID[@]} -eq 0 ] && return; watchdog "${1:-30}" & WD=$!; }
stop_watchdog() { [ -n "${WD:-}" ] && kill -9 "$WD" 2>/dev/null; WD=""; }

PROG0=$(progress_count)
QMISS=($(missing_list "$QOUT")); NQ=${#QMISS[@]}
GEMMA_READY=0; [ -f "$GSCREEN" ] && GEMMA_READY=1
GEMMA_DEAD=0;  [ -f "$GBLOCKED" ] && GEMMA_DEAD=1
RAN_G0=0
say "state: qwen draw $QDRAW $((TOTAL-NQ))/$TOTAL | gemma screen $GEMMA_READY | blocked $GEMMA_DEAD"

# QUEUE THE SUCCESSOR NOW, NOT AT THE END (fixed 2026-08-15). Chain 5 was killed by the 3 h wall
# while blocked on a hung shard, so it never reached its resubmit line and the campaign silently
# stopped for hours. With `afterany` the next chain is already queued before any work starts, so it
# runs no matter HOW this job ends: clean exit, wall kill, node failure. The cost of the successor
# being unnecessary is one short job that finds nothing to do and exits; the guards below scancel it
# whenever this chain establishes there is no more work.
WORK=0
[ "$NQ" -gt 0 ] && WORK=1
[ "$GEMMA_DEAD" -eq 0 ] && WORK=1          # gemma either needs calibrating or needs sessions
NEXT=""
if [ "$WORK" -eq 1 ] && [ "$CHAIN" -lt "$CHAIN_MAX" ]; then
  NEXT=$(sbatch --parsable --dependency=afterany:"${SLURM_JOB_ID:-0}" \
         --export=ALL,CHAIN=$((CHAIN+1)),CHAIN_MAX=$CHAIN_MAX scripts/v6_campaign.sh 2>/dev/null)
  say "queued chain $((CHAIN+1)) as job ${NEXT:-FAILED} (afterany, survives a wall kill)"
fi

# ============================================================ STAGE G0: gemma calibration
if [ "$GEMMA_READY" -eq 0 ] && [ "$GEMMA_DEAD" -eq 0 ]; then
  RAN_G0=1
  TRIES=$(cat "$GTRIES" 2>/dev/null || echo 0); TRIES=$((TRIES+1)); echo "$TRIES" > "$GTRIES"
  if [ "$TRIES" -gt 3 ]; then
    say "gemma calibration has failed $((TRIES-1))x; marking BLOCKED so the node goes to Qwen"
    echo "calibration did not complete in $((TRIES-1)) attempts" > "$GBLOCKED"
    GEMMA_DEAD=1
  else
    say "STAGE G0 attempt $TRIES: gemma calibrate on GPU 0, Qwen sessions fill GPUs 1-3"
    [ "$NQ" -gt 0 ] && run_sweep "$QMODEL" "$QV/reads_${QTAG}.npz" "$QV/screen_${QTAG}.json" \
        "$QDRAW" "$QOUT" 1,2,3 4 81920 0.90 "--deadline-min 95"
    say "  G0a calibrate (1 GPU on purpose; chunk 32, mnbt 2560, gmu 0.87, max_model_len 4096)"
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 $P -u scripts/calibrate.py \
        --model "$GMODEL" --n-fit 1200 --n-val 600 --n-depths 8 \
        --chunk 32 --gpu-frac 0.87 --max-model-len 4096 --max-num-batched-tokens 2560 \
        --out "$GV" > logs/v6_gemma_calibrate.log 2>&1
    if [ ! -f "$GREADS" ]; then
      say "  G0a FAILED (attempt $TRIES of 3). tail of logs/v6_gemma_calibrate.log:"
      tail -20 logs/v6_gemma_calibrate.log | tee -a "$STATUS"
    else
      $P -c "
import numpy as np, json
z=np.load('$GREADS',allow_pickle=True); r=json.load(open('$GV/reads_${GTAG}.json'))
print('  G0a ok: %d fit samples, %d val, layers %s, letter-dist missing %.4f'
      % (z['U_fit'].shape[1], z['U_val'].shape[1], list(z['layers']),
         r['letter_dist_missing_frac_fit']))" 2>&1 | tail -2 | tee -a "$STATUS"
      say "  G0b fit 20 draws on the intent label (CPU, 4 shards)"
      for k in 0 1 2 3; do
        OMP_NUM_THREADS=5 $P -u scripts/fit_draws.py --reads "$GREADS" --label intent \
            --draws 20 --draw-start $((5 * k)) --draw-count 5 \
            --out "$GV/draws_shard${k}.json" > "logs/v6_gemma_fit${k}.log" 2>&1 &
      done
      wait
      $P scripts/merge_draw_shards.py --shards "$GV"/draws_shard*.json --out "$GDRAWS_J" \
          2>&1 | tail -3 | tee -a "$STATUS"
      BEATS=$($P -c "import json;print(json.load(open('$GDRAWS_J'))['gates'].get('beats_prompt_only'))" 2>/dev/null || echo ERR)
      $P -c "
import json, statistics as st
r=json.load(open('$GDRAWS_J')); b=r['prompt_only_baselines']
print('  prompt-only best %.4f (%s)' % (b['best_prompt_only'], b['best_prompt_only_source']))
rows=[x for x in r['rows'] if 'r2' in x]
for L in sorted({x['layer'] for x in rows}):
    g=[x for x in rows if x['layer']==L]
    print('    layer %3d  R2 %+.3f  vs prompt %+.3f' % (L, st.median(x['r2'] for x in g),
          st.median(x['gap_over_prompt'] for x in g)))" 2>&1 | tee -a "$STATUS"
      say "  G0b beats_prompt_only=$BEATS"
      if [ "$BEATS" != "True" ]; then
        say "  STOP GEMMA: no layer beats the prompt-only baseline. That is a RESULT about this"
        say "  model, not a crash: a decoder that cannot outread the prompt has no readout to"
        say "  perturb, so its grid would measure nothing. Qwen keeps the node."
        echo "beats_prompt_only=$BEATS" > "$GBLOCKED"; GEMMA_DEAD=1
      else
        say "  G0c screen"
        OMP_NUM_THREADS=20 $P -u scripts/screen.py --reads "$GREADS" --draws 20 \
            --scout-draws 3 --n-cand 40 --fit-report "$GDRAWS_J" \
            > logs/v6_gemma_screen.log 2>&1
        if [ ! -f "$GSCREEN" ]; then
          say "  G0c FAILED:"; tail -15 logs/v6_gemma_screen.log | tee -a "$STATUS"
        else
          $P -c "
import json; r=json.load(open('$GSCREEN'))
print('  G0c ok: layer %s | G2_PASS %s | usable %s | severity-matched %s/%s'
      % (r['layer'], r['G2_PASS'], r['usable_levels'],
         r['n_draws_severity_matched'], r['draws']))" | tee -a "$STATUS"
          GEMMA_READY=1
          say "  gemma draws chosen by the pre-declared criteria: [$(gemma_draws)]"
        fi
      fi
    fi
  fi
  start_watchdog 30
  wait                     # let the Qwen shards on GPUs 1-3 finish their in-flight batches
  stop_watchdog
  say "STAGE G0 slot done"

# ============================================================ STAGE Q: all 4 GPUs on Qwen
elif [ "$NQ" -gt 0 ]; then
  say "STAGE Q: Qwen draw $QDRAW on all 4 GPUs ($NQ missing)"
  run_sweep "$QMODEL" "$QV/reads_${QTAG}.npz" "$QV/screen_${QTAG}.json" "$QDRAW" \
            "$QOUT" 0,1,2,3 4 81920 0.90 "--deadline-min 95"
  start_watchdog 30
  wait
  stop_watchdog

# ============================================================ STAGE G1/G2: the gemma sweeps
elif [ "$GEMMA_DEAD" -eq 0 ]; then
  GDR=$(gemma_draws)
  if [ -z "$GDR" ]; then
    say "no gemma draw is severity-matched on every usable level; marking BLOCKED"
    echo "no draw severity-matched across usable levels" > "$GBLOCKED"
  else
    set -- $GDR; D1=$1; D2=${2:-}
    N1=($(missing_list "outputs/v6g_d$D1")); N1=${#N1[@]}
    N2=0; [ -n "$D2" ] && { n=($(missing_list "outputs/v6g_d$D2")); N2=${#n[@]}; }
    # BOTH GEMMA DRAWS ADVANCE TOGETHER, two GPUs each, rather than draw A running to completion
    # first. Total throughput is identical, but the population question is the whole reason there are
    # two draws: two draws at three seeds each answers it, one draw at six seeds and one at zero does
    # not. Only when a draw is finished does the other take the whole node.
    if [ "$N1" -gt 0 ] && [ "$N2" -gt 0 ]; then
      say "STAGE G1+G2: gemma draws $D1 and $D2 in parallel, two GPUs each ($N1 / $N2 missing)"
      run_sweep "$GMODEL" "$GREADS" "$GSCREEN" "$D1" "outputs/v6g_d$D1" 0,1 2 32768 0.92 \
                "--wall-limit-min 999"
      run_sweep "$GMODEL" "$GREADS" "$GSCREEN" "$D2" "outputs/v6g_d$D2" 2,3 2 32768 0.92 \
                "--wall-limit-min 999"
    elif [ "$N1" -gt 0 ]; then
      say "STAGE G1: gemma draw $D1 on all 4 GPUs ($N1 missing; draw $D2 done)"
      run_sweep "$GMODEL" "$GREADS" "$GSCREEN" "$D1" "outputs/v6g_d$D1" 0,1,2,3 2 32768 0.92 \
                "--wall-limit-min 999"
    elif [ "$N2" -gt 0 ]; then
      say "STAGE G2: gemma draw $D2 on all 4 GPUs ($N2 missing; draw $D1 done)"
      run_sweep "$GMODEL" "$GREADS" "$GSCREEN" "$D2" "outputs/v6g_d$D2" 0,1,2,3 2 32768 0.92 \
                "--wall-limit-min 999"
    fi
    start_watchdog 30
    wait
    stop_watchdog
  fi
fi

# ============================================================ progress, analysis, resubmit
PROG1=$(progress_count)
NQ2=($(missing_list "$QOUT")); NQ2=${#NQ2[@]}
GLEFT=0
if [ -f "$GBLOCKED" ]; then GLEFT=0
elif [ ! -f "$GSCREEN" ]; then GLEFT=288
else
  for d in $(gemma_draws); do
    n=($(missing_list "outputs/v6g_d$d")); GLEFT=$((GLEFT + ${#n[@]}))
  done
fi
say "sessions written this slot: $((PROG1 - PROG0)).  remaining: qwen $NQ2, gemma $GLEFT"

# report on whatever is complete every slot, so the morning always has current numbers
[ "$NQ2" -eq 0 ] && [ -d "$QOUT" ] && { say "--- Qwen draw $QDRAW grid ---"
  $P scripts/analyze_grid.py --dir "$QOUT" 2>&1 | tail -20 | tee -a "$STATUS"; }
$P scripts/compare_draws.py 2>&1 | tail -34 | tee -a "$STATUS" || say "compare_draws failed"

if [ "$NQ2" -eq 0 ] && [ "$GLEFT" -eq 0 ]; then
  say "CAMPAIGN COMPLETE; regenerating figures"
  [ -n "$NEXT" ] && { scancel "$NEXT" 2>/dev/null; say "cancelled the now-unneeded successor $NEXT"; }
  $P scripts/figures.py > logs/v6_figures.log 2>&1 || say "figures failed"
  say "DONE"
elif [ "$RAN_G0" -eq 0 ] && [ "$PROG1" -le "$PROG0" ]; then
  # G0 is exempt: a failed calibration writes no sessions but does advance its attempt counter,
  # which the 3-attempt cap already bounds.
  [ -n "$NEXT" ] && scancel "$NEXT" 2>/dev/null
  say "STOPPING THE CHAIN (successor $NEXT cancelled): zero sessions written this slot."
  say "  look at logs/v6_c${CHAIN}_*.log before resubmitting."
elif [ "$CHAIN" -ge "$CHAIN_MAX" ]; then
  say "chain cap $CHAIN_MAX reached with qwen $NQ2 / gemma $GLEFT left; resubmit by hand if wanted"
elif [ -n "$NEXT" ]; then
  say "chain $((CHAIN+1)) (job $NEXT) is already queued and will pick up from here"
else
  # the up-front sbatch failed (queue limit? scheduler hiccup?), so fall back to submitting now
  say "successor was never queued; submitting chain $((CHAIN+1)) from the tail instead"
  sbatch --export=ALL,CHAIN=$((CHAIN+1)),CHAIN_MAX=$CHAIN_MAX scripts/v6_campaign.sh
fi
