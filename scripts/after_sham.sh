#!/bin/bash
# Wait for the running closed-loop job to exit, then compare the no-perturbation control
# against G4. Written as a FILE, not a heredoc: a previous version was created by a heredoc
# inside a shell command, so the parent shell's own argv contained the pgrep pattern and the
# wait loop matched it forever. Wait on a PID instead of a pattern.
set -u
PIDFILE="${1:?usage: after_sham.sh <pid>}"
cd /network/scratch/b/baldelld/MATS/neural-contraints-learn
P=/network/scratch/b/baldelld/MATS/venv-vllm/bin/python
while kill -0 "$PIDFILE" 2>/dev/null; do sleep 15; done
echo "===== SHAM CONTROL FINISHED ====="
$P scripts/analyze.py --dir outputs/sham --mode blocks --bin 4 2>&1 | tail -16
echo
echo "===== drift control vs G4 ====="
$P - <<'PYEOF'
import json, numpy as np
def load(p):
    try: return json.load(open(p))
    except Exception: return []
sham, g4 = load("outputs/sham/summary.json"), load("outputs/g4/summary.json")
print(f"{'set':>16} {'metric':>7} {'arm':>6} {'n':>3} {'L_session':>10} {'tail_mean':>10} {'impair':>8}")
for name, rows in (("no-perturbation", sham), ("G4", g4)):
    for met in ("paper", "angle"):
        arms = sorted({r.get("alignment") for r in rows if r["metric"] == met}, key=lambda x: (x is None, x))
        for al in arms:
            g = [r for r in rows if r["metric"] == met and r.get("alignment") == al]
            if not g: continue
            print(f"{name:>16} {met:>7} {str(al):>6} {len(g):3d} "
                  f"{np.mean([r['l_session'] for r in g]):10.3f} "
                  f"{np.mean([r['tail_mean'] for r in g]):+10.3f} "
                  f"{np.mean([r['initial_impairment'] for r in g]):8.2f}")
print()
print("MEASURED 2026-08-10: the control came out at L 1.66 / tail +0.61 (paper), HIGHER than")
print("every perturbed arm. L_session divides by the impairment, so with nothing perturbed the")
print("denominator is near zero and the ratio explodes. The metric therefore needs a MODERATE")
print("impairment: large enough to estimate L_max, small enough not to floor the outcome.")
print("Our severity floors it (moves 35/36); no perturbation explodes it. Neither side is usable.")
PYEOF
