#!/usr/bin/env python
"""Is the within-block degradation a learning effect, or just context length?

THE PROBLEM. Under perturbation the model takes 17 to 36 moves per trial instead of 9, so it
burns more TURNS per trial, so its conversation grows faster. If the model degrades as context
grows, the perturbed arms would show steeper degradation across trials for a reason that has
nothing to do with the perturbation or with re-learning. The control's own mild positive moves
slope is consistent with such an effect existing.

THREE CHECKS, each of which the degradation claim has to survive.

1. Degradation against TURNS elapsed rather than trials elapsed. If context length is the driver,
   the arms should agree once plotted against turns, because that is the axis that tracks context.
2. Degradation inside the WASHOUT block. Washout has the original decoder back but the LONGEST
   context of the session. If context is the driver, washout should be degraded relative to
   baseline; if the perturbation is the driver, washout should return to baseline. This is the
   cleanest of the three because it separates the two explanations by construction.
3. The control arm's own slope against turns, which is the pure context effect with no
   perturbation present at all.

    python scripts/check_context_confound.py --dir outputs/v3
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sessions(dirs):
    for d in dirs:
        mp = Path(d) / "meta.json"
        if not mp.exists():
            continue
        meta = {m["session"]: m for m in json.loads(mp.read_text())}
        for f in sorted(glob.glob(f"{d}/sess_*_trials.json")):
            i = int(Path(f).stem.split("_")[1])
            m = meta.get(i)
            if m is None:
                continue
            pert = m.get("perturbation") or {}
            arm = pert.get("alignment")
            yield {"draw": m.get("draw"), "repeat": m.get("repeat"), "session": i,
                   "arm": None if arm is None else round(float(arm), 2),
                   "trials": json.loads(Path(f).read_text())}


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return np.nan
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def summ(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < 3:
        return np.nan, np.nan, len(v)
    se = v.std(ddof=1) / math.sqrt(len(v))
    # A degenerate spread means every session gave the identical slope (the control is exactly
    # flat because it never fails a trial). Float noise makes se tiny rather than zero, which
    # printed a t of 3e16 on the first pass. Treat it as undefined, not as overwhelming evidence.
    if not np.isfinite(se) or se < 1e-12:
        return float(v.mean()), np.nan, len(v)
    return float(v.mean()), float(v.mean() / se), len(v)


def fmt(a):
    return "no perturbation" if a is None else f"alignment {a:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/v3")
    a = ap.parse_args()
    dirs = sorted(glob.glob(str(ROOT / a.dir / "shard*"))) or [str(ROOT / a.dir)]
    S = list(sessions(dirs))
    arms = sorted({s["arm"] for s in S}, key=lambda x: (x is None, -(x or 0)))
    print(f"[ctx] {len(S)} sessions\n")

    # ---- 1. slope against trials vs against cumulative turns, inside the perturbation block
    print("CHECK 1  degradation per TRIAL vs per TURN (perturbation block)")
    print("if context length drives it, the per-turn slopes should agree across arms")
    print(f"{'arm':>17} {'n':>3} {'moves/trial':>12} {'t':>7} {'moves/turn':>12} {'t':>7}")
    for arm in arms:
        st, su = [], []
        for s in (x for x in S if x["arm"] == arm):
            g = [t for t in s["trials"] if t["block"] == "perturbation"]
            if len(g) < 10:
                continue
            mv = [t["moves_used"] for t in g]
            # turns are not recorded per trial, but moves/reply is ~10, so cumulative moves is a
            # faithful monotone proxy for cumulative turns and needs no extra data
            cum = np.cumsum([0] + mv[:-1])
            st.append(slope(np.arange(len(mv)), mv))
            su.append(slope(cum, mv))
        m1, t1, n = summ(st)
        m2, t2, _ = summ(su)
        print(f"{fmt(arm):>17} {n:3d} {m1:+12.4f} {t1:+7.2f} {m2:+12.5f} {t2:+7.2f}")

    # ---- 2. washout vs baseline: longest context, original decoder
    print("\nCHECK 2  washout against baseline, paired within session")
    print("washout has the LONGEST context but the ORIGINAL decoder, so a context effect shows")
    print("up here as washout being worse than baseline, and a perturbation effect does not")
    print(f"{'arm':>17} {'n':>3} {'washout - baseline moves':>25} {'t':>7} {'success diff':>13}")
    for arm in arms:
        dm, ds = [], []
        for s in (x for x in S if x["arm"] == arm):
            b = [t for t in s["trials"] if t["block"] == "baseline"]
            w = [t for t in s["trials"] if t["block"] == "washout"]
            if len(b) < 4 or len(w) < 4:
                continue
            dm.append(float(np.median([t["moves_used"] for t in w])
                            - np.median([t["moves_used"] for t in b])))
            ds.append(float(np.mean([t["acquired"] for t in w])
                            - np.mean([t["acquired"] for t in b])))
        m, t_, n = summ(dm)
        ms, _, _ = summ(ds)
        print(f"{fmt(arm):>17} {n:3d} {m:+25.2f} {t_:+7.2f} {ms:+13.3f}")

    # ---- 3. the pure context effect: the control arm across the whole session
    print("\nCHECK 3  the control arm across its whole session, the pure context effect")
    for arm in [None]:
        sl = []
        for s in (x for x in S if x["arm"] == arm):
            mv = [t["moves_used"] for t in s["trials"]]
            sl.append(slope(np.arange(len(mv)), mv))
        m, t_, n = summ(sl)
        print(f"  {fmt(arm)}: n={n}  moves slope over all 56 trials {m:+.4f} (t={t_:+.2f})")
        print("  a control that is flat here means context length alone does not degrade the model")


if __name__ == "__main__":
    main()
