#!/usr/bin/env python
"""Analysis for the crossed closed-loop design (DESIGN 7.5): draws x arms x target seeds.

THE QUESTION. Does performance recover WITHIN the perturbation block, and does the amount of
recovery depend on whether the perturbed control space stayed inside the activity manifold?

WHY THE PAIRING MATTERS. Every arm in a repeat ran on the identical target sequence with the
identical coordinate population, and severities are matched across arms within a draw. So the arm
contrast can be taken as a paired difference per (draw, repeat) cell, which removes target
difficulty and population entirely and leaves alignment as the only thing that differs.

PRIMARY OUTCOME IS SUCCESS, NOT MOVES. Moves-to-acquire is right-censored at the 36-move budget
(20% of perturbation trials on the previous run), and every failure sits exactly at the budget.
Success per trial is the indicator of beating that budget, so it is uncensored by construction.
Moves is reported too, with the censored fraction stated, never silently averaged as if complete.

    python scripts/analyze_v3.py --dir outputs/v3
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 36


def load(dirs):
    """[(draw, arm, repeat, trials)] merged across shards, with early-stop flags."""
    out = []
    for d in dirs:
        meta_p = Path(d) / "meta.json"
        if not meta_p.exists():
            continue
        meta = {m["session"]: m for m in json.loads(meta_p.read_text())}
        for f in sorted(glob.glob(f"{d}/sess_*_trials.json")):
            idx = int(Path(f).stem.split("_")[1])
            m = meta.get(idx)
            if m is None:
                continue
            tr = json.loads(Path(f).read_text())
            pert = m.get("perturbation") or {}
            # ROUND THE ARM. `alignment` is the MEASURED alignment of the built perturbation, so
            # two sessions in the same arm differ in the last float bits and grouping on the raw
            # value silently splits every arm in two (it did, on the first pass).
            arm = pert.get("alignment")
            arm = None if arm is None else round(float(arm), 2)
            # a session that stopped early truncates its own block; flag it rather than pad it
            turns = Path(d) / f"sess_{idx:03d}.jsonl"
            stopped = None
            if turns.exists():
                for line in turns.open():
                    try:
                        r = json.loads(line)
                    except Exception:                                    # noqa: BLE001
                        continue
                    if r.get("stuck_abort"):
                        stopped = "stuck"
                    elif r.get("repeat_abort"):
                        stopped = "repeat"
                    elif r.get("turn_cap"):
                        stopped = "turn cap"
            out.append({"draw": m.get("draw"), "arm": arm, "repeat": m.get("repeat"),
                        "session": idx, "dir": d, "trials": tr, "stopped": stopped,
                        "dev_deg": pert.get("dev_deg"), "mag": pert.get("mag")})
    return out


def block(s, name):
    return [t for t in s["trials"] if t["block"] == name]


def slope(y):
    """Least-squares slope against trial index. Negative means improving."""
    y = np.asarray(y, float)
    if len(y) < 8:
        return np.nan
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def ttest_1samp(v):
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


def fmt_arm(a):
    return "no perturbation" if a is None else f"alignment {a:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/v3")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dirs = sorted(glob.glob(str(ROOT / a.dir / "shard*"))) or [str(ROOT / a.dir)]
    S = load(dirs)
    if not S:
        raise SystemExit(f"no sessions under {a.dir}")

    arms = sorted({s["arm"] for s in S}, key=lambda x: (x is None, -(x or 0)))
    draws = sorted({s["draw"] for s in S})
    print(f"[v3] {len(S)} sessions from {len(dirs)} shard(s); draws {draws}; "
          f"arms {[fmt_arm(x) for x in arms]}")
    stopped = [s for s in S if s["stopped"]]
    print(f"[v3] sessions that stopped early: {len(stopped)}"
          + ("" if not stopped else "  " + ", ".join(
              f"s{s['session']}({fmt_arm(s['arm'])}, {s['stopped']})" for s in stopped)))

    # severity actually realised, the check that the matched-difficulty control held
    print(f"\n{'arm':>17} {'n':>3} {'severity deg':>13} {'magnitude':>10}")
    for arm in arms:
        g = [s for s in S if s["arm"] == arm and s["dev_deg"]]
        if g:
            print(f"{fmt_arm(arm):>17} {len(g):3d} "
                  f"{np.mean([s['dev_deg'] for s in g]):13.1f} "
                  f"{np.mean([s['mag'] for s in g]):10.2f}")

    # The draw effect turned out to be LARGER than the arm effect on this run, so show the
    # draw x arm cells before any pooled number invites the wrong reading.
    print(f"\n{'draw':>5} {'arm':>17} {'n':>3} {'success':>8} {'moves med':>10} "
          f"{'at budget':>10} {'severity':>9}")
    for dr in draws:
        for arm in arms:
            g = [s for s in S if s["draw"] == dr and s["arm"] == arm]
            if not g:
                continue
            acq, mv = [], []
            for s in g:
                b = block(s, "perturbation")
                acq += [t["acquired"] for t in b]
                mv += [t["moves_used"] for t in b]
            devs = [s["dev_deg"] for s in g if s["dev_deg"]]
            print(f"{dr:5d} {fmt_arm(arm):>17} {len(g):3d} {np.mean(acq):8.3f} "
                  f"{np.median(mv):10.0f} {np.mean(np.array(mv) >= BUDGET):10.1%} "
                  f"{(np.mean(devs) if devs else float('nan')):9.1f}")

    # ---------------------------------------------------------------- behaviour by block
    print(f"\n{'arm':>17} {'block':>13} {'success':>8} {'moves med':>10} {'at budget':>10}")
    for arm in arms:
        for blk in ("baseline", "perturbation", "washout"):
            acq, mv = [], []
            for s in (x for x in S if x["arm"] == arm):
                g = block(s, blk)
                acq += [t["acquired"] for t in g]
                mv += [t["moves_used"] for t in g]
            if acq:
                print(f"{fmt_arm(arm):>17} {blk:>13} {np.mean(acq):8.3f} "
                      f"{np.median(mv):10.0f} {np.mean(np.array(mv) >= BUDGET):10.1%}")

    # ------------------------------------------- recovery within the perturbation block
    print("\nRECOVERY within the perturbation block (per session, then across sessions)")
    print("success slope is in units of success per trial; negative moves slope = improving")
    res = {}
    for arm in arms:
        succ, mvs = [], []
        for s in (x for x in S if x["arm"] == arm):
            g = block(s, "perturbation")
            succ.append(slope([float(t["acquired"]) for t in g]))
            mvs.append(slope([t["moves_used"] for t in g]))
        ms, ts, ns = ttest_1samp(succ)
        mm, tm, nm = ttest_1samp(mvs)
        res[arm] = {"success_slope": succ, "moves_slope": mvs}
        print(f"  {fmt_arm(arm):>17}  n={ns:2d}  success {ms:+.5f} (t={ts:+.2f})   "
              f"moves {mm:+.4f} (t={tm:+.2f})")

    # ---------------------------------------------- paired arm contrasts, within (draw, repeat)
    print("\nPAIRED arm contrasts on the recovery slope (same targets, same population,")
    print("matched severity; one difference per (draw, repeat) cell)")
    key = {(s["draw"], s["repeat"], s["arm"]): s for s in S}
    cells = sorted({(s["draw"], s["repeat"]) for s in S})
    pert_arms = [x for x in arms if x is not None]
    for i, a1 in enumerate(pert_arms):
        for a2 in pert_arms[i + 1:]:
            d_s, d_m = [], []
            for c in cells:
                s1, s2 = key.get((*c, a1)), key.get((*c, a2))
                if not (s1 and s2):
                    continue
                g1, g2 = block(s1, "perturbation"), block(s2, "perturbation")
                d_s.append(slope([float(t["acquired"]) for t in g1])
                           - slope([float(t["acquired"]) for t in g2]))
                d_m.append(slope([t["moves_used"] for t in g1])
                           - slope([t["moves_used"] for t in g2]))
            ms, ts, n = ttest_1samp(d_s)
            mm, tm, _ = ttest_1samp(d_m)
            print(f"  {fmt_arm(a1)} minus {fmt_arm(a2)}:  n={n:2d}  "
                  f"success slope {ms:+.5f} (t={ts:+.2f})   moves slope {mm:+.4f} (t={tm:+.2f})")

    # ------------------------------------------------------------ impairment and washout
    print("\nIMPAIRMENT and WASHOUT (paired within cell, against that cell's own control)")
    for arm in pert_arms:
        imp, wash = [], []
        for c in cells:
            s1, s0 = key.get((*c, arm)), key.get((*c, None))
            if not (s1 and s0):
                continue
            p1 = np.median([t["moves_used"] for t in block(s1, "perturbation")])
            b1 = np.median([t["moves_used"] for t in block(s1, "baseline")])
            w1 = np.median([t["moves_used"] for t in block(s1, "washout")])
            imp.append(p1 - b1)
            wash.append(w1 - b1)
        mi, ti, n = ttest_1samp(imp)
        mw, tw, _ = ttest_1samp(wash)
        print(f"  {fmt_arm(arm):>17}  n={n:2d}  perturbation minus baseline {mi:+.1f} moves "
              f"(t={ti:+.2f})   washout minus baseline {mw:+.1f} (t={tw:+.2f})")

    # ------------------------------------------------------------------- per-draw split
    print("\nPER DRAW, so a two-draw design never hides a draw effect")
    for dr in draws:
        for arm in pert_arms:
            g = [s for s in S if s["draw"] == dr and s["arm"] == arm]
            if not g:
                continue
            sl = [slope([float(t["acquired"]) for t in block(s, "perturbation")]) for s in g]
            m, t_, n = ttest_1samp(sl)
            print(f"  draw {dr}  {fmt_arm(arm):>17}  n={n:2d}  success slope {m:+.5f} (t={t_:+.2f})")

    out = Path(a.out or (ROOT / a.dir / "analysis_v3.json"))
    out.write_text(json.dumps(
        {"n_sessions": len(S), "draws": draws,
         "arms": [("control" if x is None else x) for x in arms],
         "stopped_early": [{"session": s["session"], "arm": s["arm"], "why": s["stopped"]}
                           for s in stopped],
         "recovery": {("control" if k is None else str(k)): v for k, v in res.items()}},
        indent=2, default=float))
    print(f"\n[v3] wrote {out}")


if __name__ == "__main__":
    main()
