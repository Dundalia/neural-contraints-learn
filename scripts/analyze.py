#!/usr/bin/env python
"""Session analysis (docs/DESIGN.md §4.7, §5).

``--mode baseline`` (G3): does closed-loop control work at all, and what does a session cost?
Freezes the geometry numbers §7 leaves provisional.

``--mode blocks`` (G4 / full): the learning metric per session, the exploration measures that
decide whether a session is interpretable at all, and the alignment contrast.

    python scripts/analyze.py --dir outputs/g3 --mode baseline
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nc.metrics import ANGLE_KEYS, PAPER_KEYS, Bin, amount_of_learning, zscore_bins  # noqa: E402


def load(d: Path):
    out = []
    for f in sorted(d.glob("sess_*.jsonl")):
        turns = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        tf = d / (f.stem + "_trials.json")
        trials = json.loads(tf.read_text()) if tf.exists() else []
        out.append({"idx": f.stem, "turns": turns, "trials": trials})
    return out


def exploration(turns):
    """§4.7: a block where the model never varied its emissions is not evidence about
    learnability. These are logged as first-class, not as diagnostics."""
    reps = [t["reply"].strip() for t in turns if not t.get("invalid")]
    if not reps:
        return {}
    syms = Counter(s for r in reps for s in r.split())
    tot = sum(syms.values()) or 1
    ent = -sum((c / tot) * math.log(c / tot) for c in syms.values())
    dup = sum(1 for a, b in zip(reps, reps[1:]) if a == b) / max(len(reps) - 1, 1)
    return {"n_turns": len(reps), "distinct_replies": len(set(reps)),
            "distinct_frac": len(set(reps)) / len(reps),
            "symbol_entropy_nats": ent, "max_entropy": math.log(len(syms) or 1),
            "duplicate_consecutive_frac": dup,
            "mean_reply_len": float(np.mean([len(r.split()) for r in reps]))}


def trial_bins(trials, bin_size):
    out = []
    for s in range(0, len(trials) - bin_size + 1, bin_size):
        ch = trials[s: s + bin_size]
        errs = [e for t in ch for e in t.get("angular_errors", [])]
        out.append(Bin(success_rate=float(np.mean([t["acquired"] for t in ch])),
                       mean_moves=float(np.mean([t["moves_used"] for t in ch])),
                       mean_angular_error=float(np.degrees(np.mean(errs))) if errs else float("nan"),
                       n=len(ch)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--mode", choices=["baseline", "blocks"], default="baseline")
    p.add_argument("--bin", type=int, default=8)
    a = p.parse_args()
    d = Path(a.dir)
    sess = load(d)
    if not sess:
        print(f"[analyze] no sessions in {d}"); return

    print(f"[analyze] {len(sess)} sessions in {d}")
    if a.mode == "baseline":
        rows = []
        for s in sess:
            tr = s["trials"]
            if not tr:
                print(f"  {s['idx']}: no completed trials"); continue
            mv = [t["moves_used"] for t in tr]
            errs = [e for t in tr for e in t.get("angular_errors", [])]
            ex = exploration(s["turns"])
            inval = sum(1 for t in s["turns"] if t.get("invalid")) / max(len(s["turns"]), 1)
            rows.append({"idx": s["idx"], "trials": len(tr),
                         "success": float(np.mean([t["acquired"] for t in tr])),
                         "moves_med": float(np.median(mv)),
                         "moves_iqr": float(np.percentile(mv, 75) - np.percentile(mv, 25)),
                         "ang_med_deg": float(np.degrees(np.median(errs))) if errs else float("nan"),
                         "invalid_frac": inval, "turns": len(s["turns"]), **ex})
            r = rows[-1]
            print(f"  {s['idx']}: trials {r['trials']:3d}  success {r['success']:.2f}  "
                  f"moves med {r['moves_med']:.1f} (IQR {r['moves_iqr']:.1f})  "
                  f"ang {r['ang_med_deg']:.1f}deg  invalid {r['invalid_frac']:.2f}  "
                  f"distinct {r.get('distinct_frac', float('nan')):.2f}  "
                  f"dup {r.get('duplicate_consecutive_frac', float('nan')):.2f}")
        if rows:
            g = lambda k: float(np.mean([r[k] for r in rows if k in r and np.isfinite(r[k])]))  # noqa: E731
            gates = {"success_ge_0.70": g("success") >= 0.70,
                     "moves_med_le_12": g("moves_med") <= 12.0,
                     "moves_iqr_ge_3": g("moves_iqr") >= 3.0}
            print(f"\n[analyze] pooled: success {g('success'):.2f}  moves med {g('moves_med'):.1f}"
                  f"  IQR {g('moves_iqr'):.1f}  ang {g('ang_med_deg'):.1f}deg"
                  f"  turns/session {g('turns'):.0f}")
            print(f"[analyze] G3 gates {gates}  G3_PASS={all(gates.values())}")
            (d / "summary.json").write_text(json.dumps(
                {"rows": rows, "gates": gates, "G3_PASS": all(gates.values())},
                indent=2, default=float))
        return

    # ---- blocks mode ----
    rows = []
    for s in sess:
        tr = s["trials"]
        blocks = {}
        for t in tr:
            blocks.setdefault(t["block"], []).append(t)
        if "perturbation" not in blocks or "baseline" not in blocks:
            continue
        al = next((t["alignment"] for t in blocks["perturbation"] if t.get("alignment")), None)
        allb = trial_bins(blocks.get("baseline", []) + blocks["perturbation"]
                          + blocks.get("washout", []), a.bin)
        nb = len(trial_bins(blocks.get("baseline", []), a.bin))
        npb = len(trial_bins(blocks["perturbation"], a.bin))
        for keys, name in ((PAPER_KEYS, "paper"), (ANGLE_KEYS, "angle")):
            Z = zscore_bins(allb, keys)
            if nb < 1 or npb < 2:
                continue
            L = amount_of_learning(Z[:nb].mean(0), Z[nb:nb + npb])
            rows.append({"idx": s["idx"], "alignment": al, "metric": name,
                         "l_session": L.l_session, "tail_mean": L.tail_mean,
                         "initial_impairment": L.initial_impairment,
                         **{f"ex_{k}": v for k, v in
                            exploration([t for t in s["turns"]
                                         if t["block"] == "perturbation"]).items()}})
    for r in rows:
        print(f"  {r['idx']} a={r['alignment']} [{r['metric']}]  L={r['l_session']:.2f}  "
              f"tail={r['tail_mean']:.2f}  impair={r['initial_impairment']:.2f}  "
              f"distinct={r.get('ex_distinct_frac', float('nan')):.2f}")
    if rows:
        for name in ("paper", "angle"):
            g = [r for r in rows if r["metric"] == name]
            for al in sorted({r["alignment"] for r in g if r["alignment"]}):
                v = [r["l_session"] for r in g if r["alignment"] == al]
                print(f"[analyze] {name}: a={al}  L mean {np.mean(v):.3f}  n={len(v)}")
        (d / "summary.json").write_text(json.dumps(rows, indent=2, default=float))


if __name__ == "__main__":
    main()
