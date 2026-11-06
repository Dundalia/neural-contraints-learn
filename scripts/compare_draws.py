#!/usr/bin/env python
"""Does the v5 alignment effect replicate across coordinate populations and across models?

THE CLAIM UNDER TEST. On v5 (Qwen3.5-9B, draw 16, 128 usable sessions) effective rotation ordered
impairment at rank -0.79, and at MATCHED effective rotation higher alignment still did better:
residual rank correlation +0.49, and positive in all five bands. That is the paper's hypothesis. But
it is n=1 in the unit that matters, because a single draw fixes the 100 coordinates, and v4 already
showed the population effect can exceed the arm effect. This script reads every sweep that exists
and reports the same two numbers for each, so replication is a comparison and not an impression.

THE MATCHED-BAND TEST IS THE PRIMARY ONE. The grid is triangular (alignment >= cos^2 theta), so
alignment and effective rotation are correlated by construction, and a cubic that mis-fits the curve
would leak curvature into the residual and look like an alignment effect. Inside a narrow band of
effective rotation the curve is locally flat, so a within-band alignment-success relation cannot be
curvature. The cubic residual is reported alongside it as the weaker, model-dependent version.

BANDS ARE FIXED, NOT FITTED, so the same band means the same thing in every sweep. A band is only
reported when it holds at least MIN_N sessions spanning at least two alignments; anything else is
printed as thin rather than silently pooled or silently dropped.

    python scripts/compare_draws.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_grid import load, spearman                                   # noqa: E402

BANDS = [(35, 50), (50, 62), (62, 78), (78, 90), (90, 115)]
MIN_N = 6
MIN_TRIALS = 40

# every sweep this campaign can produce, with the provenance that makes it a distinct unit
# The DECODER is part of a sweep's identity. Everything before 2026-08-18 ran a readout fit to the
# forced next letter, which the 2026-08-11 label change had superseded and which scores BELOW the
# prompt-only baseline on its own label; the gate had certified an intent-fit readout the loop never
# used. Those sweeps measure a decoder the model could barely steer, so their alignment numbers
# cannot be pooled with the corrected ones and the table has to say which is which.
SWEEPS = [
    ("outputs/v5",  "Qwen3.5-9B", "pop 16", "forced letter", "SUPERSEDED: the original result"),
    ("outputs/v6q", "Qwen3.5-9B", "pop 14", "forced letter", "SUPERSEDED: second population"),
]


def discover():
    out = list(SWEEPS)
    for d in sorted(ROOT.glob("outputs/v6g_d*")):
        out.append((f"outputs/{d.name}", "gemma-4-31B-it", f"pop {d.name.split('_d')[-1]}",
                    "forced letter", "SUPERSEDED: second model"))
    for d in sorted(ROOT.glob("outputs/v7q_d*")):
        out.append((f"outputs/{d.name}", "Qwen3.5-9B", f"pop {d.name.split('_d')[-1]}",
                    "model intent", "corrected decoder"))
    return [(p, m, dr, dec, note) for p, m, dr, dec, note in out if (ROOT / p).exists()]


def summarise(rows):
    pert = [r for r in rows if r["alignment"] is not None]
    ctrl = [r for r in rows if r["alignment"] is None]
    use = [r for r in pert if r["n_trials"] >= MIN_TRIALS and np.isfinite(r["effective"])]
    if len(use) < 8:
        return None
    x = np.array([r["effective"] for r in use], float)
    y = np.array([r["success"] for r in use], float)
    al = np.array([r["alignment"] for r in use], float)
    A = np.column_stack([np.ones_like(x), x, x ** 2, x ** 3])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    sst = max(float(((y - y.mean()) ** 2).sum()), 1e-12)
    r2 = 1 - float((resid ** 2).sum() / sst)
    A2 = np.column_stack([A, al])
    coef2, *_ = np.linalg.lstsq(A2, y, rcond=None)
    r2b = 1 - float((((y - A2 @ coef2) ** 2).sum()) / sst)
    bands = []
    for lo, hi in BANDS:
        g = [r for r in use if lo <= r["effective"] < hi]
        als = sorted({r["alignment"] for r in g})
        if len(g) < MIN_N or len(als) < 2:
            bands.append({"lo": lo, "hi": hi, "n": len(g), "thin": True, "corr": float("nan"),
                          "als": als})
            continue
        bands.append({"lo": lo, "hi": hi, "n": len(g), "thin": False, "als": als,
                      "corr": spearman([r["alignment"] for r in g],
                                       [r["success"] for r in g]),
                      "by_al": {a: float(np.mean([r["success"] for r in g
                                                  if r["alignment"] == a])) for a in als}})
    ok = [b for b in bands if not b["thin"] and np.isfinite(b["corr"])]
    return {
        "n_sessions": len(rows), "n_pert": len(pert), "n_use": len(use),
        "n_truncated": len(pert) - len([r for r in pert if r["n_trials"] >= MIN_TRIALS]),
        "control_success": (float(np.mean([r["success"] for r in ctrl])) if ctrl else None),
        "n_control": len(ctrl),
        "eff_range": (float(x.min()), float(x.max())),
        "success_range": (float(y.min()), float(y.max())),
        "rank_eff": spearman([r["effective"] for r in use], [r["success"] for r in use]),
        "rank_theta": spearman([r["theta"] for r in use], [r["success"] for r in use]),
        "rank_sev": spearman([r["severity"] for r in use], [r["success"] for r in use]),
        "rank_align": spearman([r["alignment"] for r in use], [r["success"] for r in use]),
        "r2_eff": r2, "r2_eff_align": r2b, "resid_align": spearman(al, resid),
        "bands": bands,
        "bands_reported": len(ok),
        "bands_positive": sum(1 for b in ok if b["corr"] > 0),
        "band_median": (float(np.median([b["corr"] for b in ok])) if ok else float("nan")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts/v6_replication.json"))
    a = ap.parse_args()

    found = discover()
    if not found:
        print("[compare] no sweeps on disk yet")
        return
    results = []
    for path, model, draw, dec, note in found:
        rows = load(path)
        s = summarise(rows) if rows else None
        results.append({"path": path, "model": model, "draw": draw, "decoder": dec,
                        "note": note, "summary": s, "n_rows": len(rows)})

    print("=" * 96)
    print("REPLICATION OF THE v5 ALIGNMENT EFFECT")
    print("=" * 96)
    for r in results:
        s = r["summary"]
        head = f"{r['model']}  {r['draw']}  [{r['decoder']} decoder]   ({r['note']})"
        print(f"\n{head}\n  {r['path']}")
        if s is None:
            print(f"  {r['n_rows']} sessions on disk, too few usable yet "
                  f"(need 8 with >= {MIN_TRIALS} trials) -- still running")
            continue
        print(f"  {s['n_use']} usable perturbed sessions "
              f"({s['n_pert']} perturbed, {s['n_truncated']} truncated, {s['n_control']} control"
              + (f", control success {s['control_success']:.3f}" if s['n_control'] else "") + ")")
        print(f"  effective rotation spans {s['eff_range'][0]:.0f} to {s['eff_range'][1]:.0f} deg, "
              f"success {s['success_range'][0]:.2f} to {s['success_range'][1]:.2f}")
        print(f"  Q1 rank corr with success:  effective {s['rank_eff']:+.2f}   "
              f"theta {s['rank_theta']:+.2f}   severity {s['rank_sev']:+.2f}   "
              f"alignment {s['rank_align']:+.2f}")
        print(f"  Q2 cubic:  R2(effective) {s['r2_eff']:.3f} -> +alignment {s['r2_eff_align']:.3f} "
              f"(gain {s['r2_eff_align'] - s['r2_eff']:+.3f}), residual corr {s['resid_align']:+.2f}")
        print(f"  Q2 MATCHED BANDS (the primary test):  "
              f"{s['bands_positive']}/{s['bands_reported']} positive, "
              f"median {s['band_median']:+.2f}")
        for b in s["bands"]:
            if b["thin"]:
                print(f"       {b['lo']:3d}-{b['hi']:3d} deg  n={b['n']:3d}  thin "
                      f"(alignments {b['als']}), not reported")
            else:
                by = "  ".join(f"a={k:.1f}:{v:.2f}" for k, v in sorted(b["by_al"].items(),
                                                                      reverse=True))
                print(f"       {b['lo']:3d}-{b['hi']:3d} deg  n={b['n']:3d}  "
                      f"corr {b['corr']:+.2f}   {by}")

    # ---- the replication verdict, stated from the measured numbers rather than assumed
    done = [r for r in results if r["summary"] is not None]
    print("\n" + "=" * 96)
    print("SIDE BY SIDE")
    print(f"  {'sweep (decoder)':34s} {'n':>4} {'eff':>6} {'resid':>7} {'gain':>7} """
          f"{'bands +':>9} {'median':>7}")
    for r in done:
        s = r["summary"]
        print(f"  {r['model'] + ' ' + r['draw'] + ' ' + r['decoder'][:6]:34s} """
              f"{s['n_use']:4d} {s['rank_eff']:+6.2f} "
              f"{s['resid_align']:+7.2f} {s['r2_eff_align'] - s['r2_eff']:+7.3f} "
              f"{str(s['bands_positive']) + '/' + str(s['bands_reported']):>9} "
              f"{s['band_median']:+7.2f}")
    if len(done) >= 2:
        pos = [r for r in done if r["summary"]["resid_align"] > 0]
        print(f"\n  {len(pos)} of {len(done)} sweeps show a POSITIVE alignment residual, i.e. the")
        print("  direction the manifold hypothesis predicts. Read the matched-band columns before")
        print("  the residual column: the residual depends on the cubic being right, the bands do not.")
        qwen = [r for r in done if r["model"].startswith("Qwen")]
        if len(qwen) >= 2:
            print(f"\n  ACROSS POPULATIONS (same model, different 100 coordinates): residuals "
                  f"{', '.join(format(r['summary']['resid_align'], '+.2f') for r in qwen)}")
        gem = [r for r in done if r["model"].startswith("gemma")]
        if gem and qwen:
            print(f"  ACROSS MODELS: Qwen "
                  f"{', '.join(format(r['summary']['resid_align'], '+.2f') for r in qwen)}"
                  f"  vs gemma "
                  f"{', '.join(format(r['summary']['resid_align'], '+.2f') for r in gem)}")
    else:
        print("\n  only one sweep is far enough along to compare; replication is still pending")

    Path(a.out).write_text(json.dumps(results, indent=2, default=float))
    print(f"\n[compare] wrote {a.out}")


if __name__ == "__main__":
    main()
