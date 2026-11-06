#!/usr/bin/env python
"""Rotation x alignment grid: the dose-response curve, and the hypothesis test it enables.

TWO QUESTIONS, IN ORDER.

1. Does EFFECTIVE ROTATION collapse the grid onto one curve? Effective rotation is the angle
   between each decoded step and the canonical direction of the letter that produced it, i.e. the
   perturbation as the model experiences it. If the 23 cells, which differ in both nominal rotation
   and alignment, fall on a single monotone curve against it, then it is the difficulty control the
   screen has been missing. Severity (what the screen actually matches on) measured -0.27 against
   impairment on the previous run where effective rotation measured -0.80.

2. Does ALIGNMENT add anything once effective rotation is held fixed? This is the paper's actual
   hypothesis and no run before this one could test it, because every previous design let effective
   rotation vary freely across arms. The grid is triangular, so different alignments reach similar
   effective rotations, which is what makes the comparison possible. The test here is deliberately
   conservative: fit success against effective rotation ALONE, then ask whether the residuals carry
   any alignment signal. If they do not, alignment adds nothing beyond how much rotation it happened
   to produce, and the manifold claim has no support in this data.

Writes artifacts/v5_grid.json so the figure code reads one source of truth.

    python scripts/analyze_grid.py --dir outputs/v5
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIRS = {"r": (1.0, 0.0), "l": (-1.0, 0.0), "u": (0.0, 1.0), "d": (0.0, -1.0)}
BUDGET_FIELD = "moves_used"


def effective_rotation(jsonl: Path, block="perturbation"):
    """Mean angle between each applied step and the letter that caused it, in degrees.

    Move j applies the state that generated letter j (DESIGN 7.2d), so the displacement
    path[j+1]-path[j] pairs with reply.split()[j].
    """
    out = []
    if not jsonl.exists():
        return float("nan"), 0
    for line in jsonl.open():
        try:
            r = json.loads(line)
        except Exception:                                            # noqa: BLE001
            continue
        if r.get("block") != block:
            continue
        p, rep = r.get("path"), (r.get("reply") or "").split()
        if not p or len(p) < 2:
            continue
        P = np.asarray(p, dtype=float)
        for j in range(len(P) - 1):
            if j >= len(rep) or rep[j] not in DIRS:
                continue
            v = P[j + 1] - P[j]
            n = float(np.linalg.norm(v))
            if n < 1e-9:
                continue
            c = float(np.dot(v / n, DIRS[rep[j]]))
            out.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    return (float(np.mean(out)) if out else float("nan")), len(out)


def stopped_reason(jsonl: Path):
    why = None
    if not jsonl.exists():
        return None
    for line in jsonl.open():
        try:
            r = json.loads(line)
        except Exception:                                            # noqa: BLE001
            continue
        if r.get("stuck_abort"):
            why = "stuck"
        elif r.get("repeat_abort"):
            why = "repetition"
        elif r.get("turn_cap"):
            why = "turn cap"
    return why


def load(base):
    rows = []
    for d in sorted(glob.glob(str(ROOT / base / "*"))):
        mp = Path(d) / "meta.json"
        if not mp.exists():
            continue
        meta = {m["session"]: m for m in json.loads(mp.read_text())}
        for f in sorted(glob.glob(f"{d}/sess_*_trials.json")):
            i = int(Path(f).stem.split("_")[1])
            m = meta.get(i)
            if m is None:
                continue
            q = m.get("perturbation") or {}
            al_raw = q.get("alignment")
            tr = json.loads(Path(f).read_text())
            g = [t for t in tr if t["block"] == "perturbation"]
            if not g:
                continue
            jf = Path(d) / f"sess_{i:03d}.jsonl"
            eff, nmov = effective_rotation(jf)
            base_g = [t for t in tr if t["block"] == "baseline"]
            wash = [t for t in tr if t["block"] == "washout"]
            rows.append({
                "dir": Path(d).name, "session": i, "seed": m.get("repeat"),
                "alignment": None if al_raw is None else round(float(al_raw), 2),
                "theta": m.get("cell_theta_deg"),
                "severity": q.get("dev_deg"), "mag": q.get("mag"),
                "rho_min": q.get("rho_min"), "gap_deg": q.get("gap_deg"),
                "effective": eff, "n_moves_measured": nmov,
                "success": float(np.mean([t["acquired"] for t in g])),
                "moves_med": float(np.median([t[BUDGET_FIELD] for t in g])),
                "n_trials": len(g),
                "baseline_success": float(np.mean([t["acquired"] for t in base_g])) if base_g else None,
                "washout_success": float(np.mean([t["acquired"] for t in wash])) if wash else None,
                "stopped": stopped_reason(jf),
            })
    return rows


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/v5")
    ap.add_argument("--min-trials", type=int, default=40,
                   help="drop sessions truncated below this many perturbation trials; they are "
                        "reported separately rather than pooled")
    a = ap.parse_args()
    rows = load(a.dir)
    if not rows:
        raise SystemExit(f"no sessions under {a.dir}")

    pert = [r for r in rows if r["alignment"] is not None]
    ctrl = [r for r in rows if r["alignment"] is None]
    short = [r for r in pert if r["n_trials"] < a.min_trials]
    use = [r for r in pert if r["n_trials"] >= a.min_trials]
    print(f"[grid] {len(rows)} sessions: {len(pert)} perturbed, {len(ctrl)} control")
    if ctrl:
        print(f"[grid] control success {np.mean([r['success'] for r in ctrl]):.3f} "
              f"(n={len(ctrl)})")
    if short:
        print(f"[grid] {len(short)} perturbed sessions truncated below {a.min_trials} trials, "
              f"excluded from the curve and listed at the end")

    # ---- per cell
    cells = sorted({(r["alignment"], r["theta"]) for r in use},
                   key=lambda c: (-c[0], c[1]))
    print(f"\n{'align':>6} {'theta':>6} {'seeds':>6} {'severity':>9} {'EFFECTIVE':>10} "
          f"{'success':>8} {'moves':>7}")
    per_cell = []
    for al, th in cells:
        g = [r for r in use if r["alignment"] == al and r["theta"] == th]
        e = float(np.mean([r["effective"] for r in g]))
        s = float(np.mean([r["success"] for r in g]))
        per_cell.append({"alignment": al, "theta": th, "seeds": len(g), "effective": e,
                         "success": s, "severity": float(np.mean([r["severity"] for r in g])),
                         "moves_med": float(np.mean([r["moves_med"] for r in g]))})
        print(f"{al:6.2f} {th:6.0f} {len(g):6d} {np.mean([r['severity'] for r in g]):9.1f} "
              f"{e:10.1f} {s:8.3f} {np.mean([r['moves_med'] for r in g]):7.1f}")

    # ---- question 1: what orders impairment?
    print("\nQ1  what predicts success in the perturbation block? (rank correlation, per session)")
    for k, lab in (("effective", "EFFECTIVE rotation, the model's own experience"),
                   ("severity", "severity, what the screen matches on"),
                   ("theta", "nominal rotation theta"),
                   ("alignment", "alignment, the hypothesis variable"),
                   ("mag", "speed ratio"), ("rho_min", "reachability rho_min"),
                   ("gap_deg", "plane-spanning gap")):
        print(f"  {lab:48s} {spearman([r[k] for r in use], [r['success'] for r in use]):+.2f}")

    # ---- question 2: does alignment add anything at matched effective rotation?
    print("\nQ2  does alignment add anything once effective rotation is held fixed?")
    x = np.array([r["effective"] for r in use], float)
    y = np.array([r["success"] for r in use], float)
    al = np.array([r["alignment"] for r in use], float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, al = x[ok], y[ok], al[ok]
    if len(x) >= 8:
        # a cubic in effective rotation is flexible enough to absorb the curve's shape without
        # being able to stand in for alignment, which is not a function of it
        A = np.column_stack([np.ones_like(x), x, x ** 2, x ** 3])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        r_al = spearman(al, resid)
        ss = 1 - float((resid ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
        print(f"  effective rotation alone explains R2 = {ss:.3f} of the success variance")
        print(f"  residual rank correlation with alignment: {r_al:+.2f}")
        A2 = np.column_stack([A, al])
        coef2, *_ = np.linalg.lstsq(A2, y, rcond=None)
        r2b = 1 - float((((y - A2 @ coef2) ** 2).sum()) / max(((y - y.mean()) ** 2).sum(), 1e-12))
        print(f"  adding alignment as a term raises it to R2 = {r2b:.3f} "
              f"(gain {r2b - ss:+.3f})")
        # Do NOT presuppose the answer here. An earlier version of this script printed "a
        # residual correlation near zero means alignment adds nothing", which would have narrated
        # a null regardless of what the numbers said.
        verdict = ("alignment carries signal beyond effective rotation"
                   if abs(r_al) >= 0.25 and (r2b - ss) >= 0.02 else
                   "alignment adds nothing detectable beyond effective rotation")
        print(f"  READ: {verdict}")
        print("  the sign matters: a POSITIVE residual correlation means that at matched effective")
        print("  rotation, HIGHER alignment (more within-manifold) does better, which is the")
        print("  direction the hypothesis predicts")

    if short:
        print("\ntruncated sessions, reported not pooled:")
        for r in short:
            print(f"  {r['dir']} session {r['session']}: {r['n_trials']} trials, "
                  f"stopped by {r['stopped']}, alignment {r['alignment']}, theta {r['theta']}")

    out = ROOT / "artifacts/v5_grid.json"
    out.write_text(json.dumps({"sessions": rows, "cells": per_cell,
                               "n_truncated": len(short)}, indent=2, default=float))
    print(f"\n[grid] wrote {out}")


if __name__ == "__main__":
    main()
