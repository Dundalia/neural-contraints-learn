#!/usr/bin/env python
"""G2 -- the kill gate (docs/DESIGN.md §4.6, §7.0). CPU only; consumes the cached reads.

Joint search over **(layer, theta, alignment level)**. A layer can decode well and admit no
usable perturbation, or vice versa, so both are chosen in one search. theta is NOT fixed:
severity is the quantity that must match across arms (the Fig-3a initial-impairment proxy) and
theta is only a means to it -- Sadtler screened into a severity window too and reported the
angles afterwards.

Each candidate must clear, in order:

  1. SEVERITY -- median PER-SAMPLE velocity-change angle in [20, 45] deg, magnitude ratio in
     [0.3, 1.5] of the intuitive median step. Per sample, not on per-direction means: means
     average away the private variance an off-manifold control space reads, and reported a
     genuinely off-manifold perturbation as 3 deg (§4.6).
  2. REACHABILITY -- the achievable per-move directions must positively span the plane (largest
     angular gap under 180 deg), since with a 4-symbol alphabet a single move cannot point
     diagonally and net direction comes from mixing; plus per-direction density >= RHO_MIN at a
     60 deg tolerance. Speed floor from the INTUITIVE median, never the candidate's own.
  3. EXTRAPOLATION -- rotations, unlike permutations, can demand activation values outside the
     observed range, which the paper's z-scoring existed to prevent.

Then candidates are matched ACROSS levels on severity, at one layer common to all levels, so a
low-alignment failure is unlearnability and not unreachability or a milder perturbation.

Staged for a 4-core box: the layer is chosen on a few draws, then the full screen runs there.

    python scripts/screen.py --reads artifacts/reads_Qwen3.5-9B.npz
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nc import labels as lb                                              # noqa: E402
from nc import manifold as mf                                            # noqa: E402
from nc.task import ALPHABETS, BoardConfig                               # noqa: E402

SYM = ALPHABETS["rlud"]
RHO_TOL = math.radians(60.0)
MAX_GAP_DEG = 180.0
SEVERITY_TARGET = math.radians(32.0)          # centre of the window; used for matching


def severity(readout, U, M2P, ref_speed):
    v0, vp = readout.velocity(U), readout.velocity(U, M2P)
    n0, npp = np.linalg.norm(v0, axis=1), np.linalg.norm(vp, axis=1)
    k = (n0 > 1e-9) & (npp > 1e-9)
    if k.sum() < 32:
        return None
    ang = np.arccos(np.clip((v0[k] * vp[k]).sum(1) / (n0[k] * npp[k]), -1, 1))
    return {"dev": float(np.median(ang)), "mag": float(np.median(npp[k] / max(ref_speed, 1e-12)))}


def spans_plane(readout, U, M2P, ref_speed, frac=0.6):
    V = readout.velocity(U, M2P)
    sp = np.linalg.norm(V, axis=1)
    V = V[sp >= frac * ref_speed]
    if len(V) < 8:
        return 360.0
    ang = np.sort(np.arctan2(V[:, 1], V[:, 0]) % (2 * np.pi))
    return float(np.degrees(np.diff(np.r_[ang, ang[0] + 2 * np.pi]).max()))


def extrapolation_ok_frac(readout, U, M2P):
    Uz = readout.z(U)
    lo, hi = np.percentile(Uz, 0.1, axis=0), np.percentile(Uz, 99.9, axis=0)
    Q = np.linalg.qr(np.asarray(M2P).T)[0]
    span = ((hi - lo)[:, None] * np.abs(Q)).sum(0)
    return float(np.mean(np.all(np.abs(Uz @ Q) <= span[None, :], axis=1)))


def fit_layer(z, coords, li, d):
    Ufit = z["U_fit"][li][:, coords].astype(np.float64)
    # same label as the live loop and the gate (nc.labels.readout_label, DESIGN 7.2c): the
    # perturbations rotate THIS control space, so a different label here would screen
    # severity and reachability for a decoder that never runs
    G, usable = lb.readout_label(z, "fit")
    ro = mf.fit_readout(Ufit[usable], G[usable], d=d)
    mf.set_gain(ro, Ufit[usable], target_step=1.0)
    Uem = z["U_val"][li][:, coords].astype(np.float64)
    return ro, Ufit, Uem


def screen_layer(z, coords, li, dirs, n_cand, rng, d):
    """All admissible candidates at one layer, across levels and their admissible thetas."""
    ro, Ufit, Uem = fit_layer(z, coords, li, d)
    Mb = mf.manifold_basis(ro.beta)
    ref = mf.reference_speed(ro, Ufit)
    base_rho = mf.solution_density(Uem, ro, ro.M2, dirs, ref, max_angle=RHO_TOL)
    base_gap = spans_plane(ro, Uem, ro.M2, ref)
    if base_rho.min() < mf.RHO_MIN or base_gap >= MAX_GAP_DEG:
        return None, {"intuitive_rho_min": float(base_rho.min()), "intuitive_gap": base_gap,
                      "abort": "intuitive decoder fails its own screen"}
    per_level = {}
    for a in mf.ALIGNMENT_GRID:
        rows = []
        for th in mf.admissible_thetas(a):
            for _ in range(n_cand):
                M2P = mf.build_perturbation(ro.M2, Mb, th, a, rng)
                sv = severity(ro, Uem, M2P, ref)
                if sv is None:
                    continue
                rho = mf.solution_density(Uem, ro, M2P, dirs, ref, max_angle=RHO_TOL)
                gap = spans_plane(ro, Uem, M2P, ref)
                ok = (mf.SEVERITY_WINDOW[0] <= sv["dev"] <= mf.SEVERITY_WINDOW[1]
                      and 0.3 <= sv["mag"] <= 1.5
                      and rho.min() >= mf.RHO_MIN and gap < MAX_GAP_DEG)
                rows.append({"a": a, "theta_deg": math.degrees(th),
                             "dev_deg": math.degrees(sv["dev"]), "mag": sv["mag"],
                             "rho_min": float(rho.min()), "gap_deg": gap, "pass": bool(ok),
                             "M2P": M2P})
        per_level[a] = rows
    info = {"intuitive_rho_min": float(base_rho.min()), "intuitive_gap": base_gap,
            "ref_speed": ref}
    return per_level, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reads", required=True)
    p.add_argument("--draws", type=int, default=mf.N_DRAWS)
    p.add_argument("--scout-draws", type=int, default=3)
    p.add_argument("--n-cand", type=int, default=40)
    p.add_argument("--match-tol", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fit-report", default=None,
                   help="draws_*.json from fit_draws.py; REQUIRED for the design's layer rule")
    p.add_argument("--layer", type=int, default=None,
                   help="override the selected layer; records itself in the report. Use when the "
                        "operating layer is a decision rather than an outcome of the rule below")
    a = p.parse_args()

    z = np.load(a.reads, allow_pickle=True)
    layers = [int(x) for x in z["layers"]]
    cfg = BoardConfig()
    dirs = np.array(cfg.bin_centres(), dtype=np.float64)
    elig = mf.eligible_coords(z["U_fit"][-1][::7].astype(np.float64))
    print(f"[screen] levels={mf.ALIGNMENT_GRID}  thetas="
          f"{[round(math.degrees(t)) for t in mf.THETA_GRID]}  rho_min={mf.RHO_MIN}")
    for lv in mf.ALIGNMENT_GRID:
        print(f"   a={lv:<4} admissible thetas: "
              f"{[round(math.degrees(t)) for t in mf.admissible_thetas(lv)] or 'NONE -> drop'}")

    # ---- stage 1: choose the layer on a few draws ----
    print(f"\n[screen] stage 1: layer scout on {a.scout_draws} draws")
    scores = {}
    for li, layer in enumerate(layers):
        n_ok, sev = 0, []
        for di in range(a.scout_draws):
            coords = mf.draw_coords(elig, mf.N_COORDS, np.random.default_rng(1000 + di))
            per, info = screen_layer(z, coords, li, dirs, max(8, a.n_cand // 4),
                                     np.random.default_rng(a.seed + di), mf.MANIFOLD_DIM)
            if per is None:
                continue
            levels_ok = sum(1 for lv in mf.ALIGNMENT_GRID
                            if any(r["pass"] for r in per.get(lv, [])))
            n_ok += levels_ok
            sev.extend(r["dev_deg"] for lv in per for r in per[lv] if r["pass"])
        scores[layer] = n_ok
        print(f"  layer {layer:3d}: levels-with-candidates {n_ok}/"
              f"{len(mf.ALIGNMENT_GRID)*a.scout_draws}"
              f"  sev med {np.median(sev):.1f}deg" if sev else
              f"  layer {layer:3d}: none", flush=True)
    # LAYER SELECTION (DESIGN 4.1): "maximise held-out decodability SUBJECT TO admitting
    # matched candidates at every alignment level". Candidate count is the CONSTRAINT, R2 the
    # OBJECTIVE. An earlier version picked purely by candidate count, which chose layer 14 on
    # Qwen3.5-9B -- a layer whose decoder has R2 = -0.18, i.e. worse than useless.
    #
    # NO SECOND CRITERION. A within-bin variance floor used to sit alongside R2 here and it is
    # what selected layer 21 on the 2026-08-10 run. It was withdrawn (DESIGN 4.5) because it
    # measured label noise rather than whether the readout tracks the model, so this rule now has
    # nothing to say about the target-detector risk of 4.2. It picks the most decodable layer that
    # admits candidates and no more than that. Use --layer to override when the operating layer
    # is a decision rather than an outcome.
    fit = {}
    if a.fit_report:
        import collections
        rep = json.loads(Path(a.fit_report).read_text())
        by = collections.defaultdict(list)
        for row in rep.get("rows", []):
            if "r2" in row:
                by[int(row["layer"])].append(row)
        for L, g in by.items():
            fit[L] = {k: float(np.median([x[k] for x in g if k in x]))
                      for k in ("r2", "direction_spread_deg")
                      if any(k in x for x in g)}
    need = (len(mf.ALIGNMENT_GRID) - 1) * a.scout_draws * 0.8   # lowest level may be unreachable
    admits = [L for L, s in scores.items() if s >= need]
    if fit:
        if admits:
            best_layer = max(admits, key=lambda L: fit.get(L, {}).get("r2", -9))
            why = "max R2 among layers admitting matched candidates (DESIGN 4.1)"
        else:
            best_layer = max(fit, key=lambda L: fit[L].get("r2", -9))
            why = ("NO layer admits matched candidates at enough levels; fell back to max R2 "
                   "-- REPORT THIS, it is a design-level conflict")
        if a.layer is not None:
            best_layer, why = a.layer, f"operator override --layer {a.layer} (rule would pick {best_layer})"
        for L in sorted(scores):
            fl = fit.get(L, {})
            print(f"  layer {L:3d}: cand {scores[L]:3d}  R2 {fl.get('r2', float('nan')):6.3f}"
                  f"{'   <-- chosen' if L == best_layer else ''}")
    else:
        best_layer = max(scores, key=lambda k: scores[k])
        why = "candidate count only (no --fit-report given; NOT the design's rule)"
    bli = layers.index(best_layer)
    print(f"[screen] chosen layer {best_layer}  ({why})")

    # ---- stage 2: full screen at the chosen layer, all draws ----
    print(f"\n[screen] stage 2: {a.draws} draws at layer {best_layer}")
    pool, report_rows = {}, []
    for di in range(a.draws):
        coords = mf.draw_coords(elig, mf.N_COORDS, np.random.default_rng(1000 + di))
        per, info = screen_layer(z, coords, bli, dirs, a.n_cand,
                                 np.random.default_rng(a.seed + 100 + di), mf.MANIFOLD_DIM)
        if per is None:
            report_rows.append({"draw": di, **info}); print(f"  draw {di}: ABORT"); continue
        chosen, miss = {}, []
        for lv in mf.ALIGNMENT_GRID:
            ok = [r for r in per[lv] if r["pass"]]
            if not ok:
                miss.append(lv); continue
            chosen[lv] = min(ok, key=lambda r: abs(math.radians(r["dev_deg"]) - SEVERITY_TARGET))
        report_rows.append({
            "draw": di, "missing_levels": miss,
            "chosen": {str(lv): {k: v for k, v in chosen[lv].items() if k != "M2P"}
                       for lv in chosen},
            **{k: v for k, v in info.items()}})
        if di == 0 and chosen:
            pool = chosen
        print(f"  draw {di:2d}: levels ok {len(chosen)}/{len(mf.ALIGNMENT_GRID)}"
              f"{'  missing ' + str(miss) if miss else ''}", flush=True)

    # Usable levels FIRST, then severity matching WITHIN them. Requiring every grid level to be
    # present before evaluating the match made "fully matched" unattainable whenever one level
    # was unreachable -- the design's rule is to truncate the grid, never to relax the windows.
    usable = [lv for lv in mf.ALIGNMENT_GRID
              if sum(1 for r in report_rows if str(lv) in r.get("chosen", {}))
              >= 0.8 * max(a.draws, 1)]
    n_matched, spreads, mag_spreads = 0, [], []
    for r in report_rows:
        ch = r.get("chosen", {})
        if not usable or any(str(lv) not in ch for lv in usable):
            r["matched"] = False
            continue
        devs = [ch[str(lv)]["dev_deg"] for lv in usable]
        mags = [ch[str(lv)]["mag"] for lv in usable]
        rel = (max(devs) - min(devs)) / max(np.mean(devs), 1e-9)
        r["severity_spread_rel"] = float(rel)
        r["mag_spread_rel"] = float((max(mags) - min(mags)) / max(np.mean(mags), 1e-9))
        r["matched"] = bool(rel <= a.match_tol)
        spreads.append(rel); mag_spreads.append(r["mag_spread_rel"])
        n_matched += int(r["matched"])
    report = {"reads": str(a.reads), "layer": int(best_layer), "layer_scores": scores,
              "layer_selection_rule": why, "layer_fit": fit,
              "layer_override": a.layer,
              "draws": a.draws, "usable_levels": usable,
              "n_draws_severity_matched": n_matched,
              "severity_spread_median": float(np.median(spreads)) if spreads else None,
              # NOT gated, but recorded: with severity matched, decoded SPEED still falls as
              # alignment falls (dry run: mag 0.93 -> 0.50). Low-alignment sessions therefore
              # have a slower marker, which reaches moves-to-acquire independently of the
              # angular perturbation. Flagged for a matching decision, not decided here.
              "mag_spread_median": float(np.median(mag_spreads)) if mag_spreads else None,
              "G2_PASS": bool(len(usable) >= 3 and n_matched >= 0.5 * max(a.draws, 1)),
              "rows": report_rows}
    out = Path(str(a.reads).replace("reads_", "screen_").replace(".npz", ".json"))
    out.write_text(json.dumps(report, indent=2, default=float))
    if pool:
        np.savez_compressed(str(out).replace(".json", ".npz"),
                            **{f"M2P_a{lv}": pool[lv]["M2P"] for lv in pool},
                            layer=best_layer)
    print(f"\n[screen] layer {best_layer}  usable levels {usable}  "
          f"severity-matched draws {n_matched}/{a.draws}  "
          f"sev spread med {report['severity_spread_median']}  "
          f"mag spread med {report['mag_spread_median']}")
    print(f"[screen] G2_PASS={report['G2_PASS']}  -> {out}", flush=True)


if __name__ == "__main__":
    main()
