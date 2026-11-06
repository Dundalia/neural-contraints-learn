#!/usr/bin/env python
"""G1 stage 2: per-draw manifold + decoder + diagnostics (docs/DESIGN.md §4.1, §4.5, §7).

CPU only. Consumes the cached reads from `calibrate.py`, then for each of N_DRAWS random
100-coordinate populations and each captured depth: fits the manifold (FA, d=10), fits the
next-letter decoder, and reports the G1 diagnostics.

The open risk (§4.2) is that the decoder reads the TARGET rather than the model's intent. The
decoder is fit to predict the NEXT letter from the state before it, but calibration paths are
shortest paths, so the next letter is nearly a function of the direction to the target, and the
target coordinates sit in the prompt. A decoder can therefore score well by extracting the
target and rounding, in which case the marker sails to the target no matter what the model
writes and the model looks like a perfect controller while contributing nothing.

THERE IS CURRENTLY NO MEASURE OF THIS RISK HERE. Two attempts were withdrawn (DESIGN 4.5): the
ambiguous-vs-unambiguous angle contrast, which demanded the unachievable at positions where the
forced letter is unknowable, and a within-bin variance fraction, which is dominated by that same
label noise and cannot separate a target detector from a faithful readout of a deterministic
policy. Both are gone. Testing the risk needs the model's own next-token distribution as the
intent ground truth rather than the RNG-chosen forced letter, which is not implemented.

The ambiguous-vs-unambiguous contrast is still reported but is DESCRIPTIVE ONLY. At an ambiguous
position (diagonal required direction) two symbols are equally optimal and the randomised order
picked one, so the state genuinely cannot know which, and the best possible decoder points at the
target direction there. A large angle-to-letter at those positions is expected, not diagnostic.

    python scripts/fit_draws.py --reads artifacts/reads_Qwen3.5-9B.npz
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
from nc import manifold as mf                                             # noqa: E402
from nc.task import ALPHABETS                                            # noqa: E402

SYM = ALPHABETS["rlud"]


# Re-exported from nc.labels so every script scores identically; kept as module-level names
# because axes.py and check_label.py import them from here.
ang_err = lb.ang_err
r2 = lb.r2


# Re-exported from nc.labels so every script scores identically; kept as module-level names
# because axes.py and check_label.py import them from here.
ang_err = lb.ang_err
r2 = lb.r2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reads", required=True)
    p.add_argument("--draws", type=int, default=mf.N_DRAWS)
    p.add_argument("--draw-start", type=int, default=0,
                   help="first draw index of this shard. Coordinate draws are seeded 1000+index, "
                        "so a shard covers exactly the same populations it would in a single run")
    p.add_argument("--draw-count", type=int, default=None,
                   help="how many draws this shard fits; merge shards with merge_draw_shards.py")
    p.add_argument("--n-coords", type=int, default=mf.N_COORDS)
    p.add_argument("--holdout", type=float, default=0.25)
    p.add_argument("--label", choices=("intent", "letter"), default="intent",
                   help="'intent' = expected direction under the model's own next-letter "
                        "distribution (default since 2026-08-11); 'letter' = the forced next "
                        "letter, kept only for comparison with earlier runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    z = np.load(a.reads, allow_pickle=True)
    layers = [int(x) for x in z["layers"]]
    U_fit_all = z["U_fit"]                                   # (L, N, D) float16
    G = np.array([SYM[m] for m in z["fit_next_move"]], dtype=np.float64)
    # LABEL (DESIGN 4.2b, changed 2026-08-11). Default is the model's own intent direction:
    # the expected direction under its next-token distribution over the four letters. The forced
    # letter is kept selectable with --label letter, because every pre-2026-08-11 number used it
    # and the two must stay comparable. G_letter is always built so both baselines can be shown.
    def _unit(V):
        return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)

    G_letter = _unit(G)
    P = z["fit_letter_p"].astype(np.float64) if "fit_letter_p" in z.files else None
    if a.label == "intent":
        if P is None:
            print("[fit] FAILED: --label intent needs fit_letter_p in the reads artifact; "
                  "re-run calibrate.py (it captures the distributions since 2026-08-11)")
            sys.exit(4)
        order = [str(x) for x in z["letter_order"]]
        G, commit, good = lb.intent_direction(P, order)
        print(f"[fit] label = model intent direction; usable {good.sum()}/{len(good)}; "
              f"median commitment |E[dir]| {np.median(commit[good]):.3f}")
    else:
        good = np.isfinite(G_letter).all(1)
        G = G_letter
        print("[fit] label = forced next letter (the pre-2026-08-11 label)")

    REQ = z["fit_req"].astype(np.float64)
    AMB = z["fit_ambiguous"].astype(bool)
    BIN = z["fit_req_bin"].astype(int)
    N = U_fit_all.shape[1]

    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(np.flatnonzero(good))
    n_te = int(a.holdout * len(perm))
    te, rest = perm[:n_te], perm[n_te:]
    # The prompt-only geometry ridge needs its own alpha-selection slice. Carve it out of TRAIN,
    # never out of test, so the decoder and the baseline are scored on identical rows.
    n_va = int(0.2 * len(rest))
    va, tr = rest[:n_va], rest[n_va:]
    print(f"[fit] {N} samples, {len(layers)} depths, train {len(tr)} / val {len(va)} / test {n_te}")
    print(f"[fit] ambiguous fraction {AMB.mean():.2f}")

    # eligibility computed once, on the deepest layer's reads (quality screen, §4.1)
    elig = mf.eligible_coords(U_fit_all[-1][::7].astype(np.float64))
    print(f"[fit] eligible coordinates {len(elig)}/{U_fit_all.shape[2]} "
          f"({U_fit_all.shape[2]-len(elig)} massive-activation dims excluded)")

    # PROMPT-ONLY BASELINES on this label and this split. Every layer's R2 is reported against
    # `best_prompt_only`; a decoder below it has shown nothing (DESIGN 4.2b).
    base = lb.prompt_only_baselines(G, REQ, BIN, AMB, tr, va, te, P=P)
    print(f"[fit] prompt-only: unfitted target {base['unfitted_target_direction']:+.4f} | "
          f"bin-mean {base['bin_conditional_mean']:+.4f} | geometry ridge "
          f"{base['geometry_ridge']:+.4f}  ->  BEST {base['best_prompt_only']:+.4f} "
          f"({base['best_prompt_only_source']})", flush=True)

    rows = []
    lo = a.draw_start
    hi = lo + (a.draw_count if a.draw_count is not None else a.draws)
    if (lo, hi) != (0, a.draws):
        print(f"[fit] shard: draws {lo}..{hi - 1} of {a.draws}")
    for di in range(lo, hi):
        drng = np.random.default_rng(1000 + di)
        coords = mf.draw_coords(elig, a.n_coords, drng)
        for li, layer in enumerate(layers):
            U = U_fit_all[li][:, coords].astype(np.float64)
            try:
                ro = mf.fit_readout(U[tr], G[tr], d=mf.MANIFOLD_DIM)
                mf.set_gain(ro, U[tr], target_step=1.0)
            except Exception as e:                       # noqa: BLE001
                rows.append({"draw": di, "layer": layer, "error": str(e)[:80]}); continue
            V = ro.velocity(U[te])
            err = ang_err(V, G[te])
            rec = {
                "draw": di, "layer": layer, "depth": layer / max(int(z["num_layers"]) - 1, 1),
                "r2": r2(V, G[te]), "median_angle_deg": float(np.median(err)),
                "gain": ro.gain,
                # the only number that shows the readout beats the prompt
                "gap_over_prompt": float(r2(V, G[te]) - base["best_prompt_only"]),
            }
            if P is not None:
                rec["match_acc"] = lb.match_accuracy(V, P[te])
                rec["match_acc_ambiguous"] = lb.match_accuracy(V, P[te], AMB[te])
                dis = lb.to_letter(lb.unit(REQ)[te]) != np.argmax(P[te], axis=1)
                rec["match_acc_where_prompt_disagrees"] = lb.match_accuracy(V, P[te], dis)
            # intent vs target, at ambiguous positions
            m = AMB[te]
            if m.sum() > 30:
                e_letter = ang_err(V[m], G[te][m])
                e_target = ang_err(V[m], REQ[te][m])
                rec["amb_angle_to_letter"] = float(np.median(e_letter))
                rec["amb_angle_to_target"] = float(np.median(e_target))
                rec["intent_minus_target"] = rec["amb_angle_to_letter"] - rec["amb_angle_to_target"]
            if (~m).sum() > 30:
                rec["unamb_angle_deg"] = float(np.median(ang_err(V[~m], G[te][~m])))
            # per-direction spread
            per = [float(np.median(err[BIN[te] == b])) for b in range(8)
                   if (BIN[te] == b).sum() > 10]
            if per:
                rec["direction_spread_deg"] = max(per) - min(per)
            rows.append(rec)
        if di == 0:                                       # stability on the first draw only
            for li, layer in enumerate(layers):
                U = U_fit_all[li][:, coords].astype(np.float64)
                st = mf.manifold_stability(U, mf.MANIFOLD_DIM, np.random.default_rng(7))
                rows[li]["split_half_max_angle_deg"] = st["max_angle_deg"]
                rows[li]["split_half_mean_cos2"] = st["mean_cos2"]
        print(f"  draw {di:2d} done", flush=True)

    ok = [r for r in rows if "r2" in r]
    print(f"\n{'layer':>6} {'depth':>6} {'R2':>7} {'vs prompt':>10} {'match':>6} {'ang':>6} "
          f"{'unamb':>6} {'amb->letter':>12} {'amb->target':>12} {'dirspread':>10} "
          f"{'splithalf':>10}")
    best = None
    for layer in layers:
        g = [r for r in ok if r["layer"] == layer]
        if not g:
            continue
        f = lambda k: np.median([r[k] for r in g if k in r]) if any(k in r for r in g) else float("nan")  # noqa: E731
        print(f"{layer:6d} {g[0]['depth']:6.2f} {f('r2'):7.3f} {f('gap_over_prompt'):+10.3f} "
              f"{f('match_acc'):6.3f} {f('median_angle_deg'):6.1f} "
              f"{f('unamb_angle_deg'):6.1f} {f('amb_angle_to_letter'):12.1f} "
              f"{f('amb_angle_to_target'):12.1f} "
              f"{f('direction_spread_deg'):10.1f} {f('split_half_max_angle_deg'):10.1f}")
        key = f("r2")
        if np.isfinite(key) and (best is None or key > best[0]):
            best = (key, layer, f)
    if best is None:
        errs = [r.get("error") for r in rows if "error" in r][:3]
        print(f"\n[fit] FAILED: no layer produced a finite R2. First errors: {errs}")
        Path(a.out or str(a.reads).replace("reads_", "draws_").replace(".npz", ".json")).write_text(
            json.dumps({"G1_PASS": False, "error": "no finite R2 at any layer",
                        "first_errors": errs, "rows": rows}, indent=2, default=float))
        sys.exit(3)
    _, blayer, bf = best
    # ONE DEFINITIONAL GATE. Hand-picked magnitude cut-offs (R2 >= 0.30, direction spread
    # <= 15 deg) and a within-bin variance floor all used to live here; none was derived and all
    # were withdrawn (DESIGN 4.5). What is left is the floor that follows from the definition.
    #
    # READ R2 AGAINST A LABEL-NOISE CEILING, NOT AGAINST 1.0. Squared distance between unit
    # vectors is 2(1 - cos), and with the four directions balanced their mean is ~0, so
    # R2 ~= 2 * mean(cos of angular error) - 1. At an ambiguous position (72% of them) the forced
    # letter is chosen by OUR rng between two adjacent symbols, so the best possible prediction
    # is the bisector, 45 deg off whichever letter was forced. Mixing 72% at cos 0.707 with 28%
    # at cos 1.0 puts the ceiling for a PERFECT readout near 0.58. Judge layers against that.
    gates = {
        # R2 > 0: the decoder beats the constant predictor that always emits the mean direction.
        "r2_beats_mean_direction": bool(bf("r2") > 0),
        # AND it beats every prompt-only predictor. Without this the readout may be a target
        # detector, which would leave the model no control channel at all (DESIGN 4.2).
        "beats_prompt_only": bool(bf("r2") > base["best_prompt_only"]),
    }
    report = {"label": a.label, "prompt_only_baselines": base,
              "draw_range": [lo, hi],
              "reads": str(a.reads), "draws": a.draws, "n_coords": a.n_coords,
              "eligible": int(len(elig)), "excluded": int(U_fit_all.shape[2] - len(elig)),
              "best_layer": int(blayer), "gates": gates,
              "G1_PASS": bool(all(gates.values())),
              "logprob_median_fit": float(np.nanmedian(z["lp_fit"])),
              "logprob_median_val": float(np.nanmedian(z["lp_val"])),
              "rows": rows}
    out = Path(a.out or str(a.reads).replace("reads_", "draws_").replace(".npz", ".json"))
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[fit] best layer {blayer}  gates={gates}  G1_PASS={report['G1_PASS']}")
    print(f"[fit] -> {out}", flush=True)


if __name__ == "__main__":
    main()
