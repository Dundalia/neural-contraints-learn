#!/usr/bin/env python
"""Two questions the G1 run left compressed (DESIGN 7.3). CPU only, cached reads only.

AXIS 1 -- how much does the 100-coordinate draw matter?
  a1  spread of R2 across the 20 existing draws, per layer, and a layer/draw variance split
  a2  is a good draw good everywhere? rank agreement of draws between layers
  a3  do two DISJOINT 100-coordinate sets decode the same thing? (the redundancy test)
  a4  R2 versus population size, 10 -> 800 coordinates

AXIS 2 -- what is the factor analysis actually doing, by draw and by depth?
  b1  shared versus private variance, and the loading spectrum
  b2  held-out FA log-likelihood versus d: is d = 10 defensible, and does it move with depth?
  b3  decoder R2 versus d: what the readout actually needs
  b4  split-half manifold stability (principal angles) across draws and layers

Writes artifacts/axes_<tag>.json incrementally, so a kill loses only the section in flight.

    python scripts/axes.py --reads artifacts/reads_Qwen3.5-9B.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nc import labels as lb                                              # noqa: E402
from nc import manifold as mf                                             # noqa: E402
from nc.task import ALPHABETS                                            # noqa: E402
from fit_draws import ang_err, r2                                        # noqa: E402

SYM = ALPHABETS["rlud"]
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)


def fa_parts(Utr: np.ndarray, d: int, seed: int = 0):
    """The FA half of fit_readout, exposed so the FA itself can be inspected and scored."""
    from sklearn.decomposition import FactorAnalysis
    mu, sd = Utr.mean(0), Utr.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    fa = FactorAnalysis(n_components=d, random_state=seed).fit((Utr - mu) / sd)
    return fa, mu, sd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reads", required=True)
    p.add_argument("--holdout", type=float, default=0.25)
    p.add_argument("--label", choices=("intent", "letter"), default="intent")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--draws-report", default=None, help="draws_*.json, for a1/a2")
    p.add_argument("--layers", type=int, nargs="*", default=[14, 21, 28],
                   help="subset used for the expensive sweeps")
    p.add_argument("--sections", nargs="*",
                   default=["a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4"])
    a = p.parse_args()

    z = np.load(a.reads, allow_pickle=True)
    layers = [int(x) for x in z["layers"]]
    U_all = z["U_fit"]
    # Label: the model's intent direction by default (DESIGN 7.2c), the forced letter on request
    # so the pre-2026-08-11 numbers stay reproducible.
    if a.label == "intent":
        if "fit_letter_p" not in z.files:
            print("[axes] FAILED: --label intent needs fit_letter_p; use a v2 reads artifact")
            sys.exit(4)
        G, _commit, _ok = lb.intent_direction(z["fit_letter_p"].astype(np.float64),
                                              [str(x) for x in z["letter_order"]])
    else:
        G = lb.unit(np.array([SYM[m] for m in z["fit_next_move"]], dtype=np.float64))
    print(f"[axes] label = {a.label}")
    N = U_all.shape[1]
    nl = int(z["num_layers"])

    # SAME split as fit_draws.py: one permutation, seed 0, 25% holdout.
    perm = np.random.default_rng(a.seed).permutation(N)
    nte = int(a.holdout * N)
    te, tr = perm[:nte], perm[nte:]
    elig = mf.eligible_coords(U_all[-1][::7].astype(np.float64))
    log(f"{N} samples, train {len(tr)} / test {nte}; {len(elig)} eligible coords; "
        f"layers {layers}")

    dest = Path(str(a.reads).replace("reads_", "axes_").replace(".npz", ".json"))
    # MERGE, do not clobber. Sections are meant to be runnable one at a time and a later run
    # must not wipe an earlier run's sections out of the file.
    out = json.loads(dest.read_text()) if dest.exists() else {}
    if out:
        log(f"merging into existing {dest.name} (sections {sorted(k for k in out if len(k) == 2)})")
    out.update({"reads": str(a.reads), "layers": layers, "num_layers": nl,
                "n_train": int(len(tr)), "n_test": int(nte), "sweep_layers": a.layers})

    def save():
        dest.write_text(json.dumps(out, indent=2, default=float))

    def U_of(li, coords):
        return U_all[li][:, coords].astype(np.float64)

    def fit_and_score(li, coords, d=mf.MANIFOLD_DIM):
        U = U_of(li, coords)
        ro = mf.fit_readout(U[tr], G[tr], d=d)
        V = ro.velocity(U[te])
        return ro, V, float(r2(V, G[te])), float(np.median(ang_err(V, G[te])))

    # ---------------------------------------------------------------- a1 + a2
    if ("a1" in a.sections or "a2" in a.sections) and a.draws_report:
        rep = json.loads(Path(a.draws_report).read_text())
        rows = [x for x in rep["rows"] if "r2" in x]
        per = {}
        for L in sorted({x["layer"] for x in rows}):
            v = np.array([x["r2"] for x in rows if x["layer"] == L], float)
            per[L] = {"n": int(v.size), "median": float(np.median(v)),
                      "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                      "min": float(v.min()), "max": float(v.max()),
                      "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))]}
        # two-way split of R2 variance: layer main effect vs draw main effect vs residual
        Ls = sorted(per)
        M = np.array([[x["r2"] for x in rows if x["layer"] == L] for L in Ls], float)  # (layer, draw)
        grand = M.mean()
        ss_tot = float(((M - grand) ** 2).sum())
        ss_layer = float(M.shape[1] * ((M.mean(1) - grand) ** 2).sum())
        ss_draw = float(M.shape[0] * ((M.mean(0) - grand) ** 2).sum())
        out["a1"] = {"per_layer": per,
                     "var_share_layer": ss_layer / ss_tot,
                     "var_share_draw": ss_draw / ss_tot,
                     "var_share_resid": 1 - (ss_layer + ss_draw) / ss_tot}
        log(f"a1 done: layer explains {out['a1']['var_share_layer']:.3f} of R2 variance, "
            f"draw {out['a1']['var_share_draw']:.3f}")

        # a2: does the draw ranking agree between layers? (Spearman on each layer pair)
        def rk(v):
            return np.argsort(np.argsort(v)).astype(float)

        cors = []
        for i in range(len(Ls)):
            for j in range(i + 1, len(Ls)):
                x, y = rk(M[i]), rk(M[j])
                x, y = x - x.mean(), y - y.mean()
                den = np.sqrt((x ** 2).sum() * (y ** 2).sum())
                cors.append(float((x * y).sum() / den) if den > 0 else float("nan"))
        deep = [Ls.index(L) for L in Ls if L >= 21]
        dcors = []
        for a_ in range(len(deep)):
            for b_ in range(a_ + 1, len(deep)):
                x, y = rk(M[deep[a_]]), rk(M[deep[b_]])
                x, y = x - x.mean(), y - y.mean()
                dcors.append(float((x * y).sum() /
                                   np.sqrt((x ** 2).sum() * (y ** 2).sum())))
        out["a2"] = {"draw_rank_corr_all_pairs_median": float(np.median(cors)),
                     "draw_rank_corr_all_pairs": cors,
                     "draw_rank_corr_deep_layers_median": float(np.median(dcors)) if dcors else None,
                     "note": "draws share seeds across layers, so the pairing is exact"}
        log(f"a2 done: median draw-rank agreement between layers "
            f"{out['a2']['draw_rank_corr_all_pairs_median']:+.2f}, "
            f"deep-only {out['a2']['draw_rank_corr_deep_layers_median']}")
        save()

    # -------------------------------------------------------------------- a3
    # Two DISJOINT 100-coordinate populations at the same layer. If they decode the same
    # velocity field, the computation is redundant across coordinates; if not, each draw
    # reads its own thing and the 20-draw median is averaging over different readouts.
    if "a3" in a.sections:
        res = []
        for L in a.layers:
            li = layers.index(L)
            for pair in range(6):
                rng = np.random.default_rng(7000 + pair)
                both = mf.draw_coords(elig, 2 * mf.N_COORDS, rng)
                cA, cB = both[:mf.N_COORDS], both[mf.N_COORDS:]
                _, VA, r2A, _ = fit_and_score(li, cA)
                _, VB, r2B, _ = fit_and_score(li, cB)
                agree = ang_err(VA, VB)                     # A vs B, not vs the label
                res.append({"layer": L, "pair": pair, "r2_A": r2A, "r2_B": r2B,
                            "median_angle_A_vs_B": float(np.median(agree)),
                            "frac_within_30deg": float((agree < 30).mean()),
                            "r2_of_mean_field": float(r2((VA + VB) / 2, G[te]))})
                log(f"a3 layer {L} pair {pair}: R2 {r2A:.3f}/{r2B:.3f}, "
                    f"A-vs-B median {res[-1]['median_angle_A_vs_B']:.1f}deg")
            save()
        out["a3"] = res
        save()

    # -------------------------------------------------------------------- a4
    # R2 versus population size. Two effects are folded together on purpose and named in
    # the writeup: more coordinates carry more signal AND estimate the FA better.
    if "a4" in a.sections:
        res = []
        for L in a.layers:
            li = layers.index(L)
            for ncoord in (10, 25, 50, 100, 200, 400, 800):
                for di in range(5):
                    coords = mf.draw_coords(elig, ncoord, np.random.default_rng(9000 + di))
                    d_eff = min(mf.MANIFOLD_DIM, ncoord - 1)
                    try:
                        _, _, rr, ang = fit_and_score(li, coords, d=d_eff)
                    except Exception as e:                            # noqa: BLE001
                        res.append({"layer": L, "n_coords": ncoord, "draw": di,
                                    "error": str(e)[:60]}); continue
                    res.append({"layer": L, "n_coords": ncoord, "draw": di, "d": d_eff,
                                "r2": rr, "median_angle_deg": ang})
                got = [x["r2"] for x in res if x["layer"] == L and x["n_coords"] == ncoord
                       and "r2" in x]
                log(f"a4 layer {L} n={ncoord:4d}: R2 median {np.median(got):.3f} "
                    f"(min {min(got):.3f} max {max(got):.3f})")
            out["a4"] = res
            save()

    # --------------------------------------------------------------- b1 + b2 + b3
    if any(s in a.sections for s in ("b1", "b2", "b3")):
        b1, b2, b3 = [], [], []
        for L in (layers if "b1" in a.sections else a.layers):
            li = layers.index(L)
            for di in range(3):
                coords = mf.draw_coords(elig, mf.N_COORDS, np.random.default_rng(1000 + di))
                U = U_of(li, coords)
                # b1: shared vs private variance at d = 10, plus the loading spectrum
                fa, mu, sd = fa_parts(U[tr], mf.MANIFOLD_DIM)
                Lam, Psi = fa.components_.T, np.maximum(fa.noise_variance_, 1e-8)
                comm = (Lam ** 2).sum(1)                     # communality per coordinate
                sv = np.linalg.svd(Lam, compute_uv=False)
                b1.append({"layer": L, "depth": L / max(nl - 1, 1), "draw": di,
                           "shared_frac_mean": float((comm / (comm + Psi)).mean()),
                           "shared_frac_median": float(np.median(comm / (comm + Psi))),
                           "shared_frac_total": float(comm.sum() / (comm.sum() + Psi.sum())),
                           "loading_sv": [float(x) for x in sv],
                           "sv_top1_share": float(sv[0] ** 2 / (sv ** 2).sum()),
                           "participation_ratio": float((sv ** 2).sum() ** 2 / (sv ** 4).sum())})
                log(f"b1 layer {L} draw {di}: shared {b1[-1]['shared_frac_total']:.3f}, "
                    f"PR {b1[-1]['participation_ratio']:.2f}")
            out["b1"] = b1
            save()

        for L in a.layers:
            li = layers.index(L)
            for di in range(2):
                coords = mf.draw_coords(elig, mf.N_COORDS, np.random.default_rng(1000 + di))
                U = U_of(li, coords)
                for d in (1, 2, 3, 5, 8, 10, 15, 20, 30):
                    if "b2" in a.sections:
                        fa, mu, sd = fa_parts(U[tr], d)
                        ll_tr = float(fa.score((U[tr] - mu) / sd))
                        ll_te = float(fa.score((U[te] - mu) / sd))
                        b2.append({"layer": L, "draw": di, "d": d,
                                   "loglik_train": ll_tr, "loglik_test": ll_te})
                    if "b3" in a.sections:
                        _, _, rr, ang = fit_and_score(li, coords, d=d)
                        b3.append({"layer": L, "draw": di, "d": d, "r2": rr,
                                   "median_angle_deg": ang})
                    log(f"b2/b3 layer {L} draw {di} d={d:2d}: "
                        f"llte {b2[-1]['loglik_test'] if b2 else float('nan'):.2f} "
                        f"R2 {b3[-1]['r2'] if b3 else float('nan'):.3f}")
                out["b2"], out["b3"] = b2, b3
                save()

    # -------------------------------------------------------------------- a5
    # Follows from a3: at mid layers two DISJOINT coordinate sets agree with each other to
    # ~23 deg while both have NEGATIVE R2 on the next move, so they share a signal that is not
    # the label. The obvious candidate is the direction to the target, which sits in the prompt.
    # Test it directly by refitting the same chain against `fit_req` instead of the next letter.
    # This is the DESIGN 4.2 target-detector question asked of the data rather than argued.
    if "a5" in a.sections:
        REQ = np.asarray(z["fit_req"], dtype=np.float64)
        REQ = REQ / np.maximum(np.linalg.norm(REQ, axis=1, keepdims=True), 1e-12)
        res = []
        for L in layers:
            li = layers.index(L)
            for di in range(3):
                coords = mf.draw_coords(elig, mf.N_COORDS, np.random.default_rng(1000 + di))
                U = U_of(li, coords)
                ro_n = mf.fit_readout(U[tr], G[tr], d=mf.MANIFOLD_DIM)     # fit to next letter
                ro_r = mf.fit_readout(U[tr], REQ[tr], d=mf.MANIFOLD_DIM)   # fit to target dir
                Vn, Vr = ro_n.velocity(U[te]), ro_r.velocity(U[te])
                res.append({
                    "layer": L, "depth": L / max(nl - 1, 1), "draw": di,
                    "r2_next_move": float(r2(Vn, G[te])),
                    "r2_target_dir": float(r2(Vr, REQ[te])),
                    "next_field_vs_label_deg": float(np.median(ang_err(Vn, G[te]))),
                    "next_field_vs_target_deg": float(np.median(ang_err(Vn, REQ[te]))),
                })
                log(f"a5 layer {L} draw {di}: R2 next {res[-1]['r2_next_move']:+.3f} | "
                    f"R2 target {res[-1]['r2_target_dir']:+.3f} | next-field to target "
                    f"{res[-1]['next_field_vs_target_deg']:.1f}deg")
            out["a5"] = res
            save()

    # -------------------------------------------------------------------- b4
    if "b4" in a.sections:
        res = []
        for L in layers:
            li = layers.index(L)
            for di in range(3):
                coords = mf.draw_coords(elig, mf.N_COORDS, np.random.default_rng(1000 + di))
                U = U_of(li, coords)
                st = mf.manifold_stability(U, mf.MANIFOLD_DIM, np.random.default_rng(7 + di))
                res.append({"layer": L, "depth": L / max(nl - 1, 1), "draw": di,
                            **{k: float(v) for k, v in st.items()
                               if isinstance(v, (int, float))}})
                log(f"b4 layer {L} draw {di}: split-half max angle "
                    f"{res[-1].get('max_angle_deg', float('nan')):.1f}deg")
            out["b4"] = res
            save()

    # -------------------------------------------------------------------- a6
    # THE BASELINE THAT SHOULD HAVE EXISTED FROM THE START. Every R2 in this project is a
    # decoder's score at predicting the next move. None of it means anything without knowing what
    # the PROMPT alone already predicts, because the target coordinates are printed in the prompt
    # and calibration paths are shortest paths. No factor analysis and no fitting on activations
    # is involved here: these are the scores of predictors that never look at the model at all.
    if "a6" in a.sections:
        REQ = np.asarray(z["fit_req"], dtype=np.float64)
        REQ = REQ / np.maximum(np.linalg.norm(REQ, axis=1, keepdims=True), 1e-12)
        BIN = np.asarray(z["fit_req_bin"], dtype=int)
        AMBv = np.asarray(z["fit_ambiguous"], dtype=bool)

        # 1. conditional mean of the label given the required-direction bin, fit on train rows
        P = np.zeros((len(te), 2))
        for b in np.unique(BIN):
            m_tr, m_te = BIN[tr] == b, BIN[te] == b
            if m_tr.sum() >= 5 and m_te.any():
                P[m_te] = G[tr][m_tr].mean(0)
        # 2. the continuous required direction, used directly with no fitting whatsoever
        # 3. oracle: the realised letter where it is determined, the bin mean where it is a tie.
        #    Nothing can beat this, because at a tie OUR rng chose the letter.
        O = G[te].copy()
        for b in np.unique(BIN):
            m_te, m_tr = (BIN[te] == b) & AMBv[te], (BIN[tr] == b) & AMBv[tr]
            if m_tr.sum() >= 5 and m_te.any():
                O[m_te] = G[tr][m_tr].mean(0)
        # The oracle is only meaningful for the FORCED-LETTER label, where the irreducible error
        # is our rng's tie-break and the bin mean is the best possible answer there. Under the
        # intent label the ceiling is 1.0 by construction (the label is a deterministic function
        # of the state), and this construction is not an upper bound at all: measured 0.777
        # against 0.791 for the unfitted target, i.e. below it. Report it only where it means
        # something rather than emitting a number that looks like a ceiling and is not one.
        oracle = float(r2(O, G[te])) if a.label == "letter" else None
        out["a6"] = {
            "r2_bin_conditional_mean": float(r2(P, G[te])),
            "r2_required_direction_unfitted": float(r2(REQ[te], G[te])),
            "r2_oracle": oracle,
            "ambiguous_fraction": float(AMBv.mean()),
            "headroom_over_target": (None if oracle is None
                                     else float(oracle - r2(REQ[te], G[te]))),
            "note": ("Any activation decoder must BEAT r2_required_direction_unfitted to have "
                     "shown it reads the model rather than the prompt. Measured 2026-08-11: the "
                     "best decoder over all layers and all d reaches 0.504 and the unfitted "
                     "target direction reaches 0.562, so none of them clears it. Headroom between "
                     "the target baseline and the oracle is under 0.01 of label variance, so the "
                     "calibration task cannot separate intent from target at all."),
        }
        log(f"a6: bin-mean {out['a6']['r2_bin_conditional_mean']:+.3f} | "
            f"req-direct {out['a6']['r2_required_direction_unfitted']:+.3f} | "
            f"oracle {oracle if oracle is None else round(oracle, 3)} "
            f"(letter label only)")
        save()

    # -------------------------------------------------------------------- b5
    # Follows from b2/b3: at layer 14 the held-out R2 goes from -0.17 at d = 10 to +0.19 at
    # d = 30 and the held-out FA log-likelihood is still climbing, so the frozen d = 10 may be
    # what makes mid layers look empty. Redraw the WHOLE depth profile at several d to find out
    # whether the depth story survives a better-specified manifold. This is the one that decides
    # whether "mid layers carry none of it" is a fact about the model or about our d.
    if "b5" in a.sections:
        res = []
        for L in layers:
            li = layers.index(L)
            for di in range(2):
                coords = mf.draw_coords(elig, mf.N_COORDS, np.random.default_rng(1000 + di))
                for d in (10, 20, 30):
                    try:
                        _, _, rr, ang = fit_and_score(li, coords, d=d)
                    except Exception as e:                            # noqa: BLE001
                        res.append({"layer": L, "draw": di, "d": d, "error": str(e)[:60]})
                        continue
                    res.append({"layer": L, "depth": L / max(nl - 1, 1), "draw": di, "d": d,
                                "r2": rr, "median_angle_deg": ang})
                    log(f"b5 layer {L:2d} draw {di} d={d:2d}: R2 {rr:+.3f}")
            out["b5"] = res
            save()

    save()
    log(f"wrote {dest}")


if __name__ == "__main__":
    main()
