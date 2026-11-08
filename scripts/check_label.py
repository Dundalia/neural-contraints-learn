#!/usr/bin/env python
"""Decide, before committing an hour, whether the intent-distribution label opens a usable gap.

THE QUESTION. Today the decoder is fit to the letter that came next, and that letter is nearly a
function of the direction to the target, which is printed in the prompt. Measured on the G1
artifact: the unfitted target direction predicts the label at R2 0.562, the oracle ceiling is
0.5715, so under 0.01 of label variance separates "reads the model" from "reads the prompt", and
no decoder we have clears the target baseline at all.

THE PROPOSED FIX. Keep the corpus, change the label. At read position j take the model's own
next-token distribution restricted to the four move letters, renormalise, and use its expected
direction. The label then has no rng noise in it and the ceiling rises from 0.5715 to 1.0.

WHY THIS SCRIPT EXISTS. Removing the noise also helps the trivial predictor, because the noise was
hurting it too. At a tie the target direction predicts the 45 deg bisector; against the old label
that was 45 deg wrong every time, against the new label it is only wrong by the model's deviation
from 50/50. So R2(target -> new label) will RISE, and the fix is only worth having if the
activation readout rises faster. That difference is the whole question, and one number answers it.

DESIGN NOTES. The readout here is deliberately GENEROUS: ridge straight from raw coordinates with
alpha chosen on a validation split, no factor analysis and no 10-dimensional bottleneck. That
upper-bounds what the real pipeline could extract, so if this cannot beat the target baseline, no
tuning of d or the population will. Reads at two deep layers only, since mid layers are hopeless.

    python scripts/check_label.py --model Qwen/Qwen3.5-9B --n-turns 1000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nc.task import ALPHABETS, BoardConfig                                   # noqa: E402
from calibrate import read_chunked, simulate_turns                           # noqa: E402
from fit_draws import ang_err, r2                                            # noqa: E402

SYMS = ("r", "l", "u", "d")
DIRS = np.array([ALPHABETS["rlud"][s] for s in SYMS], dtype=np.float64)      # (4, 2)


def unit(V):
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def letter_distributions(backend, turns, chunk, top=20):
    """Per read position, the renormalised distribution over the four move letters.

    Uses prompt_logprobs, so a letter outside the top-`top` at that position is missing; the
    fraction of positions where that happens is reported rather than silently zero-filled.
    """
    SP, TP = backend._SamplingParams, backend._TokensPrompt
    rows, missing, n = [], 0, 0
    sym_ids: dict[str, int] = {}
    for s in range(0, len(turns), chunk):
        batch = turns[s: s + chunk]
        built = [backend._build_ids(b["conv"], b["reply"], None, False) for b in batch]
        outs = backend.llm.generate(
            [TP(prompt_token_ids=ids) for ids, _, _ in built],
            SP(temperature=0.0, max_tokens=1, prompt_logprobs=top))
        for b, (ids, ss, sl), o in zip(batch, built, outs):
            moves = b["reply"].split()
            # Learn the token id of each space-prefixed symbol from the ids themselves. Positions
            # >= 1 are the ones we need, and those are the space-prefixed variants.
            for j in range(1, min(sl, len(moves))):
                sym_ids.setdefault(moves[j], ids[ss + j])
            plp = o.prompt_logprobs
            for j in range(min(sl - 1, len(b["labels"]))):
                cell = plp[ss + j + 1] if plp and plp[ss + j + 1] else {}
                lp = np.full(4, -np.inf)
                for k, sym in enumerate(SYMS):
                    tid = sym_ids.get(sym)
                    if tid is not None and tid in cell:
                        lp[k] = float(getattr(cell[tid], "logprob", -np.inf))
                n += 1
                if not np.isfinite(lp).any():
                    missing += 1
                    rows.append(np.full(4, np.nan))
                    continue
                missing += int(not np.isfinite(lp).all())
                p = np.exp(lp - np.nanmax(lp[np.isfinite(lp)]))
                p[~np.isfinite(lp)] = 0.0
                rows.append(p / max(p.sum(), 1e-12))
        print(f"  logits {min(s+chunk, len(turns))}/{len(turns)}", flush=True)
    return np.asarray(rows, dtype=np.float64), missing / max(n, 1), sym_ids


def geometry_features(meta):
    """Everything a PROMPT-ONLY predictor could use. This is the real adversary.

    The unfitted target direction is a weak baseline: it cannot express a systematic tie-break.
    My stated worry was that the model breaks ties by a geometric rule ("take the axis with more
    distance remaining"), which IS a function of the prompt, so a fitted geometry predictor would
    capture it and the activations would add nothing. `absdiff` below is exactly that rule, and
    the polynomial terms cover any other smooth function of the offset. If the activation readout
    beats THIS, the intent information genuinely is not in the prompt.
    """
    req = np.array([m["req"] for m in meta], dtype=np.float64)
    x, y = req[:, 0], req[:, 1]
    bins = np.array([m["req_bin"] for m in meta], dtype=int)
    onehot = np.zeros((len(meta), 8))
    onehot[np.arange(len(meta)), np.clip(bins, 0, 7)] = 1.0
    amb = np.array([m["ambiguous"] for m in meta], dtype=np.float64)
    cols = [x, y, np.abs(x), np.abs(y),
            np.abs(x) - np.abs(y),            # "prefer the longer remaining axis"
            np.sign(x), np.sign(y),
            x * y, x ** 2, y ** 2, x ** 3, y ** 3, x ** 2 * y, x * y ** 2,
            np.hypot(x, y), amb, np.ones(len(meta))]
    return np.column_stack(cols + [onehot])


def to_letter(V):
    """Collapse a 2D direction to the nearest of the four move directions."""
    return np.argmax(np.asarray(V, dtype=np.float64) @ DIRS.T, axis=1)


def match_stats(pred, P, amb, disagree):
    """Exact-match accuracy against the argmax of the model's own 4-way distribution."""
    want = np.argmax(P, axis=1)
    got = to_letter(pred)
    hit = got == want
    return {
        "acc": float(hit.mean()),
        "acc_ambiguous": float(hit[amb].mean()) if amb.any() else float("nan"),
        "acc_unambiguous": float(hit[~amb].mean()) if (~amb).any() else float("nan"),
        # THE SUBSET THAT MATTERS: positions where the prompt's nearest letter is NOT the letter
        # the model prefers. A prompt-only predictor is wrong here by construction, so accuracy
        # here is the cleanest evidence that the readout carries intent rather than geometry.
        "acc_where_prompt_disagrees": float(hit[disagree].mean()) if disagree.any() else float("nan"),
        "n_where_prompt_disagrees": int(disagree.sum()),
    }


def ridge_cv(Xtr, Ytr, Xva, Yva, Xte, alphas=(1e1, 1e2, 1e3, 1e4, 1e5, 1e6)):
    """Ridge with alpha picked on a validation split. Generous readout on purpose."""
    mu, sd = Xtr.mean(0), np.where(Xtr.std(0) < 1e-9, 1.0, Xtr.std(0))
    A, B, C = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    G = A.T @ A
    best = None
    for al in alphas:
        W = np.linalg.solve(G + al * np.eye(A.shape[1]), A.T @ Ytr)
        sc = r2(B @ W, Yva)
        if best is None or sc > best[0]:
            best = (sc, al, W)
    _, al, W = best
    return C @ W, al


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--n-turns", type=int, default=1000)
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="default: the two deepest sampled in G1 (21, 28 on a 32-layer model)")
    p.add_argument("--n-coords", type=int, default=512)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--gpu-frac", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=24000)
    p.add_argument("--out", default=str(ROOT / "artifacts/label_check.json"))
    a = p.parse_args()
    t0 = time.time()

    cfg = BoardConfig()
    turns = simulate_turns(cfg, a.n_turns, random.Random(0), "fit")
    print(f"[check] {len(turns)} turns simulated ({time.time()-t0:.0f}s)", flush=True)

    from nc.backends.vllm_lens_backend import VLLMLensBackend, VLLMLensConfig
    backend = VLLMLensBackend(VLLMLensConfig(
        model_name=a.model, gpu_memory_utilization=a.gpu_frac,
        max_model_len=a.max_model_len, enable_prefix_caching=False))
    nl = backend.num_layers
    layers = a.layers or [round(21 * nl / 32), round(28 * nl / 32)]
    print(f"[check] layers {layers} of {nl} ({time.time()-t0:.0f}s)", flush=True)

    U, meta = read_chunked(backend, turns, layers, a.chunk)
    print(f"[check] reads done {U.shape} ({time.time()-t0:.0f}s)", flush=True)
    P, miss_frac, sym_ids = letter_distributions(backend, turns, a.chunk)
    print(f"[check] distributions done, letters outside top-20 in {miss_frac:.3%} of positions; "
          f"token ids {sym_ids} ({time.time()-t0:.0f}s)", flush=True)

    assert P.shape[0] == U.shape[1] == len(meta), (P.shape, U.shape, len(meta))
    ok = np.isfinite(P).all(1)
    G_old = unit(np.array([ALPHABETS["rlud"][m["next_move"]] for m in meta], dtype=np.float64))
    REQ = unit(np.array([m["req"] for m in meta], dtype=np.float64))
    V_new_raw = P @ DIRS                                   # expected direction under the model
    G_new = unit(V_new_raw)
    conf = np.linalg.norm(V_new_raw, axis=1)               # 1 = committed, 0 = a perfect tie
    amb = np.array([m["ambiguous"] for m in meta], dtype=bool)
    ok &= np.isfinite(G_new).all(1) & (conf > 1e-6)
    print(f"[check] usable positions {ok.sum()}/{len(ok)}; ambiguous {amb[ok].mean():.3f}; "
          f"median commitment |E[dir]| {np.median(conf[ok]):.3f} "
          f"(ambiguous only {np.median(conf[ok & amb]):.3f})", flush=True)

    # Cache everything the GPU produced, so any further metric is an offline question.
    cache = Path(a.out).with_suffix(".npz")
    np.savez_compressed(
        cache, U=U, P=P, ok=ok,
        next_move=np.array([m["next_move"] for m in meta]),
        req=np.array([m["req"] for m in meta], dtype=np.float32),
        req_bin=np.array([m["req_bin"] for m in meta], dtype=np.int16),
        ambiguous=np.array([m["ambiguous"] for m in meta]),
        depth=np.array([m["depth"] for m in meta], dtype=np.int32),
        layers=np.array(layers))
    print(f"[check] cached raw arrays to {cache} ({time.time()-t0:.0f}s)", flush=True)

    idx = np.random.default_rng(0).permutation(np.flatnonzero(ok))
    n = len(idx)
    tr, va, te = idx[: int(0.6 * n)], idx[int(0.6 * n): int(0.8 * n)], idx[int(0.8 * n):]
    coords = np.random.default_rng(1).choice(U.shape[2], size=min(a.n_coords, U.shape[2]),
                                            replace=False)

    res = {"n_turns": len(turns), "n_positions": int(n), "layers": layers,
           "n_coords": int(len(coords)), "missing_letter_frac": float(miss_frac),
           "ambiguous_frac": float(amb[ok].mean()),
           "median_commitment": float(np.median(conf[ok])),
           "median_commitment_ambiguous": float(np.median(conf[ok & amb])),
           "baselines": {}, "readouts": {}}

    # --- prompt-only baselines against each label. THE COMPARISON.
    # Three of them, weakest to strongest. The activation readout has to beat the STRONGEST.
    Fg = geometry_features(meta)
    for lab, Y in (("old_letter", G_old), ("new_intent", G_new)):
        # 1. the required direction, unfitted
        b = {"target_direction_unfitted": float(r2(REQ[te], Y[te])),
             "target_direction_deg": float(np.median(ang_err(REQ[te], Y[te])))}
        # 2. conditional mean of the label given the 8-bin required direction, fit on train
        Pb = np.zeros((len(te), 2))
        bins = np.array([m["req_bin"] for m in meta], dtype=int)
        for bb in np.unique(bins):
            m_tr, m_te = bins[tr] == bb, bins[te] == bb
            if m_tr.sum() >= 5 and m_te.any():
                Pb[m_te] = Y[tr][m_tr].mean(0)
        b["bin_conditional_mean"] = float(r2(Pb, Y[te]))
        # 3. fitted ridge on all geometry features, same splits as the activation readout
        pg, alg = ridge_cv(Fg[tr], Y[tr], Fg[va], Y[va], Fg[te])
        b["geometry_ridge"] = float(r2(pg, Y[te]))
        b["geometry_ridge_alpha"] = float(alg)
        b["best_prompt_only"] = max(b["target_direction_unfitted"], b["bin_conditional_mean"],
                                    b["geometry_ridge"])
        if lab == "new_intent":
            amb_te = amb[te]
            want = np.argmax(P[te], axis=1)
            disagree = to_letter(REQ[te]) != want          # prompt's letter != model's letter
            b["match_unfitted_target"] = match_stats(REQ[te], P[te], amb_te, disagree)
            b["match_geometry_ridge"] = match_stats(pg, P[te], amb_te, disagree)
            b["frac_prompt_disagrees_with_model"] = float(disagree.mean())
            # majority-class floor, for reference
            b["match_majority_class"] = float(
                max((want == k).mean() for k in range(4)))
        res["baselines"][lab] = b
        print(f"[check] PROMPT-ONLY {lab:11s}: unfitted {b['target_direction_unfitted']:+.4f} | "
              f"bin-mean {b['bin_conditional_mean']:+.4f} | geometry-ridge "
              f"{b['geometry_ridge']:+.4f} -> best {b['best_prompt_only']:+.4f}", flush=True)

    # --- the generous activation readout against each label, per layer
    for li, L in enumerate(layers):
        X = U[li][:, coords].astype(np.float64)
        for lab, Y in (("old_letter", G_old), ("new_intent", G_new)):
            pred, al = ridge_cv(X[tr], Y[tr], X[va], Y[va], X[te])
            res["readouts"][f"layer{L}_{lab}"] = {
                "r2": float(r2(pred, Y[te])), "alpha": float(al),
                "median_angle_deg": float(np.median(ang_err(pred, Y[te]))),
                # measured against the STRONGEST prompt-only predictor, not the unfitted one
                "gap_over_target": float(r2(pred, Y[te])
                                         - res["baselines"][lab]["best_prompt_only"]),
                "gap_over_unfitted": float(r2(pred, Y[te])
                                           - res["baselines"][lab]["target_direction_unfitted"]),
            }
            if lab == "new_intent":
                res["readouts"][f"layer{L}_{lab}"]["match"] = match_stats(
                    pred, P[te], amb[te], to_letter(REQ[te]) != np.argmax(P[te], axis=1))
            d = res["readouts"][f"layer{L}_{lab}"]
            print(f"[check] layer {L:3d} {lab:11s}: R2 {d['r2']:+.4f}  "
                  f"gap over target {d['gap_over_target']:+.4f}  (alpha {al:g})", flush=True)

    gaps = {k: v["gap_over_target"] for k, v in res["readouts"].items() if "new_intent" in k}
    res["verdict"] = {
        "best_gap_new_label": max(gaps.values()) if gaps else None,
        "gap_today_old_label": max(v["gap_over_target"] for k, v in res["readouts"].items()
                                   if "old_letter" in k),
        "gap_is_measured_against": "best_prompt_only (max of unfitted, bin-mean, geometry-ridge)",
        "note": ("A POSITIVE gap on the new label means the activations carry intent the prompt "
                 "does not, which is the precondition for the whole experiment. Compare against "
                 "the old-label gap, which is negative today (no decoder beats the target)."),
    }
    Path(a.out).write_text(json.dumps(res, indent=2, default=float))
    print(f"\n[check] best gap over target, NEW label: {res['verdict']['best_gap_new_label']:+.4f}")
    print(f"[check] same gap, OLD label:              {res['verdict']['gap_today_old_label']:+.4f}")
    print(f"[check] wrote {a.out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
