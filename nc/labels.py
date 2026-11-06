"""Labels, prompt-only baselines, and the scores used to compare them (DESIGN 4.2b).

WHY THIS MODULE EXISTS. Every R2 in this project is a decoder's score at predicting where the
model will move next, and on 2026-08-11 that number turned out to mean almost nothing on its own:
the direction to the target is printed in the prompt, the calibration paths were shortest paths,
so the old label (the letter that came next) was nearly a function of the prompt. Measured on the
2026-08-10 artifact: the unfitted target direction scored 0.562 against an oracle ceiling of
0.5715, and no decoder at any layer or any manifold dimension cleared it.

So a readout score is only reportable next to what a PROMPT-ONLY predictor achieves on the same
label and the same split. That comparison is the point of this file, and `best_prompt_only` is the
number every decoder has to beat.

The label itself is now the model's own intent direction: the expected direction under its
next-token distribution over the four move letters. It contains no rng of ours, and at a tie it
carries the model's private preference, which the prompt does not determine.
"""
from __future__ import annotations

import numpy as np

SYMS = ("r", "l", "u", "d")
SYM_DIRS = {"r": (1.0, 0.0), "l": (-1.0, 0.0), "u": (0.0, 1.0), "d": (0.0, -1.0)}


def unit(V: np.ndarray) -> np.ndarray:
    V = np.atleast_2d(np.asarray(V, dtype=np.float64))
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def ang_err(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Angle in degrees between each prediction and its target, both taken as directions."""
    p, t = unit(pred), unit(true)
    return np.degrees(np.arccos(np.clip((p * t).sum(1), -1, 1)))


def r2(pred: np.ndarray, true: np.ndarray) -> float:
    """1 - SSE/SST on unit directions. Prediction is normalised, so gain gets no credit.

    Read it against a measured baseline, never against 1.0: see the module docstring.
    """
    p = unit(pred)
    true = np.atleast_2d(np.asarray(true, dtype=np.float64))
    ssr = float(((p - true) ** 2).sum())
    sst = float(((true - true.mean(0)) ** 2).sum())
    return 1.0 - ssr / max(sst, 1e-12)


def intent_direction(P: np.ndarray, order=SYMS):
    """``(directions, commitment, usable)`` from a distribution over the four letters.

    ``commitment`` is |E[direction]|: 1.0 when the model is certain, 0.707 for an even split
    between two adjacent letters, 0 for a symmetric split between opposites. It is the quantity
    that decides whether this label carries anything the prompt does not, because a model sitting
    at exactly 0.5/0.5 on every tie would put the expected direction back on the target bisector.
    """
    P = np.atleast_2d(np.asarray(P, dtype=np.float64))
    D = np.array([SYM_DIRS[s] for s in order], dtype=np.float64)
    raw = P @ D
    commitment = np.linalg.norm(raw, axis=1)
    usable = np.isfinite(raw).all(1) & (commitment > 1e-6)
    return unit(raw), commitment, usable


def readout_label(z, split="fit"):
    """The label the readout is FIT on, for the live loop and the screen (DESIGN 7.2c).

    ONE definition, imported by fit_draws.py, screen.py and run.py, because the 2026-08-11 label
    change landed in the offline fitting path only: the gate certified an intent-fit decoder while
    scripts/run.py kept fitting the superseded forced-letter label, so 570 sessions ran a decoder
    that was never the one gated on. Anything that builds a readout now calls this.

    Returns ``(G, usable)`` where G is (N, 2) unit directions and ``usable`` masks rows whose
    distribution is too symmetric to carry a direction at all.
    """
    P = z[f"{split}_letter_p"]
    order = [str(x) for x in z["letter_order"]]
    G, _commit, usable = intent_direction(np.asarray(P, dtype=np.float64), order)
    return G, usable


def to_letter(V: np.ndarray, order=SYMS) -> np.ndarray:
    """Collapse 2D directions to the nearest of the four move directions."""
    D = np.array([SYM_DIRS[s] for s in order], dtype=np.float64)
    return np.argmax(np.atleast_2d(np.asarray(V, dtype=np.float64)) @ D.T, axis=1)


def geometry_features(req: np.ndarray, req_bin: np.ndarray, ambiguous: np.ndarray) -> np.ndarray:
    """Everything a prompt-only predictor could use, as features.

    The unfitted target direction is a weak adversary because it cannot express a systematic
    tie-break. A plausible one is "take the axis with more distance remaining", which is a
    function of the prompt, so `|x| - |y|` is included explicitly; the polynomial terms cover
    other smooth functions of the offset and the one-hot covers any per-bin rule. A readout that
    beats a ridge on THESE features is carrying something the prompt does not contain.
    """
    req = np.atleast_2d(np.asarray(req, dtype=np.float64))
    x, y = req[:, 0], req[:, 1]
    n = len(x)
    bins = np.asarray(req_bin, dtype=int)
    onehot = np.zeros((n, 8))
    onehot[np.arange(n), np.clip(bins, 0, 7)] = 1.0
    amb = np.asarray(ambiguous, dtype=np.float64)
    cols = [x, y, np.abs(x), np.abs(y), np.abs(x) - np.abs(y),
            np.sign(x), np.sign(y), x * y, x ** 2, y ** 2,
            x ** 3, y ** 3, x ** 2 * y, x * y ** 2, np.hypot(x, y), amb, np.ones(n)]
    return np.column_stack(cols + [onehot])


def ridge_fit_predict(Xtr, Ytr, Xte, alpha: float):
    mu = Xtr.mean(0)
    sd = np.where(Xtr.std(0) < 1e-9, 1.0, Xtr.std(0))
    A = (Xtr - mu) / sd
    W = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ Ytr)
    return ((Xte - mu) / sd) @ W


def ridge_cv(Xtr, Ytr, Xva, Yva, Xte, alphas=(1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)):
    """Ridge with alpha chosen on a validation split. Returns ``(prediction, alpha)``."""
    best = None
    for al in alphas:
        sc = r2(ridge_fit_predict(Xtr, Ytr, Xva, al), Yva)
        if best is None or sc > best[0]:
            best = (sc, al)
    return ridge_fit_predict(Xtr, Ytr, Xte, best[1]), best[1]


def prompt_only_baselines(Y, req, req_bin, ambiguous, tr, va, te, P=None) -> dict:
    """Three prompt-only predictors of the same label on the same split, weakest to strongest.

    Every decoder score must be reported against ``best_prompt_only``. When ``P`` (the model's
    letter distribution) is given, the same predictors are also scored by exact match against the
    model's own most likely letter, which is the interpretable version of the comparison.
    """
    Y = np.asarray(Y, dtype=np.float64)
    REQ = unit(req)
    out = {"unfitted_target_direction": float(r2(REQ[te], Y[te])),
           "unfitted_target_direction_deg": float(np.median(ang_err(REQ[te], Y[te])))}

    bins = np.asarray(req_bin, dtype=int)
    Pb = np.zeros((len(te), 2))
    for b in np.unique(bins):
        m_tr, m_te = bins[tr] == b, bins[te] == b
        if m_tr.sum() >= 5 and m_te.any():
            Pb[m_te] = Y[tr][m_tr].mean(0)
    out["bin_conditional_mean"] = float(r2(Pb, Y[te]))

    F = geometry_features(req, req_bin, ambiguous)
    pg, alpha = ridge_cv(F[tr], Y[tr], F[va], Y[va], F[te])
    out["geometry_ridge"] = float(r2(pg, Y[te]))
    out["geometry_ridge_alpha"] = float(alpha)
    out["geometry_ridge_deg"] = float(np.median(ang_err(pg, Y[te])))
    out["best_prompt_only"] = max(out["unfitted_target_direction"],
                                  out["bin_conditional_mean"], out["geometry_ridge"])
    out["best_prompt_only_source"] = max(
        ("unfitted_target_direction", "bin_conditional_mean", "geometry_ridge"),
        key=lambda k: out[k])

    if P is not None:
        P = np.atleast_2d(np.asarray(P, dtype=np.float64))
        want = np.argmax(P[te], axis=1)
        # The subset where the prompt's nearest letter is NOT the model's preferred letter. The
        # unfitted target direction scores 0 here BY CONSTRUCTION, since the subset is defined by
        # its disagreement, so read the geometry ridge as the prompt-side number.
        disagree = to_letter(REQ[te]) != want
        out["frac_prompt_disagrees_with_model"] = float(disagree.mean())
        out["match_majority_class"] = float(max((want == k).mean() for k in range(P.shape[1])))
        for nm, pred in (("unfitted_target_direction", REQ[te]), ("geometry_ridge", pg)):
            hit = to_letter(pred) == want
            out[f"match_{nm}"] = float(hit.mean())
            out[f"match_{nm}_ambiguous"] = float(hit[np.asarray(ambiguous, bool)[te]].mean())
            out[f"match_{nm}_where_prompt_disagrees"] = (
                float(hit[disagree].mean()) if disagree.any() else float("nan"))
        out["n_where_prompt_disagrees"] = int(disagree.sum())
    return out


def match_accuracy(pred, P, mask=None) -> float:
    """Exact-match rate: does the prediction collapse to the model's own most likely letter?"""
    want = np.argmax(np.atleast_2d(np.asarray(P, dtype=np.float64)), axis=1)
    hit = to_letter(pred) == want
    return float(hit.mean()) if mask is None else (
        float(hit[mask].mean()) if np.any(mask) else float("nan"))
