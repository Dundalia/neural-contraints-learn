"""Performance binning and Sadtler's amount-of-learning metric (docs/DESIGN.md §1.7, §3.7).

Performance is a vector per bin of trials. The learning metric is the signed projection of
the recovery onto the maximum-recovery direction, so component sign conventions do not
matter: ``L_max`` points from the impaired point toward baseline, and any movement along it
is improvement whether the component is success rate (up is good) or moves and angular
error (up is bad).

Two metric pairs are computed and both are pre-registered (§3.7):
  * ``("success_rate", "mean_moves")``          -- the paper-faithful pair
  * ``("success_rate", "mean_angular_error")``  -- the co-primary, far higher resolution
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PAPER_KEYS = ("success_rate", "mean_moves")
ANGLE_KEYS = ("success_rate", "mean_angular_error")


@dataclass
class Bin:
    success_rate: float
    mean_moves: float
    mean_angular_error: float
    n: int


def bin_trials(trials, bin_size: int) -> list[Bin]:
    """Non-overlapping bins over a trial sequence; a trailing partial bin is dropped.

    ``trials`` is any sequence of objects exposing ``acquired``, ``moves_used`` and
    ``angular_errors`` (i.e. :class:`nc.task.Trial`). Failed trials contribute their full
    move budget, which is the analog of the paper crediting a timeout at the time limit.
    """
    out = []
    for s in range(0, len(trials) - bin_size + 1, bin_size):
        chunk = trials[s: s + bin_size]
        errs = np.concatenate([np.asarray(t.angular_errors, dtype=float) for t in chunk]) \
            if chunk else np.array([])
        errs = errs[~np.isnan(errs)]
        out.append(Bin(
            success_rate=float(np.mean([bool(t.acquired) for t in chunk])),
            mean_moves=float(np.mean([t.moves_used for t in chunk])),
            mean_angular_error=float(errs.mean()) if errs.size else float("nan"),
            n=len(chunk),
        ))
    return out


def zscore_bins(bins: list[Bin], keys=PAPER_KEYS) -> np.ndarray:
    """``(n_bins, len(keys))`` z-scored across **all** bins passed.

    Pass every bin of the session (baseline + perturbation + washout) together, as the
    paper does -- the normalisation constants must not differ between blocks.
    """
    raw = np.array([[getattr(b, k) for k in keys] for b in bins], dtype=float)
    sd = raw.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (raw - raw.mean(axis=0)) / sd


@dataclass
class Learning:
    l_bin: np.ndarray          # signed recovery fraction per perturbation bin (l_bin[0] == 0)
    l_session: float           # max over bins -- the paper's statistic (length-dependent)
    tail_mean: float           # mean of the last two bins -- the pre-registered non-max summary
    initial_impairment: float  # ||L_max||


def amount_of_learning(p_baseline: np.ndarray, p_perturb: np.ndarray) -> Learning:
    """Sadtler eq. 16-20 with a signed projection.

    ``p_baseline`` is the mean z-scored performance over baseline bins, ``p_perturb`` is
    ``(n_bins, n_components)`` over the perturbation block. The projection is signed so a
    bin that got *worse* scores negative, which is what makes the published range of
    -0.5 to 1 possible; taking the magnitude would fold worsening into apparent learning.

    Returns all-NaN when the initial impairment is ~0: with no impairment there is no
    recovery direction, and the ratio is undefined rather than zero.
    """
    p_baseline = np.asarray(p_baseline, dtype=float)
    p_perturb = np.asarray(p_perturb, dtype=float)
    l_max = p_baseline - p_perturb[0]
    denom = float(l_max @ l_max)
    if denom < 1e-12:
        n = len(p_perturb)
        return Learning(np.full(n, np.nan), float("nan"), float("nan"), 0.0)
    l_bin = ((p_perturb - p_perturb[0]) @ l_max) / denom
    return Learning(
        l_bin=l_bin,
        l_session=float(np.max(l_bin)),
        tail_mean=float(np.mean(l_bin[-2:])),
        initial_impairment=float(np.sqrt(denom)),
    )


def after_effect(p_baseline: np.ndarray, p_washout_first: np.ndarray) -> float:
    """Size of the impairment at the start of washout, measured like the perturbation-block
    impairment (paper, Extended Data Fig 3). Larger means more learning had occurred."""
    d = np.asarray(p_baseline, dtype=float) - np.asarray(p_washout_first, dtype=float)
    return float(np.linalg.norm(d))
