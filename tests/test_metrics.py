import math
from dataclasses import dataclass, field

import numpy as np
import pytest

from nc.metrics import (ANGLE_KEYS, PAPER_KEYS, after_effect, amount_of_learning, bin_trials,
                        zscore_bins)


@dataclass
class FakeTrial:
    acquired: bool
    moves_used: int
    angular_errors: list = field(default_factory=list)


def trials(spec):
    return [FakeTrial(a, m, list(e)) for a, m, e in spec]


# --------------------------------------------------------------------- binning
def test_binning_drops_the_trailing_partial_bin():
    ts = trials([(True, 4, [0.1])] * 10)
    assert len(bin_trials(ts, 4)) == 2       # 10 trials, bin 4 -> 2 full bins, 2 discarded


def test_bin_aggregates_each_component():
    ts = trials([(True, 4, [0.0, 0.5]), (False, 24, [1.0, 1.5])])
    b = bin_trials(ts, 2)[0]
    assert b.success_rate == pytest.approx(0.5)
    assert b.mean_moves == pytest.approx(14.0)
    assert b.mean_angular_error == pytest.approx(0.75)   # pooled over moves, not over trials
    assert b.n == 2


def test_nan_angular_errors_are_dropped_not_propagated():
    ts = trials([(True, 4, [0.2, float("nan")]), (True, 4, [0.4, 0.6])])
    b = bin_trials(ts, 2)[0]
    assert b.mean_angular_error == pytest.approx(0.4)


def test_all_nan_angular_errors_give_nan_not_a_crash():
    ts = trials([(True, 4, [float("nan")])] * 2)
    assert math.isnan(bin_trials(ts, 2)[0].mean_angular_error)


# ------------------------------------------------------------------ z-scoring
def test_zscore_uses_all_bins_passed():
    ts = trials([(True, 4, [0.1])] * 4 + [(False, 24, [1.0])] * 4)
    z = zscore_bins(bin_trials(ts, 4), PAPER_KEYS)
    assert z.shape == (2, 2)
    assert z.mean(axis=0) == pytest.approx([0.0, 0.0])
    assert z[0, 0] > z[1, 0]        # the successful bin scores higher on success rate
    assert z[0, 1] < z[1, 1]        # and lower on moves


def test_zscore_survives_a_constant_component():
    ts = trials([(True, 4, [0.1])] * 8)
    z = zscore_bins(bin_trials(ts, 4), ANGLE_KEYS)
    assert np.all(np.isfinite(z))


# -------------------------------------------------------- amount of learning
def test_full_recovery_scores_one():
    P_B = np.array([1.0, -1.0])
    P_P = np.array([[-1.0, 1.0], [0.0, 0.0], [1.0, -1.0]])
    out = amount_of_learning(P_B, P_P)
    assert out.l_bin[0] == pytest.approx(0.0)
    assert out.l_bin[-1] == pytest.approx(1.0)
    assert out.l_session == pytest.approx(1.0)


def test_no_recovery_scores_zero():
    P_B = np.array([1.0, -1.0])
    P_P = np.array([[-1.0, 1.0]] * 4)
    assert amount_of_learning(P_B, P_P).l_session == pytest.approx(0.0)


def test_worsening_scores_negative():
    """The projection must be signed: taking the magnitude would fold a bin that got worse
    into apparent learning, and the published range of -0.5 to 1 would be impossible."""
    P_B = np.array([1.0, 0.0])
    P_P = np.array([[0.0, 0.0], [-1.0, 0.0]])
    out = amount_of_learning(P_B, P_P)
    assert out.l_bin[-1] < 0
    assert out.l_session == pytest.approx(0.0)     # max over bins, and bin 0 is 0 by definition


def test_movement_orthogonal_to_the_recovery_direction_does_not_count():
    P_B = np.array([2.0, 0.0])
    P_P = np.array([[0.0, 0.0], [0.0, 5.0]])       # moved a lot, none of it toward baseline
    assert amount_of_learning(P_B, P_P).l_bin[-1] == pytest.approx(0.0)


def test_overshoot_beyond_baseline_scores_above_one():
    P_B = np.array([1.0, 0.0])
    P_P = np.array([[0.0, 0.0], [2.0, 0.0]])
    assert amount_of_learning(P_B, P_P).l_session == pytest.approx(2.0)


def test_zero_impairment_is_undefined_not_zero():
    P_B = np.array([0.5, 0.5])
    P_P = np.array([[0.5, 0.5], [0.9, 0.1]])
    out = amount_of_learning(P_B, P_P)
    assert math.isnan(out.l_session) and np.all(np.isnan(out.l_bin))
    assert out.initial_impairment == pytest.approx(0.0)


def test_initial_impairment_is_the_distance_to_baseline():
    P_B = np.array([3.0, 4.0])
    P_P = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert amount_of_learning(P_B, P_P).initial_impairment == pytest.approx(5.0)


def test_tail_mean_is_insensitive_to_a_single_lucky_bin():
    """l_session is a max statistic and grows with the number of bins; the pre-registered
    non-max summary must not be moved by one outlying bin (DESIGN 6)."""
    P_B = np.array([1.0, 0.0])
    spiky = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    flat = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    assert amount_of_learning(P_B, spiky).l_session == pytest.approx(1.0)
    assert amount_of_learning(P_B, spiky).tail_mean == pytest.approx(
        amount_of_learning(P_B, flat).tail_mean)


def test_l_session_max_bias_is_real_and_visible():
    """Same underlying noise, more bins -> larger max. Recorded as a known property so the
    statistic is never compared across different block lengths."""
    rng = np.random.default_rng(0)
    P_B = np.array([1.0, 0.0])
    short, long = [], []
    for _ in range(200):
        noise = rng.normal(0, 0.3, size=(16, 2))
        noise[0] = 0.0
        P = np.array([0.0, 0.0]) + noise
        short.append(amount_of_learning(P_B, P[:4]).l_session)
        long.append(amount_of_learning(P_B, P).l_session)
    assert np.mean(long) > np.mean(short)


# ------------------------------------------------- end-to-end positive control
def _planted_session(recovers: bool, n_base=16, n_pert=40, seed=0):
    """Trials with a KNOWN answer: performance drops at the perturbation and either climbs
    back or does not. The whole chain (binning -> z-scoring -> signed projection) must recover
    that, otherwise a real null and a broken pipeline are indistinguishable."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_base):
        out.append(FakeTrial(True, 8, [float(rng.uniform(0, 0.3))] * 8))
    for i in range(n_pert):
        frac = i / max(n_pert - 1, 1)
        good = 0.25 + (0.6 * frac if recovers else 0.0)
        mv = int(8 + (1 - good) * 20)
        out.append(FakeTrial(bool(rng.random() < good), mv,
                             [float(rng.uniform(0, (1 - good) * 2.5 + 0.2))] * mv))
    return out, n_base, n_pert


@pytest.mark.parametrize("keys", [PAPER_KEYS, ANGLE_KEYS])
def test_pipeline_recovers_a_planted_recovery(keys):
    trials, nb, npt = _planted_session(True, seed=1)
    bins = bin_trials(trials, 8)
    Z = zscore_bins(bins, keys)
    nbb = nb // 8
    out = amount_of_learning(Z[:nbb].mean(0), Z[nbb:])
    assert out.l_session > 0.4, f"planted recovery not detected: {out.l_session}"


@pytest.mark.parametrize("keys", [PAPER_KEYS, ANGLE_KEYS])
def test_l_session_has_a_POSITIVE_null_floor(keys):
    """MEASURED 2026-08-10, 200 synthetic sessions with no learning planted: ``L_session``
    averages 0.12 (paper pair) / 0.18 (angle pair) and reaches p95 = 0.28 / 0.37. It is a max
    over bins, so noise alone lifts it well above zero.

    Consequence: any criterion of the form "L_session > 0" is nearly vacuous, which is why the
    G4 gate compares against the no-perturbation control rather than against zero, and why the
    primary analysis is a REGRESSION on alignment -- a constant offset cancels in a slope."""
    vals = []
    for s in range(40):
        trials, nb, _ = _planted_session(False, seed=100 + s)
        Z = zscore_bins(bin_trials(trials, 8), keys)
        vals.append(amount_of_learning(Z[:nb // 8].mean(0), Z[nb // 8:]).l_session)
    assert np.mean(vals) > 0.03, "the null floor is a documented property; if it vanished, check"
    assert np.mean(vals) < 0.35, f"null floor unexpectedly high: {np.mean(vals)}"


@pytest.mark.parametrize("keys", [PAPER_KEYS, ANGLE_KEYS])
def test_tail_mean_is_unbiased_under_the_null(keys):
    """The non-max summary the design prescribes alongside ``L_session`` IS centred at zero
    (measured: 0.008 / -0.011), so it is the statistic for any absolute claim."""
    vals = []
    for s in range(40):
        trials, nb, _ = _planted_session(False, seed=200 + s)
        Z = zscore_bins(bin_trials(trials, 8), keys)
        vals.append(amount_of_learning(Z[:nb // 8].mean(0), Z[nb // 8:]).tail_mean)
    assert abs(np.mean(vals)) < 0.10, f"tail_mean should be unbiased, got {np.mean(vals)}"


@pytest.mark.parametrize("keys", [PAPER_KEYS, ANGLE_KEYS])
def test_the_metric_separates_recovery_from_no_recovery(keys):
    """The property that actually matters: planted recovery must score clearly above planted
    non-recovery. Measured separation is p05(recovery) 0.56-0.61 vs p95(null) 0.28-0.37."""
    def med(rec, off):
        v = []
        for s in range(30):
            trials, nb, _ = _planted_session(rec, seed=off + s)
            Z = zscore_bins(bin_trials(trials, 8), keys)
            v.append(amount_of_learning(Z[:nb // 8].mean(0), Z[nb // 8:]).l_session)
        return float(np.median(v))
    assert med(True, 300) - med(False, 400) > 0.35


# ---------------------------------------------------------------- after-effect
def test_after_effect_is_zero_when_washout_returns_to_baseline():
    assert after_effect(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == pytest.approx(0.0)


def test_after_effect_grows_with_the_washout_impairment():
    P_B = np.array([0.0, 0.0])
    assert after_effect(P_B, np.array([1.0, 0.0])) < after_effect(P_B, np.array([3.0, 0.0]))
