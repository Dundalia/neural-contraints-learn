"""Tests for nc/labels.py: the intent label and the prompt-only baselines (DESIGN 4.2b)."""
from __future__ import annotations

import numpy as np
import pytest

from nc import labels as lb


def test_intent_direction_certain_distribution_is_the_letter():
    P = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
    V, commit, ok = lb.intent_direction(P)
    assert np.allclose(V, [[1, 0], [-1, 0], [0, 1], [0, -1]])
    assert np.allclose(commit, 1.0)
    assert ok.all()


def test_intent_direction_even_tie_is_the_bisector_with_commitment_root_half():
    """An even split between adjacent letters puts the label ON the target bisector.

    This is the failure mode the label change has to avoid: if the model split every tie 50/50,
    the intent label would coincide with the target direction and carry nothing extra.
    """
    V, commit, ok = lb.intent_direction(np.array([[0.5, 0.0, 0.5, 0.0]]))
    assert np.allclose(V[0], [np.sqrt(0.5), np.sqrt(0.5)])
    assert commit[0] == pytest.approx(np.sqrt(0.5), abs=1e-9)   # 0.707
    assert ok[0]


def test_intent_direction_asymmetric_tie_leaves_the_bisector():
    V, commit, _ = lb.intent_direction(np.array([[0.85, 0.0, 0.15, 0.0]]))
    ang = np.degrees(np.arctan2(V[0, 1], V[0, 0]))
    assert 5.0 < ang < 25.0                                     # well away from 45
    assert commit[0] > np.sqrt(0.5)


def test_intent_direction_marks_symmetric_opposites_unusable():
    """r against l at 50/50 cancels to the zero vector, which carries no direction at all."""
    _, commit, ok = lb.intent_direction(np.array([[0.5, 0.5, 0.0, 0.0]]))
    assert commit[0] == pytest.approx(0.0, abs=1e-12)
    assert not ok[0]


def test_r2_of_perfect_and_mean_predictors():
    Y = lb.unit(np.array([[1.0, 0], [0, 1.0], [-1.0, 0], [0, -1.0]] * 5))
    assert lb.r2(Y, Y) == pytest.approx(1.0, abs=1e-9)
    mean_pred = np.repeat(Y.mean(0, keepdims=True) + 1e-9, len(Y), axis=0)
    assert lb.r2(mean_pred, Y) < 0.05          # a constant predictor earns ~nothing


def test_r2_ignores_prediction_magnitude():
    Y = lb.unit(np.random.default_rng(0).normal(size=(40, 2)))
    assert lb.r2(Y * 7.3, Y) == pytest.approx(lb.r2(Y, Y), abs=1e-12)


def test_to_letter_snaps_to_nearest_axis():
    V = np.array([[0.9, 0.1], [0.1, 0.9], [-0.9, -0.1], [-0.1, -0.9]])
    assert list(lb.to_letter(V)) == [0, 2, 1, 3]      # r, u, l, d in SYMS order


def test_geometry_features_include_the_longer_axis_rule():
    req = np.array([[0.8, 0.6], [0.6, 0.8]])
    F = lb.geometry_features(req, np.array([1, 2]), np.array([True, True]))
    col = F[:, 4]                                     # |x| - |y|
    assert col[0] == pytest.approx(0.2) and col[1] == pytest.approx(-0.2)
    assert F.shape[1] == 17 + 8


def test_prompt_only_baseline_recovers_a_label_that_is_a_function_of_geometry():
    """If the label IS the target direction, a prompt-only predictor must score ~1.

    This is the guard on the whole comparison: a baseline that could not detect a
    prompt-determined label would make any decoder look good.
    """
    rng = np.random.default_rng(1)
    ang = rng.uniform(0, 2 * np.pi, 600)
    req = np.column_stack([np.cos(ang), np.sin(ang)])
    Y = lb.unit(req)
    bins = ((ang / (2 * np.pi) * 8).astype(int) % 8)
    idx = rng.permutation(600)
    te, va, tr = idx[:150], idx[150:250], idx[250:]
    base = lb.prompt_only_baselines(Y, req, bins, np.zeros(600, bool), tr, va, te)
    assert base["unfitted_target_direction"] > 0.99
    assert base["best_prompt_only"] > 0.99


def test_prompt_only_baseline_cannot_reach_a_label_independent_of_geometry():
    """A label uncorrelated with the prompt must leave the prompt-only baseline near zero."""
    rng = np.random.default_rng(2)
    ang = rng.uniform(0, 2 * np.pi, 600)
    req = np.column_stack([np.cos(ang), np.sin(ang)])
    Y = lb.unit(rng.normal(size=(600, 2)))            # independent of req
    bins = ((ang / (2 * np.pi) * 8).astype(int) % 8)
    idx = rng.permutation(600)
    te, va, tr = idx[:150], idx[150:250], idx[250:]
    base = lb.prompt_only_baselines(Y, req, bins, np.zeros(600, bool), tr, va, te)
    assert base["best_prompt_only"] < 0.15


def test_match_accuracy_counts_agreement_with_the_models_argmax():
    P = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]])
    good = np.array([[1.0, 0.0], [0.0, 1.0]])         # r, u -> both correct
    bad = np.array([[-1.0, 0.0], [0.0, 1.0]])         # l, u -> one correct
    assert lb.match_accuracy(good, P) == pytest.approx(1.0)
    assert lb.match_accuracy(bad, P) == pytest.approx(0.5)


def test_ridge_cv_picks_an_alpha_that_generalises():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 12))
    W = rng.normal(size=(12, 2))
    Y = lb.unit(X @ W + 0.1 * rng.normal(size=(400, 2)))
    idx = rng.permutation(400)
    te, va, tr = idx[:100], idx[100:180], idx[180:]
    pred, alpha = lb.ridge_cv(X[tr], Y[tr], X[va], Y[va], X[te])
    assert lb.r2(pred, Y[te]) > 0.8
    assert alpha > 0
