import math

import numpy as np
import pytest

from nc.manifold import (ALIGNMENT_GRID, MANIFOLD_DIM, RHO_MIN, THETA_PERT, alignment,
                         draw_coords as mf_draw, eligible_coords as mf_eligible,
                         manifold_stability as mf_stability,
                         beta_from_fa, build_perturbation, control_angles, cv_fa_loglik,
                         fit_readout, manifold_basis, reference_speed, required_pd_change,
                         set_gain, solution_density)

D, TRUE_D, N = 48, 5, 3000
MOVES = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])


def synth(seed=0, d_true=TRUE_D, noise=0.3):
    """Activations generated from a known low-dimensional structure plus private noise.

    Each sample precedes one of four moves; the move direction is embedded in the factors,
    so a decoder fit to the emitted move is recoverable -- the synthetic analog of the
    pre-move state predicting the next symbol.
    """
    rng = np.random.default_rng(seed)
    Lam = rng.standard_normal((D, d_true))
    idx = rng.integers(0, len(MOVES), N)
    G = MOVES[idx]
    A = rng.standard_normal((d_true, 2))                    # move direction -> factor space
    Z = G @ A.T + 0.5 * rng.standard_normal((N, d_true))
    U = Z @ Lam.T + noise * rng.standard_normal((N, D))
    return U, G, idx


# ----------------------------------------------------------------- FA algebra
def test_beta_matches_the_direct_high_dimensional_inverse():
    rng = np.random.default_rng(1)
    Lam = rng.standard_normal((30, 4))
    Psi = rng.uniform(0.2, 1.5, 30)
    direct = Lam.T @ np.linalg.inv(Lam @ Lam.T + np.diag(Psi))
    assert np.allclose(beta_from_fa(Lam, Psi), direct, atol=1e-8)


def test_manifold_basis_is_orthonormal_and_spans_the_readout_rowspace():
    U, G, _ = synth()
    r = fit_readout(U, G, d=6)
    Mb = manifold_basis(r.beta)
    assert Mb.shape == (6, D)
    assert np.allclose(Mb @ Mb.T, np.eye(6), atol=1e-8)
    # every row of beta is reproduced by its projection onto the basis
    assert np.allclose(r.beta @ Mb.T @ Mb, r.beta, atol=1e-8)


# -------------------------------------------------------------- decoder chain
def test_intuitive_control_space_lies_inside_the_manifold():
    """M2 = A @ beta for some A, so rowspace(M2) is a subspace of M by construction (3.3)."""
    U, G, _ = synth()
    r = fit_readout(U, G, d=6)
    assert alignment(r.M2, manifold_basis(r.beta)) == pytest.approx(1.0, abs=1e-8)


def test_velocity_equals_the_explicit_chain():
    U, G, _ = synth()
    r = fit_readout(U, G, d=6)
    manual = r.factors(U[:20]) @ r.W.T * r.gain
    assert np.allclose(r.velocity(U[:20]), manual, atol=1e-8)


def test_decoder_recovers_the_emitted_move_direction():
    U, G, _ = synth()
    r = fit_readout(U, G, d=6)
    V = r.velocity(U)
    cos = (V / np.linalg.norm(V, axis=1, keepdims=True) * G).sum(1)
    assert np.median(np.degrees(np.arccos(np.clip(cos, -1, 1)))) < 45.0


def test_gain_sets_the_median_step_length():
    U, G, _ = synth()
    r = fit_readout(U, G, d=6)
    set_gain(r, U, target_step=1.0)
    assert np.median(np.linalg.norm(r.velocity(U), axis=1)) == pytest.approx(1.0, rel=1e-6)


def test_gain_is_idempotent():
    U, G, _ = synth()
    r = fit_readout(U, G, d=6)
    set_gain(r, U, 1.0)
    assert set_gain(r, U, 1.0) == pytest.approx(r.gain, rel=1e-9)


# -------------------------------------------------------------- perturbations
@pytest.mark.parametrize("target_a", [1.0, 0.75, 0.5, 0.25, 0.05])
def test_perturbation_hits_the_requested_alignment(target_a):
    U, G, _ = synth()
    r = fit_readout(U, G, d=8)
    Mb = manifold_basis(r.beta)
    theta = math.radians(80.0)                        # sin^2 = 0.97, so a >= 0.03 is reachable
    M2P = build_perturbation(r.M2, Mb, theta, target_a, np.random.default_rng(3))
    assert alignment(M2P, Mb) == pytest.approx(target_a, abs=1e-6)


@pytest.mark.parametrize("deg", [30.0, 60.0, 80.0])
def test_both_principal_angles_to_the_intuitive_space_equal_theta(deg):
    """This is what matches the arms on the paper's Fig-3b criterion: the same
    intuitive-to-perturbed angle at every alignment level."""
    U, G, _ = synth()
    r = fit_readout(U, G, d=8)
    Mb = manifold_basis(r.beta)
    theta = math.radians(deg)
    for a in (1.0, 0.5):
        if (1 - a) > math.sin(theta) ** 2:
            continue
        M2P = build_perturbation(r.M2, Mb, theta, a, np.random.default_rng(4))
        assert np.allclose(control_angles(r.M2, M2P), theta, atol=1e-6)


def test_within_manifold_perturbation_stays_exactly_inside():
    U, G, _ = synth()
    r = fit_readout(U, G, d=8)
    Mb = manifold_basis(r.beta)
    M2P = build_perturbation(r.M2, Mb, math.radians(60.0), 1.0, np.random.default_rng(5))
    assert alignment(M2P, Mb) == pytest.approx(1.0, abs=1e-8)
    assert not np.allclose(M2P, r.M2)                 # it really did move


def test_unreachable_alignment_raises_rather_than_silently_missing():
    """The alignment axis and the matched-angle control are coupled: a = 0 forces theta = 90
    degrees. The constructor must refuse instead of returning something off-target."""
    U, G, _ = synth()
    r = fit_readout(U, G, d=8)
    Mb = manifold_basis(r.beta)
    with pytest.raises(ValueError, match="unreachable"):
        build_perturbation(r.M2, Mb, math.radians(30.0), 0.0, np.random.default_rng(6))


def test_full_orthogonality_is_reachable_at_ninety_degrees():
    U, G, _ = synth()
    r = fit_readout(U, G, d=8)
    Mb = manifold_basis(r.beta)
    M2P = build_perturbation(r.M2, Mb, math.radians(90.0), 0.0, np.random.default_rng(7))
    assert alignment(M2P, Mb) == pytest.approx(0.0, abs=1e-6)


def test_small_factor_dimensionality_is_refused_not_fudged():
    """Rotating inside the manifold needs two spare directions there, i.e. d >= 4."""
    U, G, _ = synth()
    r = fit_readout(U, G, d=3)
    Mb = manifold_basis(r.beta)
    with pytest.raises(ValueError, match="independent directions"):
        build_perturbation(r.M2, Mb, math.radians(60.0), 1.0, np.random.default_rng(8))


def test_perturbation_is_deterministic_given_a_seed():
    U, G, _ = synth()
    r = fit_readout(U, G, d=8)
    Mb = manifold_basis(r.beta)
    kw = (r.M2, Mb, math.radians(70.0), 0.5)
    a = build_perturbation(*kw, np.random.default_rng(11))
    b = build_perturbation(*kw, np.random.default_rng(11))
    assert np.allclose(a, b)


# -------------------------------------------------- matched-difficulty diagnostics
def test_required_pd_change_is_zero_for_no_perturbation():
    U, G, idx = synth()
    r = fit_readout(U, G, d=6)
    u_B = np.stack([r.z(U[idx == i]).mean(0) for i in range(len(MOVES))])
    angles = np.arctan2(MOVES[:, 1], MOVES[:, 0])
    assert required_pd_change(r.M2, r.M2, u_B, angles) == pytest.approx(0.0, abs=1e-6)


def test_required_pd_change_is_positive_and_bounded_for_a_real_perturbation():
    U, G, idx = synth()
    r = fit_readout(U, G, d=8)
    Mb = manifold_basis(r.beta)
    u_B = np.stack([r.z(U[idx == i]).mean(0) for i in range(len(MOVES))])
    angles = np.arctan2(MOVES[:, 1], MOVES[:, 0])
    M2P = build_perturbation(r.M2, Mb, math.radians(60.0), 0.5, np.random.default_rng(9))
    got = required_pd_change(r.M2, M2P, u_B, angles)
    assert 0.0 < got <= math.pi


def test_solution_density_is_high_under_the_intuitive_decoder():
    """Every target direction must be reachable by some emission -- the sanity end of the
    reachability screen. Under the intuitive decoder this is near-trivially satisfied."""
    U, G, _ = synth()
    r = fit_readout(U, G, d=6)
    set_gain(r, U, 1.0)
    rho = solution_density(U, r, r.M2, MOVES, reference_speed(r, U))
    assert rho.shape == (4,)
    assert np.all((rho >= 0.0) & (rho <= 1.0))
    assert rho.min() > RHO_MIN


def test_solution_density_uses_a_fixed_speed_yardstick():
    """A perturbation producing uniformly tiny velocities must FAIL the screen. Scaling the
    floor by the perturbed decoder's own median would shrink the bar with it and let such a
    perturbation through -- the exact failure the screen exists to catch."""
    U, G, _ = synth()
    r = fit_readout(U, G, d=6)
    set_gain(r, U, 1.0)
    ref = reference_speed(r, U)
    crippled = r.M2 * 0.05                      # same directions, 20x too slow
    assert solution_density(U, r, crippled, MOVES, ref).max() == 0.0
    # ... whereas a self-referential floor would score it as if nothing were wrong
    self_ref = reference_speed(r, U) * 0.05
    assert solution_density(U, r, crippled, MOVES, self_ref).min() > RHO_MIN


def test_solution_density_falls_as_the_control_space_leaves_the_manifold():
    """The quantity the screen gates on must actually be sensitive to alignment, otherwise
    it certifies nothing."""
    U, G, _ = synth()
    r = fit_readout(U, G, d=8)
    Mb = manifold_basis(r.beta)
    set_gain(r, U, 1.0)
    ref = reference_speed(r, U)
    inside = build_perturbation(r.M2, Mb, THETA_PERT, 1.0, np.random.default_rng(10))
    outside = build_perturbation(r.M2, Mb, THETA_PERT, 0.25, np.random.default_rng(10))
    assert solution_density(U, r, inside, MOVES, ref).min() >= 0.0
    assert alignment(outside, Mb) < alignment(inside, Mb)


# ------------------------------------------------------------ frozen constants
def test_every_grid_level_has_an_admissible_theta():
    """theta is searched, not fixed (DESIGN 4.6). Each alignment level must have at least one
    candidate angle that both reaches it and avoids the floor degeneracy."""
    from nc.manifold import admissible_thetas
    missing = [a for a in ALIGNMENT_GRID if not admissible_thetas(a)]
    assert not missing, f"no admissible theta for alignment {missing} -- truncate the grid"


def test_admissible_theta_excludes_the_degenerate_floor():
    """At the floor the whole rotation goes out of the manifold and the perturbation becomes an
    attenuated copy of the original decoder. That angle must be refused for that level."""
    from nc.manifold import admissible_thetas
    floor_a = math.cos(math.radians(60.0)) ** 2          # = 0.25, the floor at theta = 60
    assert math.radians(60.0) not in admissible_thetas(floor_a)
    # ... but a level comfortably above that floor may use 60 deg
    assert math.radians(60.0) in admissible_thetas(0.6)


def test_lower_alignment_requires_larger_theta():
    from nc.manifold import admissible_thetas
    mins = [min(admissible_thetas(a)) for a in (0.9, 0.7, 0.4, 0.2)]
    assert mins == sorted(mins), "the minimum admissible angle must rise as alignment falls"


def test_every_grid_level_builds_and_hits_its_alignment():
    from nc.manifold import admissible_thetas
    U, G, _ = synth()
    r = fit_readout(U, G, d=MANIFOLD_DIM)
    Mb = manifold_basis(r.beta)
    for a in ALIGNMENT_GRID:
        th = min(admissible_thetas(a))
        M2P = build_perturbation(r.M2, Mb, th, a, np.random.default_rng(21))
        assert alignment(M2P, Mb) == pytest.approx(a, abs=1e-6)
        assert np.allclose(control_angles(r.M2, M2P), th, atol=1e-6)


def test_coordinate_draws_are_random_and_exclude_outliers():
    """A QUALITY screen only (DESIGN 4.1): massive-activation dims are excluded so one cannot
    dominate a draw's covariance. Nothing is selected for being informative."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 300))
    X[:, 7] *= 5000.0                                    # a synthetic attention-sink dim
    X[:, 99] *= 5000.0
    elig = mf_eligible(X)
    assert 7 not in elig and 99 not in elig
    assert len(elig) >= 290
    d1 = mf_draw(elig, 100, np.random.default_rng(1))
    d2 = mf_draw(elig, 100, np.random.default_rng(2))
    assert len(set(d1)) == 100 and set(d1) <= set(elig.tolist())
    assert not np.array_equal(d1, d2), "different seeds must give different populations"


def test_manifold_stability_is_high_when_data_is_sufficient():
    """The split-half check replaces the arbitrary N/D ratio revision 5 asserted."""
    U, _, _ = synth(seed=5, d_true=6, noise=0.4)
    st = mf_stability(U[:, :100], MANIFOLD_DIM, np.random.default_rng(0))
    assert 0.0 <= st["mean_cos2"] <= 1.0
    assert st["max_angle_deg"] <= 90.0


def test_manifold_dim_leaves_room_for_a_within_manifold_rotation():
    assert MANIFOLD_DIM >= 4


# ------------------------------------------------------------ dimensionality
def test_cross_validated_likelihood_peaks_near_the_true_dimensionality():
    U, _, _ = synth(seed=2, d_true=4, noise=0.5)
    ll = cv_fa_loglik(U, dims=[1, 2, 3, 4, 5, 6, 8, 12], folds=4)
    best = max(ll, key=lambda d: ll[d][0])
    assert 3 <= best <= 6, ll
    assert all(len(v) == 2 and np.isfinite(v[0]) for v in ll.values())
