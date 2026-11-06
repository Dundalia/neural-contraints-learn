"""Manifold estimation, the velocity decoder, and perturbation construction.

See docs/DESIGN.md §3. The chain is

    u  --(z-score)-->  uz  --(beta)-->  z  --(s_z)-->  zs  --(W)-->  velocity

and the intrinsic manifold is ``M = rowspace(beta)``, the d-dimensional subspace of the
ambient space that the factor readout can access. Any control space assembled from the
factors lies inside M by construction, which is the property the perturbations manipulate.

Perturbations are built by rotation rather than permutation (§3.6). Given the intuitive
control space and a target manifold alignment, :func:`build_perturbation` rotates the
control space by a fixed angle ``theta`` while splitting the rotation direction between
inside-M and outside-M components, so that

    angle(intuitive, perturbed) = theta   for both principal angles
    alignment(perturbed, M)     = 1 - sin^2(theta) sin^2(psi)

with ``psi`` solving for the requested alignment. **The two axes are not independent**: a
fully orthogonal control space (alignment 0) forces theta = 90 degrees, so holding the
intuitive-to-perturbed angle fixed -- the paper's matched-principal-angle control -- bounds
the reachable alignment range at ``1 - sin^2(theta)``. Callers pick theta accordingly and
:func:`build_perturbation` raises rather than silently missing the requested alignment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import subspace_angles

# --- frozen design constants (docs/DESIGN.md) --------------------------------------------
#: Manifold dimensionality. FIXED A PRIORI, as the paper fixed d = 10 for every session and
#: ran its cross-validated sweep only retrospectively. Not chosen from our data: the paper's
#: ED Fig 7b shows the cross-validated estimate still GROWING with sample count at 1600
#: points over 90 coordinates, so peak-picking at our 30k samples over 4096 coordinates would
#: be estimating a different, sample-size-dependent quantity. We report the curve as a check.
MANIFOLD_DIM = 10
#: Reference angle, used by unit tests and by the alignment-floor worked examples. The SCREEN
#: DOES NOT USE THIS -- theta is searched over THETA_GRID (below).
THETA_PERT = math.radians(60.0)
#: Candidate angles the screen searches. Theta is NOT fixed (DESIGN 4.6, measured 2026-08-10):
#: severity is the quantity that must match across arms (it is the Fig-3a initial-impairment
#: proxy) and theta is only a means to it. Sadtler screened into a severity window too and
#: reported the resulting angles afterwards. Fixing theta produced severity that CO-VARIED with
#: alignment (43 deg at a=1.0 down to 3 deg at a=0.25), i.e. the opposite of matched difficulty.
THETA_GRID = tuple(math.radians(d) for d in (50.0, 60.0, 70.0, 78.0, 84.0, 87.0))
#: Median per-sample velocity-change angle a candidate must produce. The paper's window.
SEVERITY_WINDOW = (math.radians(20.0), math.radians(45.0))
#: Minimum cos(psi): the fraction of the rotation direction that lies INSIDE the manifold.
#: This is the anti-degeneracy constraint, and it belongs here rather than as a margin on
#: alignment. At the alignment floor psi = 90 deg, so cos(psi) = 0: the entire rotation goes
#: out of the manifold and the in-manifold part of the perturbed control space is cos(theta) x
#: the ORIGINAL direction -- scaled, not rotated. The result is "same mapping, slower" rather
#: than a perturbation (measured: 3 deg of velocity change, 0/200 candidates admissible).
#: Requiring cos(psi) >= 0.3 forces at least some genuine within-manifold reorientation.
MIN_INMANIFOLD_ROT = 0.30
#: Alignment levels (DESIGN 4.6). Extends to 0.1; a = 0.1 needs theta > arccos(sqrt(0.1)) = 72
#: deg merely to exist, and more than that once ALIGNMENT_MARGIN is applied.
ALIGNMENT_GRID = (1.0, 0.7, 0.4, 0.2, 0.1)
#: Population size per draw -- Sadtler's 85-91 units. See DESIGN 4.1 for why not all d_model.
N_COORDS = 100
#: Draws (one per session; a draw is the analog of one recording day).
N_DRAWS = 20
#: Reachability floor: every target direction must be served by at least this fraction of
#: sampled emissions. Set so that a uniform emission sample of 5000 yields >= 50 solutions
#: per target, enough that the model could plausibly stumble on one.
RHO_MIN = 0.01


def admissible_thetas(target_alignment: float, thetas=THETA_GRID) -> list[float]:
    """Which candidate angles can build this alignment WITHOUT degenerating.

    Two conditions. Reachability: ``1 - a <= sin^2(theta)``, else the alignment is below the
    floor and cannot be built at all. Non-degeneracy: ``cos^2(psi) = 1 - (1-a)/sin^2(theta)
    >= MIN_INMANIFOLD_ROT^2``, so a real share of the rotation reorients the control space
    *within* the manifold instead of merely tilting it out and leaving a scaled copy behind.

    Returns them ascending, so a caller preferring the mildest perturbation takes the first.
    An empty list means this alignment level must be dropped from the grid -- the doc's rule is
    to truncate the grid, never to relax the windows.
    """
    out = []
    for th in thetas:
        s2 = math.sin(th) ** 2
        if (1.0 - target_alignment) > s2 + 1e-12:
            continue
        cos2_psi = 1.0 - (1.0 - target_alignment) / max(s2, 1e-12)
        if cos2_psi >= MIN_INMANIFOLD_ROT ** 2 - 1e-12:
            out.append(th)
    return sorted(out)


# ------------------------------------------------------------ coordinate draws
def eligible_coords(U: np.ndarray, factor: float = 20.0) -> np.ndarray:
    """Indices of coordinates eligible for a draw, excluding massive-activation dims.

    A QUALITY screen, not an informativeness screen (DESIGN 4.1) -- the analog of Sadtler
    discarding electrodes whose threshold crossings did not resemble action potentials. A few
    residual dimensions (attention sinks) carry values orders of magnitude above the rest; one
    landing in a draw would dominate that draw's covariance and make it behave unlike the
    others. Nothing is selected FOR being informative, which is where selection artifacts live.

    Rule: drop coordinates whose median absolute value exceeds ``factor`` x the median of that
    statistic across coordinates. Frozen before any fit; the count is reported.
    """
    scale = np.median(np.abs(np.asarray(U, dtype=np.float64)), axis=0)
    return np.flatnonzero(scale <= factor * float(np.median(scale)))


def draw_coords(eligible: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` coordinates drawn uniformly at random without replacement. No ranking."""
    if len(eligible) < n:
        raise ValueError(f"only {len(eligible)} eligible coordinates, need {n}")
    return np.sort(rng.choice(eligible, size=n, replace=False))


def manifold_stability(U: np.ndarray, d: int, rng: np.random.Generator) -> dict:
    """Split-half agreement of the manifold: fit FA on each half, compare the subspaces.

    This is the direct test of whether we have enough samples -- it replaces the arbitrary
    N/D ratio that revision 5 asserted without derivation. Reports the principal angles
    between the two d-dimensional manifolds; small angles mean the estimate is data-sufficient.
    """
    from sklearn.decomposition import FactorAnalysis

    U = np.asarray(U, dtype=np.float64)
    idx = rng.permutation(len(U))
    halves = []
    for part in (idx[: len(idx) // 2], idx[len(idx) // 2:]):
        X = U[part]
        mu, sd = X.mean(0), np.where(X.std(0) < 1e-12, 1.0, X.std(0))
        fa = FactorAnalysis(n_components=d, random_state=0).fit((X - mu) / sd)
        halves.append(manifold_basis(beta_from_fa(fa.components_.T,
                                                 np.maximum(fa.noise_variance_, 1e-8))))
    ang = subspace_angles(halves[0].T, halves[1].T)
    return {"max_angle_deg": float(np.degrees(ang.max())),
            "median_angle_deg": float(np.degrees(np.median(ang))),
            "mean_cos2": float(np.mean(np.cos(ang) ** 2))}


# --------------------------------------------------------------------- fitting
def beta_from_fa(Lambda: np.ndarray, Psi: np.ndarray) -> np.ndarray:
    """Factor readout ``beta = Lambda^T (Lambda Lambda^T + Psi)^-1``, computed in the
    equivalent ``(I + Lambda^T Psi^-1 Lambda)^-1 Lambda^T Psi^-1`` form so the only
    inverse is d x d rather than D x D (Woodbury). ``Lambda`` is (D, d), ``Psi`` is (D,)."""
    d = Lambda.shape[1]
    LtPi = Lambda.T / Psi                                  # (d, D)
    return np.linalg.solve(np.eye(d) + LtPi @ Lambda, LtPi)  # (d, D)


@dataclass
class Readout:
    """Everything needed to turn a raw activation into a velocity. Frozen at calibration."""
    mu: np.ndarray       # (D,)  per-coordinate calibration mean
    sd: np.ndarray       # (D,)  per-coordinate calibration sd
    Lambda: np.ndarray   # (D, d) FA loadings
    Psi: np.ndarray      # (D,)  diagonal private variances
    beta: np.ndarray     # (d, D) factor readout
    s_z: np.ndarray      # (d,)  1/sd of each factor over calibration
    W: np.ndarray        # (2, d) velocity map from z-scored factors
    gain: float = 1.0

    def z(self, U: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(U) - self.mu) / self.sd

    def factors(self, U: np.ndarray) -> np.ndarray:
        return (self.z(U) @ self.beta.T) * self.s_z

    @property
    def M2(self) -> np.ndarray:
        """(2, D). ``velocity = z(u) @ M2.T``. Row space lies inside M by construction."""
        return self.gain * (self.W * self.s_z) @ self.beta

    def velocity(self, U: np.ndarray, M2: np.ndarray | None = None) -> np.ndarray:
        """(N, 2) velocities. Pass ``M2`` to decode through a perturbed control space."""
        return self.z(U) @ (self.M2 if M2 is None else M2).T


def fit_readout(U: np.ndarray, G: np.ndarray, d: int, alpha: float = 1.0,
                random_state: int = 0) -> Readout:
    """Fit the whole chain on calibration data.

    ``U`` is (N, D) raw activations at pre-move positions; ``G`` is (N, 2) unit vectors of
    the move each row precedes. The decoder is fit to the **emitted move**, not to
    target-directed intent (§3.4): the target is in the prompt, so an intent-fit decoder
    could score well by reading the context while the model contributes nothing.
    """
    from sklearn.decomposition import FactorAnalysis

    U = np.asarray(U, dtype=np.float64)
    mu, sd = U.mean(0), U.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Uz = (U - mu) / sd

    fa = FactorAnalysis(n_components=d, random_state=random_state).fit(Uz)
    Lambda, Psi = fa.components_.T, np.maximum(fa.noise_variance_, 1e-8)
    beta = beta_from_fa(Lambda, Psi)

    Z = Uz @ beta.T
    zsd = Z.std(0)
    s_z = 1.0 / np.where(zsd < 1e-12, 1.0, zsd)
    Zs = Z * s_z

    W = fit_decoder(Zs, np.asarray(G, dtype=np.float64), alpha)
    return Readout(mu=mu, sd=sd, Lambda=Lambda, Psi=Psi, beta=beta, s_z=s_z, W=W)


def fit_decoder(Zs: np.ndarray, G: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Ridge map (2, d) from z-scored factors to move direction."""
    d = Zs.shape[1]
    return np.linalg.solve(Zs.T @ Zs + alpha * np.eye(d), Zs.T @ G).T


def set_gain(readout: Readout, U: np.ndarray, target_step: float) -> float:
    """Scale so the median decoded step length equals ``target_step``. Set once on the
    intuitive decoder and held fixed across every perturbed decoder (§3.4)."""
    readout.gain = 1.0
    med = float(np.median(np.linalg.norm(readout.velocity(U), axis=1)))
    readout.gain = target_step / max(med, 1e-12)
    return readout.gain


# ------------------------------------------------------------------- geometry
def manifold_basis(beta: np.ndarray) -> np.ndarray:
    """(d, D) orthonormal rows spanning ``M = rowspace(beta)``."""
    return np.linalg.svd(beta, full_matrices=False)[2]


def alignment(M2P: np.ndarray, Mb: np.ndarray) -> float:
    """Mean cos^2 of the principal angles between ``rowspace(M2P)`` and M. 1 = inside."""
    ang = subspace_angles(np.asarray(M2P).T, np.asarray(Mb).T)
    return float(np.mean(np.cos(ang) ** 2))


def control_angles(M2A: np.ndarray, M2B: np.ndarray) -> np.ndarray:
    """Principal angles (radians) between two control spaces -- the paper's Fig 3b check."""
    return subspace_angles(np.asarray(M2A).T, np.asarray(M2B).T)


def _rand_orthonormal(X: np.ndarray, k: int) -> np.ndarray:
    """``k`` orthonormal columns spanning directions available in ``colspace(X)``.

    Uses the SVD rather than QR because the inputs here are deliberately rank-deficient
    (a subspace projected onto the complement of another), and QR would hand back
    arbitrary null-space columns that silently violate the intended constraints.
    """
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    keep = U[:, s > 1e-8 * max(float(s[0]) if s.size else 0.0, 1e-30)]
    if keep.shape[1] < k:
        raise ValueError(
            f"only {keep.shape[1]} independent directions available, need {k} "
            "(is the factor dimensionality d >= 4?)"
        )
    return keep[:, :k]


def build_perturbation(M2: np.ndarray, Mb: np.ndarray, theta: float,
                       target_alignment: float, rng: np.random.Generator) -> np.ndarray:
    """Rotate the control space by ``theta`` (radians) to a prescribed manifold alignment.

    ``target_alignment = 1`` keeps it inside M (the within-manifold arm, the continuous
    generalisation of Sadtler's factor permutation); lower values tilt it out. Both
    principal angles to the intuitive control space equal ``theta`` regardless, which is
    what matches the arms on the paper's Fig-3b criterion.
    """
    if not 0.0 <= target_alignment <= 1.0:
        raise ValueError("target_alignment must be in [0, 1]")
    st = math.sin(theta)
    if (1.0 - target_alignment) > st * st + 1e-12:
        raise ValueError(
            f"alignment {target_alignment:.3f} unreachable at theta={math.degrees(theta):.1f} deg: "
            f"needs 1-a <= sin^2(theta) = {st*st:.3f}. Increase theta (a=0 requires 90 deg)."
        )
    sin_psi = 0.0 if st < 1e-12 else math.sqrt(max(0.0, 1.0 - target_alignment)) / st
    cos_psi = math.sqrt(max(0.0, 1.0 - sin_psi ** 2))

    Q, R = np.linalg.qr(np.asarray(M2, dtype=np.float64).T)   # Q (D,2) spans rowspace(M2)
    MbT = np.asarray(Mb, dtype=np.float64).T                  # (D, d)
    D = Q.shape[0]
    zero = np.zeros((D, 2))

    # Inside-M directions, with the current control space projected out first -- so any
    # mixture of them is automatically orthogonal to Q. Needs d >= 4.
    in_M = (MbT - Q @ (Q.T @ MbT)) @ rng.standard_normal((MbT.shape[1], 10))
    m = _rand_orthonormal(in_M, 2) if cos_psi > 1e-12 else zero
    # Outside-M directions. Orthogonal to Q for free, since rowspace(M2) is a subspace of M.
    Y = rng.standard_normal((D, 10))
    out_M = Y - MbT @ (MbT.T @ Y)
    o = _rand_orthonormal(out_M, 2) if sin_psi > 1e-12 else zero

    n = cos_psi * m + sin_psi * o
    Qp = math.cos(theta) * Q + st * n
    return R.T @ Qp.T


# -------------------------------------------------- matched-difficulty diagnostics
def required_pd_change(M2: np.ndarray, M2P: np.ndarray, u_B: np.ndarray,
                       target_angles: np.ndarray) -> float:
    """Mean absolute required change in per-coordinate preferred direction (paper Fig 3c).

    Uses the paper's minimal-change closed form for the activations that would restore the
    intuitive open-loop velocity under ``M2P``, then refits cosine tuning per coordinate and
    reports the mean wrapped angular change.

    The cosine fit is over **required-direction bins** (the paper: over its 8 *targets*, eq
    22, where ``theta_i`` is "the direction of the ith target") -- NOT over the move symbols.
    ``u_B`` is (n_bins, D) z-scored mean activations and ``target_angles`` the corresponding
    directions in radians. Fitting over the 4 symbols instead would compute a different
    quantity; see docs/DESIGN.md deviations.
    """
    M2, M2P, u_B = (np.asarray(x, dtype=np.float64) for x in (M2, M2P, u_B))
    delta = (M2 - M2P) @ u_B.T                                     # (2, n)
    corr = M2P.T @ np.linalg.solve(M2P @ M2P.T, delta)             # (D, n)
    u_P = u_B + corr.T
    return float(np.mean(_wrap(_pref_dir(u_P, target_angles) - _pref_dir(u_B, target_angles))))


def _pref_dir(U: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Per-coordinate preferred direction from a cosine tuning fit over targets."""
    X = np.column_stack([np.cos(angles), np.sin(angles), np.ones_like(angles)])
    A = np.linalg.lstsq(X, U, rcond=None)[0]                       # (3, D)
    return np.arctan2(A[1], A[0])


def _wrap(a: np.ndarray) -> np.ndarray:
    """Absolute angular difference wrapped into [0, pi]."""
    return np.abs((a + np.pi) % (2 * np.pi) - np.pi)


def reference_speed(readout: Readout, U: np.ndarray) -> float:
    """Median decoded step length under the INTUITIVE decoder. The fixed speed yardstick
    for :func:`solution_density` -- see the warning in that docstring."""
    return float(np.median(np.linalg.norm(readout.velocity(U), axis=1)))


def solution_density(U: np.ndarray, readout: Readout, M2P: np.ndarray,
                     target_dirs: np.ndarray, ref_speed: float,
                     max_angle: float = math.radians(30.0),
                     min_speed_frac: float = 0.6) -> np.ndarray:
    """Reachability screen: fraction of sampled states that decode to within ``max_angle``
    of each target direction at adequate speed. One fraction per target; the screen requires
    the minimum over targets to clear ``RHO_MIN`` and matches it across arms, so that an
    off-manifold failure is unlearnability and not unreachability.

    ``ref_speed`` MUST be the intuitive decoder's median step (:func:`reference_speed`), not
    the perturbed decoder's own median. Scaling the floor by the perturbed decoder's own
    speed would let a perturbation that produces uniformly tiny velocities pass the screen,
    since the bar would shrink with it -- exactly the failure the screen exists to catch.

    NOTE (design §4.6): this samples the *emission space*, so it certifies GEOMETRIC
    reachability -- that solutions exist among sequences the model could in principle write.
    It does not certify POLICY reachability, i.e. that the model's actual emissions visit
    them. That is what the exploration-breadth measures test."""
    V = readout.velocity(U, M2P)
    speed = np.linalg.norm(V, axis=1)
    floor = min_speed_frac * float(ref_speed)
    unit = V / np.maximum(speed, 1e-12)[:, None]
    cos_min = math.cos(max_angle)
    return np.array([
        float(np.mean((unit @ t >= cos_min) & (speed >= floor)))
        for t in np.asarray(target_dirs, dtype=np.float64)
    ])


def cv_fa_loglik(U: np.ndarray, dims, folds: int = 4, random_state: int = 0) -> dict:
    """Cross-validated FA log-likelihood per dimensionality -- the paper's Fig 4a-c
    procedure, used to pick d and to report the estimated intrinsic dimensionality."""
    from sklearn.decomposition import FactorAnalysis
    from sklearn.model_selection import KFold

    U = np.asarray(U, dtype=np.float64)
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    out = {}
    for d in dims:
        scores = [
            FactorAnalysis(n_components=d, random_state=random_state).fit(U[tr]).score(U[te])
            for tr, te in kf.split(U)
        ]
        out[int(d)] = (float(np.mean(scores)), float(np.std(scores) / math.sqrt(folds)))
    return out
