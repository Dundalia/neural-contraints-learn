"""The decoder that RUNS must be fit on the label the gate certifies (DESIGN 7.2c).

WHY THIS TEST EXISTS. The 2026-08-11 label change landed in nc/labels.py, fit_draws.py and the G1
gate, but not in scripts/run.py or scripts/screen.py. The gate therefore certified an intent-fit
decoder while the closed loop fitted the superseded forced-letter label, and 570 sessions across two
models ran a readout that scored BELOW the prompt-only baseline on its own label. Nothing failed;
the two code paths simply disagreed in silence. These tests make that disagreement fail loudly.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nc import labels as lb


def synthetic_reads(n=600, dim=40, n_layers=2, seed=0):
    """Reads whose INTENT label deliberately disagrees with the forced-letter label.

    The activations are built to encode the intent direction. A decoder fit on intent therefore
    recovers them; one fit on the letter cannot, so the two labels are distinguishable by outcome
    rather than by reading the source.
    """
    rng = np.random.default_rng(seed)
    # intent: a soft distribution favouring one letter but with real mass elsewhere
    P = rng.dirichlet(np.ones(4) * 0.6, size=n)
    order = ["r", "l", "u", "d"]
    G, _c, usable = lb.intent_direction(P, order)
    # the forced letter is the WORST of the four, so letter and intent point differently
    letters = np.array([order[i] for i in np.argmin(P, axis=1)])
    A = rng.standard_normal((2, dim))
    U = (G @ A + 0.05 * rng.standard_normal((n, dim))).astype(np.float64)
    return {
        "U_fit": np.stack([U] * n_layers), "U_val": np.stack([U] * n_layers),
        "layers": np.array([0, 1]), "fit_letter_p": P, "letter_order": np.array(order),
        "fit_next_move": letters,
    }, G, usable


def test_live_decoder_is_fit_on_intent_not_the_forced_letter():
    from run import fit_for_draw
    z, G, usable = synthetic_reads()
    coords = np.arange(z["U_fit"].shape[2])
    ro, U, _ = fit_for_draw(z, 0, coords)
    pred = ro.velocity(U)
    letter = np.array([lb.SYM_DIRS[m] for m in z["fit_next_move"]], float)[usable]
    r2_intent, r2_letter = lb.r2(pred, G[usable]), lb.r2(pred, letter)
    assert r2_intent > 0.5, f"live decoder does not track intent (R2 {r2_intent:+.3f})"
    assert r2_intent > r2_letter + 0.2, (
        f"live decoder tracks the forced letter as well as intent "
        f"(intent {r2_intent:+.3f} vs letter {r2_letter:+.3f}): the 7.2c label change has been "
        f"reverted in scripts/run.py")


def test_screen_fits_the_same_label_as_the_live_loop():
    """The perturbations rotate the screen's control space, so it must be the loop's decoder."""
    import screen as scr
    from run import fit_for_draw
    z, _G, _u = synthetic_reads()
    coords = np.arange(z["U_fit"].shape[2])
    ro_run, U, _ = fit_for_draw(z, 0, coords)
    ro_scr, _, _ = scr.fit_layer(z, coords, 0, d=4)
    a = ro_run.velocity(U)
    b = ro_scr.velocity(U)
    cos = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    assert float(np.median(cos)) > 0.99, (
        f"screen and live loop decode differently (median cos {np.median(cos):.4f}): they are "
        f"fitting different labels, so severity and reachability are screened for a decoder "
        f"that never runs")


def test_readout_label_is_the_single_definition():
    """Both call sites must route through nc.labels.readout_label, not rebuild the label."""
    for f in ("scripts/run.py", "scripts/screen.py"):
        src = (ROOT / f).read_text()
        assert "readout_label" in src, f"{f} does not use nc.labels.readout_label"
        assert 'SYM[m] for m in z["fit_next_move"]' not in src, (
            f"{f} still builds the forced-letter label inline")
