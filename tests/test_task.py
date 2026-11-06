import math
import re

import pytest

import random

from nc.task import (ALPHABETS, BoardConfig, INVALID_EMPTY, INVALID_LONG, INVALID_SHORT,
                     INVALID_SYMBOL, Trial, parse_reply, render_invalid, render_system,
                     render_trial_end, render_turn, shortest_path)

CFG = BoardConfig()


# ------------------------------------------------------------------ geometry
def test_bin_centres_are_unit_and_evenly_spaced():
    ts = CFG.bin_centres()
    assert len(ts) == CFG.n_targets
    for t in ts:
        assert math.hypot(*t) == pytest.approx(1.0)
    angles = sorted(math.atan2(y, x) % (2 * math.pi) for x, y in ts)
    gaps = [b - a for a, b in zip(angles, angles[1:])]
    assert all(g == pytest.approx(2 * math.pi / CFG.n_targets) for g in gaps)


def test_sampled_targets_are_random_in_angle_at_fixed_radius():
    """DESIGN 3.1: random angle kills target memorisation; fixed radius keeps difficulty
    uniform so it does not leak into the metrics the learning measure is built from."""
    rng = random.Random(0)
    ts = [CFG.sample_target(rng) for _ in range(300)]
    for t in ts:
        assert math.hypot(*t) == pytest.approx(CFG.radius)
    angles = [math.atan2(y, x) % (2 * math.pi) for x, y in ts]
    assert len(set(round(a, 6) for a in angles)) == len(ts), "angles must not repeat"
    # roughly uniform: every 45deg sector populated
    bins = {CFG.direction_bin(t) for t in ts}
    assert bins == set(range(CFG.n_targets))


def test_direction_bin_maps_the_axes_and_diagonals():
    assert CFG.direction_bin((1.0, 0.0)) == 0
    assert CFG.direction_bin((0.0, 1.0)) == 2
    assert CFG.direction_bin((-1.0, 0.0)) == 4
    assert CFG.direction_bin((0.7, 0.7)) == 1


def test_alphabet_directions_are_unit_vectors():
    for name, syms in ALPHABETS.items():
        for s, v in syms.items():
            assert math.hypot(*v) == pytest.approx(1.0), f"{name}:{s}"


def test_literal_execution_reaches_an_axis_target():
    trial = Trial(target=(CFG.radius, 0.0), cfg=CFG)
    n = 0
    while not trial.done:
        trial.apply(*trial.literal("r"))
        n += 1
    assert trial.acquired
    # radius 8, step 1, acquisition radius 1.5 -> caught at 7 moves
    assert n == math.ceil((CFG.radius - CFG.target_radius) / CFG.step)


def test_budget_exhaustion_ends_the_trial_unacquired():
    trial = Trial(target=(0.0, CFG.radius), cfg=CFG)
    while not trial.done:
        trial.apply(*trial.literal("d"))          # away from the target, forever
    assert not trial.acquired
    assert trial.moves_used == CFG.move_budget
    with pytest.raises(RuntimeError):
        trial.apply(0.0, 1.0)


def test_diagonal_target_needs_mixed_symbols():
    """The diagonal targets are the interesting ones: no single repeated symbol reaches them."""
    tx = CFG.radius * math.cos(math.pi / 4)
    trial = Trial(target=(tx, tx), cfg=CFG)
    while not trial.done:
        trial.apply(*trial.literal("r"))
    assert not trial.acquired


# ------------------------------------------------------------- angular error
@pytest.mark.parametrize("disp,expected", [
    ((1.0, 0.0), 0.0),
    ((-1.0, 0.0), math.pi),
    ((0.0, 1.0), math.pi / 2),
    ((0.0, -1.0), math.pi / 2),
])
def test_angular_error_against_the_ideal_direction(disp, expected):
    trial = Trial(target=(CFG.radius, 0.0), cfg=CFG)
    trial.apply(*disp)
    assert trial.angular_errors[0] == pytest.approx(expected, abs=1e-9)


def test_angular_error_is_nan_for_a_zero_displacement():
    trial = Trial(target=(CFG.radius, 0.0), cfg=CFG)
    trial.apply(0.0, 0.0)
    assert math.isnan(trial.angular_errors[0])
    assert trial.moves_used == 1, "a null move still consumes budget"


def test_angular_error_is_measured_before_the_move():
    """Overshooting past the target must not be scored against the post-move direction."""
    trial = Trial(target=(2.0, 0.0), cfg=BoardConfig(target_radius=0.1, radius=2.0))
    trial.apply(10.0, 0.0)
    assert trial.angular_errors[0] == pytest.approx(0.0)


# --------------------------------------------------------------- reply parsing
def test_parse_accepts_a_well_formed_reply():
    moves, reason = parse_reply("r r u l", CFG)
    assert moves == ["r", "r", "u", "l"] and reason is None


def test_parse_tolerates_surrounding_and_repeated_whitespace():
    moves, reason = parse_reply("  r   r\tu\n", CFG)
    assert moves == ["r", "r", "u"] and reason is None


@pytest.mark.parametrize("text,reason", [
    ("", INVALID_EMPTY),
    ("   ", INVALID_EMPTY),
    ("r r x", INVALID_SYMBOL),
    ("r r right", INVALID_SYMBOL),
    ("rru", INVALID_SYMBOL),               # unseparated: one token, not in the alphabet
    ("r r", INVALID_SHORT),                # k_min = 3
    (" ".join(["r"] * 13), INVALID_LONG),  # k_max = 12
])
def test_parse_rejects_and_names_the_reason(text, reason):
    moves, got = parse_reply(text, CFG)
    assert moves is None and got == reason


def test_k_min_forbids_degenerate_single_move_replies():
    """DESIGN 3.2: one read per emitted symbol, so a 1-symbol reply gives one read with no
    within-reply context variation. The validity rule must forbid them."""
    assert CFG.k_min >= 3
    assert parse_reply("r", CFG)[1] == INVALID_SHORT


# ------------------------------------------------------------------- prompts
def _unfilled(s):
    return re.findall(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", s)


def test_no_prompt_leaves_an_unfilled_placeholder():
    trial = Trial(target=(4.0, -3.0), cfg=CFG)
    texts = [
        render_system(CFG),
        render_turn(trial, 1),
        render_trial_end(trial, 1),
        *[render_invalid(r, CFG)
          for r in (INVALID_EMPTY, INVALID_SYMBOL, INVALID_SHORT, INVALID_LONG)],
    ]
    for t in texts:
        assert not _unfilled(t), t


def test_there_is_no_separate_calibration_prompt():
    """DESIGN 3.5: calibration and play must use a byte-identical template, so calibration
    is plain teacher-forcing of the reply with no extra instruction line."""
    import nc.task as task
    assert not any(n for n in dir(task) if "calib" in n.lower())


def test_turn_prompt_shows_integer_coordinates_only():
    """The model's world is a grid even though the internal state is continuous (2.1)."""
    trial = Trial(target=(5.657, 5.657), cfg=CFG)
    trial.apply(0.37, -0.44)
    text = render_turn(trial, 3)
    assert "(0, 0)" in text and "(6, 6)" in text
    coords = re.findall(r"\([^)]*\)", text)
    assert coords, text
    for c in coords:
        assert re.fullmatch(r"\(-?\d+, -?\d+\)", c), c


def test_system_prompt_never_mentions_the_mechanism():
    """The subject is told nothing about activations, decoders, layers or manifolds (2.3)."""
    text = render_system(CFG).lower()
    for word in ("activation", "decoder", "residual", "layer", "manifold", "neural", "internal"):
        assert word not in text


# ------------------------------------------------------- calibration paths
def test_shortest_path_reaches_the_target():
    """Executed literally, a calibration path acquires every target. Moves left over after
    acquisition are discarded, exactly as in play."""
    rng = random.Random(0)
    for t in [CFG.sample_target(random.Random(s)) for s in range(8)]:
        trial = Trial(target=t, cfg=CFG)
        for sym in shortest_path(trial, rng, max_len=100):
            if trial.done:
                break
            trial.apply(*trial.literal(sym))
        assert trial.acquired, t


def test_shortest_path_has_minimal_length():
    """Same multiset of moves as the L1-optimal route: no wandering."""
    rng = random.Random(1)
    trial = Trial(target=(5.0, -3.0), cfg=CFG)
    p = shortest_path(trial, rng, max_len=100)
    assert sorted(p) == sorted(["r"] * 5 + ["d"] * 3)


def test_shortest_path_order_is_randomised():
    """DESIGN 3.5: a target direction must be served by many different symbol orders, so
    order carries no information about the target."""
    trial = Trial(target=(4.0, 4.0), cfg=CFG)
    seen = {tuple(shortest_path(trial, random.Random(s), max_len=100)) for s in range(50)}
    assert len(seen) > 5
    for p in seen:
        assert sorted(p) == sorted(["r"] * 4 + ["u"] * 4)


def test_shortest_path_respects_the_length_cap():
    rng = random.Random(2)
    trial = Trial(target=(40.0, 40.0), cfg=CFG)
    assert len(shortest_path(trial, rng)) == CFG.k_max


def test_shortest_path_works_from_an_arbitrary_marker_position():
    """Calibration starts the marker all over the workspace, not just at the origin, so the
    required direction varies within a path rather than being constant."""
    rng = random.Random(3)
    trial = Trial(target=(0.0, 0.0), cfg=BoardConfig(target_radius=0.6), pos=(3.0, -2.0))
    p = shortest_path(trial, rng, max_len=100)
    assert sorted(p) == sorted(["l"] * 3 + ["u"] * 2)
