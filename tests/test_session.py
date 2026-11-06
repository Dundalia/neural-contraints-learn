"""Closed-loop session-runner tests, against a mock backend (no GPU).

`nc/session.py` is what consumes GPU time in G3/G4, and its logic is easy to get subtly wrong
in ways the outputs would not reveal -- which already happened once: a single block schedule was
applied to every session in a lockstep batch, so every session would silently have run the first
one's perturbation. These tests pin the behaviours that failure class lives in.
"""
import random

import numpy as np
import pytest
import torch

from nc.manifold import Readout
from nc.session import Block, SessionState, run_lockstep
from nc.task import BoardConfig

D, HID = 6, 32


class MockBackend:
    """Replies come from a scripted list (cycled); reads are deterministic in the reply."""

    def __init__(self, replies, hidden=HID):
        self.replies = list(replies)
        self.hidden_size = hidden
        self.n_generate = 0
        self.read_spans = []
        self.extra_lefts = []

    def generate_batch(self, convs, **kw):
        self.n_generate += 1
        return [self.replies[(self.n_generate - 1 + i) % len(self.replies)]
                for i in range(len(convs))]

    def read_residual_multi_batch(self, items, layers, position=None, extra_left=0, **kw):
        out = []
        for conv, reply in items:
            k = len(reply.split())
            self.read_spans.append(k)
            self.extra_lefts.append(extra_left)
            # Mirror the real backend: extra_left widens the span leftwards, so the caller gets
            # k + extra_left rows and row j is the state that generated letter j.
            n = k + extra_left
            base = np.arange(n, dtype=np.float32)[:, None] + np.arange(self.hidden_size)[None, :]
            out.append(torch.from_numpy(base[None, :, :].copy()))   # (1, k+extra_left, hidden)
        return out


def make_readout(direction=(1.0, 0.0)):
    """A readout whose decoded velocity is a fixed direction of length 1, for any input."""
    rng = np.random.default_rng(0)
    Lam = rng.standard_normal((HID, D))
    Psi = np.ones(HID)
    from nc.manifold import beta_from_fa
    beta = beta_from_fa(Lam, Psi)
    W = np.zeros((2, D))
    return Readout(mu=np.zeros(HID), sd=np.ones(HID), Lambda=Lam, Psi=Psi, beta=beta,
                   s_z=np.ones(D), W=W, gain=1.0), np.array(direction, dtype=float)


class FixedReadout(Readout):
    """Decodes every state to the same velocity -- lets a test drive the marker exactly."""

    def __init__(self, vel):
        self.vel = np.asarray(vel, dtype=float)
        r, _ = make_readout()
        for k, v in r.__dict__.items():
            setattr(self, k, v)

    def velocity(self, U, M2=None):
        n = np.atleast_2d(U).shape[0]
        v = self.vel if M2 is None else self.vel[::-1] * np.array([1.0, -1.0])
        return np.tile(v, (n, 1))


def mk_session(idx, blocks, vel=(1.0, 0.0), cfg=None, seed=0):
    cfg = cfg or BoardConfig()
    return SessionState(idx=idx, cfg=cfg, readout=FixedReadout(vel),
                        coords=np.arange(HID), rng=random.Random(seed), blocks=blocks)


# ------------------------------------------------------------ displacement indexing
def test_k_letter_reply_produces_k_displacements():
    """One applied move per emitted letter (re-indexed 2026-08-12).

    This test previously asserted k-1, encoding the belief that the last prompt position should be
    skipped because "its decoded velocity would be a pure function of the prompt". That reasoning
    was wrong: that position is precisely the state that GENERATED the first letter, so skipping it
    meant the model's first move was never executed and roughly 15% of every plan was dropped from
    the front. The runner now asks the backend for one extra position on the left (extra_left=1) and
    applies rows 0..k-1.
    """
    b = MockBackend(["r r r r r"])                     # 5 letters
    s = mk_session(0, [Block("baseline", 1)])
    run_lockstep(b, [s], layer=0, verbose=False)
    first = s.turns[0]
    assert first["n_moves"] == 5, "5 letters must give 5 displacements"
    assert b.extra_lefts and all(x == 1 for x in b.extra_lefts), (
        f"the runner must request one extra left position, got {b.extra_lefts}")


def test_reply_too_short_to_yield_a_displacement_is_not_read():
    b = MockBackend(["r"])                              # invalid: below k_min
    s = mk_session(0, [Block("baseline", 1)])
    run_lockstep(b, [s], layer=0, verbose=False, stuck_limit=2)
    assert s.turns[0]["invalid"] is not None
    assert s.turns[0]["n_moves"] == 0
    assert b.read_spans == [], "an invalid reply must not be read"


# ------------------------------------------------------------------ block routing
def test_each_session_uses_its_own_block_schedule():
    """The regression that motivated this file: with one shared schedule, every session in a
    batch silently ran the first session's perturbation."""
    M2_a = np.zeros((2, HID))
    M2_b = np.ones((2, HID))
    s0 = mk_session(0, [Block("baseline", 1), Block("perturbation", 1, M2_a, 1.0)])
    s1 = mk_session(1, [Block("baseline", 1), Block("perturbation", 1, M2_b, 0.2)])
    b = MockBackend(["r r r r r r r r"])
    run_lockstep(b, [s0, s1], layer=0, verbose=False)
    a0 = {t["alignment"] for t in s0.turns if t["block"] == "perturbation"}
    a1 = {t["alignment"] for t in s1.turns if t["block"] == "perturbation"}
    assert a0 == {1.0} and a1 == {0.2}, f"schedules leaked: {a0} {a1}"


def test_block_boundaries_follow_trial_number():
    blocks = [Block("baseline", 2), Block("perturbation", 3), Block("washout", 1)]
    s = mk_session(0, blocks)
    b = MockBackend(["r r r r r r r r r r r r"])       # 12 letters: reaches a target fast
    run_lockstep(b, [s], layer=0, verbose=False)
    seen = {}
    for t in s.turns:
        seen.setdefault(t["block"], set()).add(t["trial"])
    assert seen["baseline"] <= {1, 2}
    assert seen["perturbation"] <= {3, 4, 5}
    assert seen.get("washout", set()) <= {6}


def test_session_stops_after_its_total_trials():
    s = mk_session(0, [Block("baseline", 3)])
    b = MockBackend(["r r r r r r r r r r r r", "r r r r r r r r r r r", "r r r r r r r r r r"])
    run_lockstep(b, [s], layer=0, verbose=False)
    assert max(t["trial"] for t in s.turns) == 3
    assert len(s.done_trials) == 3


# ------------------------------------------------------------------- movement
def test_marker_moves_in_the_decoded_direction_and_acquires():
    s = mk_session(0, [Block("baseline", 1)], vel=(1.0, 0.0))
    s.rng = random.Random(0)
    b = MockBackend(["r r r r r r r r r r r r"])
    run_lockstep(b, [s], layer=0, verbose=False)
    xs = [t["pos_after"][0] for t in s.turns]
    assert xs[0] > 0, "a decoded +x velocity must move the marker in +x"


def test_a_trial_that_never_arrives_ends_on_the_move_budget():
    cfg = BoardConfig(move_budget=10)
    s = mk_session(0, [Block("baseline", 1)], vel=(0.0, 0.0), cfg=cfg)
    b = MockBackend(["r r r r r r r r r r r r"])
    run_lockstep(b, [s], layer=0, verbose=False, stuck_limit=99)
    assert s.done_trials[0]["moves_used"] == cfg.move_budget
    assert not s.done_trials[0]["acquired"]


def test_zero_velocity_counts_as_a_move_and_is_not_silently_dropped():
    """A null displacement must still consume budget -- otherwise a decoder that outputs
    nothing would give the model unlimited attempts."""
    cfg = BoardConfig(move_budget=6)
    s = mk_session(0, [Block("baseline", 1)], vel=(0.0, 0.0), cfg=cfg)
    b = MockBackend(["r r r r"])
    run_lockstep(b, [s], layer=0, verbose=False, stuck_limit=99)
    assert sum(t["n_moves"] for t in s.turns) == cfg.move_budget


# ------------------------------------------------------------------- robustness
def test_stuck_detector_aborts_a_session():
    """DESIGN 8.5 greedy fixed point: if the marker stops moving the prompt stops changing, so
    a greedy reply stops changing and the session would spin forever."""
    b = MockBackend(["x x x"])                          # always invalid
    s = mk_session(0, [Block("baseline", 50)])
    run_lockstep(b, [s], layer=0, verbose=False, stuck_limit=3)
    assert s.turns[-1].get("stuck_abort") is True
    assert len(s.turns) == 3


def test_repetition_detector_aborts_identical_replies():
    """The failure mode that actually fired in testing (§8.5). Identical replies still MOVE the
    marker, so the stuck counter never trips -- it just moves the same way regardless of the
    target, marching in a straight line. Measured on a 0.6B: 100% duplicate replies, 0/3 trials
    acquired, the full move budget burned every trial, and no abort."""
    b = MockBackend(["r r r r"])                     # the same reply every turn
    s = mk_session(0, [Block("baseline", 50)], vel=(0.2, 0.0))
    run_lockstep(b, [s], layer=0, verbose=False, stuck_limit=99, repeat_limit=5)
    assert s.turns[-1].get("repeat_abort") is True
    assert s.turns[-1].get("stuck_abort") is False, "this is repetition, not immobility"
    # repeat_limit counts identical replies AFTER the first, so the abort lands on turn 6
    assert len(s.turns) == 6


def test_varying_replies_do_not_trip_the_repetition_detector():
    b = MockBackend(["r r r r", "r r u r", "u r r r", "r u r u"])
    s = mk_session(0, [Block("baseline", 1)], vel=(1.0, 0.0))
    run_lockstep(b, [s], layer=0, verbose=False, stuck_limit=99, repeat_limit=3)
    assert not any(t.get("repeat_abort") for t in s.turns)


def test_conversation_alternates_strictly():
    b = MockBackend(["r r r r"])
    s = mk_session(0, [Block("baseline", 2)])
    run_lockstep(b, [s], layer=0, verbose=False, stuck_limit=99)
    roles = [m["role"] for m in s.conv]
    assert roles == ["user", "assistant"] * (len(roles) // 2)


def test_no_mid_session_message_is_ever_emitted_at_a_block_boundary():
    """DESIGN 3.3: the decoder is swapped silently. Nothing in the user text may mark it."""
    blocks = [Block("baseline", 1), Block("perturbation", 1, np.ones((2, HID)), 0.2)]
    s = mk_session(0, blocks)
    b = MockBackend(["r r r r r r r r r r r r"])
    run_lockstep(b, [s], layer=0, verbose=False)
    text = " ".join(m["content"] for m in s.conv if m["role"] == "user").lower()
    for word in ("decoder", "perturb", "changed", "mapping", "activation", "internal"):
        assert word not in text, f"leaked {word!r} into the transcript"


def test_logging_receives_one_record_per_turn():
    got = []
    b = MockBackend(["r r r r"])
    s = mk_session(0, [Block("baseline", 2)])
    run_lockstep(b, [s], layer=0, verbose=False, stuck_limit=99,
                 log=lambda idx, rec: got.append((idx, rec["turn"])))
    assert [t for _, t in got] == list(range(1, len(s.turns) + 1))
    assert all(i == 0 for i, _ in got)


def test_sessions_with_different_lengths_finish_independently():
    s0 = mk_session(0, [Block("baseline", 1)])
    s1 = mk_session(1, [Block("baseline", 3)])
    b = MockBackend(["r r r r r r r r r r r r", "r r r r r r r r r r r", "r r r r r r r r r r"])
    run_lockstep(b, [s0, s1], layer=0, verbose=False)
    assert len(s0.done_trials) == 1 and len(s1.done_trials) == 3


def test_per_move_path_is_logged_and_consistent():
    """The trajectory record must let a reader reconstruct every step, not just the endpoints.

    pos_before and pos_after only bracket a whole reply, so without this a per-step trajectory
    cannot be drawn at all. Asserts the path brackets match the recorded endpoints and that its
    length matches the number of applied moves, which is what makes it safe to concatenate turns
    into one trial trajectory.
    """
    b = MockBackend(["r r u u"])
    s = mk_session(0, [Block("baseline", 1)])
    run_lockstep(b, [s], layer=0, verbose=False)
    rec = s.turns[0]
    path = rec["path"]
    assert len(path) == rec["n_moves"] + 1, (len(path), rec["n_moves"])
    assert path[0] == rec["pos_before"], (path[0], rec["pos_before"])
    assert path[-1] == rec["pos_after"], (path[-1], rec["pos_after"])


# ---------------------------------------------------------------- checkpoint / resume
def _fresh_session(idx=0, trials=(3, 4, 2)):
    """A session with the three-block shape the campaign uses, at toy sizes.

    Uses the same FixedReadout as the other tests here: what is under test is the checkpoint
    round-trip, not the perturbation algebra, so the decoder is held constant on purpose.
    """
    blocks = [Block("baseline", trials[0]),
              Block("perturbation", trials[1], np.ones((2, HID)), 1.0),
              Block("washout", trials[2])]
    return mk_session(idx, blocks, seed=7000)


REPLIES = ["r r u", "u u r r", "r u", "d d l", "r r r u u", "l u r"]


def test_resume_reproduces_an_uninterrupted_run_exactly():
    """The property the whole checkpoint scheme rests on.

    A session killed mid-run and resumed from its checkpoint must produce the SAME trials as one
    that was never interrupted. If it does not, resumed sessions carry silently different data from
    fresh ones and every aggregate mixes two populations.
    """
    from nc.session import restore_session, session_state

    ref = _fresh_session()
    run_lockstep(MockBackend(REPLIES), [ref], layer=0, verbose=False)

    # now the same session, but stopped at the first checkpoint and resumed into a fresh object
    saved = {}
    cut = _fresh_session()

    def grab(s):
        if "d" not in saved:                       # keep only the FIRST checkpoint, then stop
            saved["d"] = session_state(s, "sha")
            raise KeyboardInterrupt                # crude stand-in for the 3 h wall

    with pytest.raises(KeyboardInterrupt):
        run_lockstep(MockBackend(REPLIES), [cut], layer=0, verbose=False, checkpoint=grab)

    resumed = _fresh_session()
    assert restore_session(resumed, saved["d"], "sha")
    assert resumed.trial_idx >= 1 and len(resumed.done_trials) >= 1
    run_lockstep(MockBackend(REPLIES), [resumed], layer=0, verbose=False)

    assert len(resumed.done_trials) == len(ref.done_trials)
    assert resumed.done_trials == ref.done_trials


def test_checkpoint_is_refused_when_the_configuration_differs():
    from nc.session import restore_session, session_state
    s = _fresh_session()
    run_lockstep(MockBackend(REPLIES), [s], layer=0, verbose=False)
    d = session_state(s, "sha-A")
    other = _fresh_session()
    assert restore_session(other, d, "sha-B") is False        # different cell
    assert restore_session(other, {**d, "schema": 999}, "sha-A") is False
    assert restore_session(_fresh_session(idx=7), d, "sha-A") is False   # different session index


def test_on_finish_fires_per_session_not_per_batch():
    """A session that finishes early must be durable before its batch-mates are done."""
    short, long_ = _fresh_session(0, trials=(1, 1, 1)), _fresh_session(1, trials=(4, 5, 3))
    seen = []
    run_lockstep(MockBackend(REPLIES), [short, long_], layer=0, verbose=False,
                 on_finish=lambda s: seen.append((s.idx, len(s.done_trials))))
    assert [i for i, _ in seen] == [0, 1]              # the short one lands first
    assert seen[0][1] == 3 and seen[1][1] == 12
