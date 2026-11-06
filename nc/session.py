"""Closed-loop session runner (docs/DESIGN.md §3, §4.2, §4.7).

Runs B sessions in lockstep so vLLM sees real batches. Each session has its own coordinate
draw, its own perturbation, and its own target sequence -- a draw is the analog of one recording
day (§4.1), fixed for the whole context.

Per turn: batched greedy generation -> batched span read -> per-symbol decode -> apply -> log.
**No message is ever delivered mid-session**: the decoder is swapped silently at each block
boundary, exactly as the animals were never cued (§3.3).

DISPLACEMENT INDEXING. Calibration fits ``state at letter j -> direction of letter j+1``, so
``decode(state_j)`` is the intent for the move *after* letter j. A k-letter reply therefore
yields **k-1 displacements**, from states 1..k-1, matching the k-1 samples per reply that
calibration produced. We deliberately do NOT use state_0 (the last prompt token): it precedes
any emission, so its decoded velocity would be a pure function of the prompt and the model
would have no control over that step.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .task import BoardConfig, Trial, parse_reply, render_invalid, render_trial_end, render_user_turn


@dataclass
class Block:
    name: str
    trials: int
    M2: np.ndarray | None = None          # None = the intuitive decoder
    alignment: float | None = None


@dataclass
class SessionState:
    idx: int
    cfg: BoardConfig
    readout: object                        # nc.manifold.Readout, already coordinate-subset
    coords: np.ndarray
    rng: object
    conv: list = field(default_factory=list)
    trial: Trial | None = None
    trial_idx: int = 0
    first: bool = True
    preamble: str | None = None
    done_trials: list = field(default_factory=list)
    turns: list = field(default_factory=list)
    stuck: int = 0
    repeats: int = 0                             # consecutive identical replies
    last_reply: str | None = None
    blocks: list = field(default_factory=list)   # THIS session's schedule; see run_lockstep

    @property
    def total_trials(self) -> int:
        return sum(b.trials for b in self.blocks)

    def new_trial(self):
        self.trial_idx += 1
        self.trial = Trial(target=self.cfg.sample_target(self.rng), cfg=self.cfg)


SCHEMA = 1          # bump if the checkpoint contents change; old checkpoints are then ignored


def session_state(s: SessionState, config_sha: str) -> dict:
    """Everything about a running session that the run configuration does NOT reproduce.

    cfg, readout, coords and blocks are deliberately absent: run.py rebuilds them as a pure
    function of (draw, cell, seed), so storing them would only create a second source of truth that
    can disagree with the first. ``config_sha`` pins which configuration this state belongs to, so a
    checkpoint written under a different design is refused rather than silently resumed into.
    """
    st = s.rng.getstate()
    return {
        "schema": SCHEMA, "config_sha": config_sha, "idx": s.idx,
        "trial_idx": s.trial_idx, "first": s.first, "preamble": s.preamble,
        "done_trials": s.done_trials, "stuck": s.stuck, "repeats": s.repeats,
        "last_reply": s.last_reply, "conv": s.conv, "turn_no": getattr(s, "turn_no", 0),
        "trial": None if s.trial is None else {
            "target": list(s.trial.target), "pos": list(s.trial.pos),
            "moves_used": s.trial.moves_used, "acquired": s.trial.acquired,
            "angular_errors": list(s.trial.angular_errors)},
        "rng": [st[0], list(st[1]), st[2]],
    }


def restore_session(s: SessionState, d: dict, config_sha: str) -> bool:
    """Put a checkpoint back into a freshly built session. False if it must not be used."""
    if not d or d.get("schema") != SCHEMA or d.get("config_sha") != config_sha:
        return False
    if d.get("idx") != s.idx:
        return False
    s.trial_idx, s.first, s.preamble = d["trial_idx"], d["first"], d["preamble"]
    s.done_trials, s.stuck, s.repeats = list(d["done_trials"]), d["stuck"], d["repeats"]
    s.last_reply, s.conv = d["last_reply"], list(d["conv"])
    s.turn_no = d.get("turn_no", 0)
    t = d.get("trial")
    s.trial = None if t is None else Trial(
        target=tuple(t["target"]), cfg=s.cfg, pos=tuple(t["pos"]),
        moves_used=t["moves_used"], acquired=t["acquired"],
        angular_errors=list(t["angular_errors"]))
    v, keys, gauss = d["rng"]
    s.rng.setstate((v, tuple(int(x) for x in keys), gauss))
    return True


def _block_for(trial_no: int, blocks: list[Block]) -> Block:
    n = 0
    for b in blocks:
        n += b.trials
        if trial_no <= n:
            return b
    return blocks[-1]


def run_lockstep(backend, sessions: list[SessionState], layer: int,
                 max_new_tokens: int = 24, stuck_limit: int = 6, repeat_limit: int = 8,
                 max_turns: int | None = None, log=None, verbose=True,
                 checkpoint=None, on_finish=None):
    """Advance every session turn-by-turn until all have finished ITS OWN blocks.

    ``stuck_limit`` counts consecutive turns with NO displacement applied; ``repeat_limit``
    counts consecutive turns whose reply is byte-identical to the previous one (so the abort
    lands on the ``repeat_limit + 1``-th turn). Both are needed -- see the comment at the
    abort site.

    Each session carries its own readout (already restricted to that session's coordinate draw)
    AND its own ``blocks`` schedule, so sessions at different alignment levels can share one
    lockstep batch. Using a single shared schedule here would silently apply the first session's
    perturbation to every session in the batch.

    DURABILITY (2026-08-15). A gemma session runs 1-3 h, longer than is safe inside a 3 h wall, and
    until now nothing was written until the whole batch finished: fourteen sessions of 74-96 trials
    each were killed mid-batch and thrown away. Two callbacks fix that.

    ``checkpoint(s)`` is called after every completed trial with the session whose trial just ended.
    ``session_state`` below serialises everything that is not reproducible from the run
    configuration, so a killed job resumes at the last finished trial instead of at trial 0.
    ``on_finish(s)`` is called the moment a session finishes rather than when its batch does, so a
    session that completes at minute 40 is on disk even if its batch-mate runs into the wall.
    """
    for s in sessions:
        if not s.blocks:
            raise ValueError(f"session {s.idx} has no blocks")
    live = [s for s in sessions]
    for s in live:
        if s.trial is None:            # a restored session already holds its in-flight trial
            s.new_trial()
    t_start = time.time()
    turn_no = max((getattr(s, "turn_no", 0) for s in sessions), default=0)

    while live:
        turn_no += 1
        convs, prompts = [], []
        for s in live:
            txt = render_user_turn(s.trial, s.trial_idx, first=s.first, preamble=s.preamble)
            s.first, s.preamble = False, None
            convs.append(s.conv + [{"role": "user", "content": txt}])
            prompts.append(txt)
        replies = backend.generate_batch(convs, max_new_tokens=max_new_tokens,
                                         do_sample=False, enable_thinking=False)

        # parse; only valid replies need a read
        parsed, need = [], []
        for k, (s, rep) in enumerate(zip(live, replies)):
            moves, reason = parse_reply(rep, s.cfg)
            parsed.append((moves, reason, rep))
            if moves is not None and len(moves) >= 2:
                need.append(k)
        reads = {}
        if need:
            out = backend.read_residual_multi_batch(
                [(convs[k], " ".join(parsed[k][0])) for k in need], [layer],
                position=None, enable_thinking=False, extra_left=1)
            for k, r in zip(need, out):
                reads[k] = r[0].numpy().astype(np.float64)      # (span, hidden)

        finished = []
        for k, s in enumerate(live):
            moves, reason, rep = parsed[k]
            blk = _block_for(s.trial_idx, s.blocks)
            rec = {"turn": turn_no, "trial": s.trial_idx, "block": blk.name,
                   "alignment": blk.alignment, "reply": rep, "invalid": reason,
                   "target": list(s.trial.target), "pos_before": list(s.trial.pos),
                   "n_moves": 0, "angular_errors": [], "acquired": False}
            if moves is None or k not in reads:
                s.preamble = render_invalid(reason or "empty", s.cfg)
                s.stuck += 1
            else:
                U = reads[k][:, s.coords]
                V = s.readout.velocity(U, blk.M2)                 # (k+1, 2)
                applied = 0
                # Per-move positions, so a trajectory can be drawn one arrow per step. pos_before
                # and pos_after only bracket a whole reply, which is not enough to see WHERE the
                # marker went inside it. path[0] == pos_before by construction.
                path = [list(s.trial.pos)]
                # Row j is the state that GENERATED letter j (extra_left=1 supplies row 0, the
                # last prompt position), so a k-letter reply yields k moves. The old indexing
                # applied rows 0..k-2 as letters 1..k-1 and dropped the model's first move
                # entirely, executing a truncated version of every plan it wrote.
                for j in range(min(len(V) - 1, len(moves))):
                    if s.trial.done:
                        break
                    s.trial.apply(float(V[j, 0]), float(V[j, 1]))
                    applied += 1
                    path.append(list(s.trial.pos))
                rec["n_moves"] = applied
                rec["path"] = path
                rec["angular_errors"] = [None if math.isnan(e) else e
                                         for e in s.trial.angular_errors[-applied:]] if applied else []
                rec["speeds"] = [float(np.linalg.norm(V[j])) for j in range(applied)]
                s.stuck = s.stuck + 1 if applied == 0 else 0
            rec["pos_after"] = list(s.trial.pos)
            rec["acquired"] = bool(s.trial.acquired)
            # Decide the aborts BEFORE logging. An earlier version set these flags after the
            # log() call, so the reason a session died never reached the jsonl -- a control
            # session that hit the repetition detector at trial 19 looked simply truncated,
            # and was only noticed because it silently shortened an aggregate plot.
            s.repeats = s.repeats + 1 if rep.strip() == (s.last_reply or object()) else 0
            s.last_reply = rep.strip()
            hit_stuck, hit_rep = s.stuck >= stuck_limit, s.repeats >= repeat_limit
            if hit_stuck or hit_rep:
                rec["stuck_abort"], rec["repeat_abort"] = bool(hit_stuck), bool(hit_rep)
            s.turns.append(rec)
            if log is not None:
                log(s.idx, rec)

            s.conv = s.conv + [{"role": "user", "content": prompts[k]},
                               {"role": "assistant", "content": rep}]
            if s.trial.done:
                s.done_trials.append({"trial": s.trial_idx, "block": blk.name,
                                      "alignment": blk.alignment,
                                      "acquired": bool(s.trial.acquired),
                                      "moves_used": s.trial.moves_used,
                                      "angular_errors": [e for e in s.trial.angular_errors
                                                         if not math.isnan(e)]})
                s.preamble = render_trial_end(s.trial, s.trial_idx)
                if s.trial_idx >= s.total_trials:
                    finished.append(s)
                else:
                    s.new_trial()
                    s.turn_no = turn_no
                    if checkpoint is not None:
                        checkpoint(s)          # the ONLY safe resume point: between trials
            # Two distinct greedy failure modes (§8.5), and the second is the one that
             # actually fired in testing:
            #  * stuck   -- no displacement applied at all, so the marker is frozen;
            #  * repeats -- identical replies turn after turn. The marker still MOVES, so the
            #    stuck counter never trips, but it moves the same way regardless of the target,
            #    marching in a straight line forever. Measured on a 0.6B: 100% duplicate
            #    replies, 0/3 trials acquired, 36 moves burned per trial. This is the
            #    repetition degeneracy LITTLEBOY measured, and it needs its own detector.
            if hit_stuck or hit_rep:
                finished.append(s)
            elif max_turns is not None and turn_no >= max_turns:
                # CONTEXT GUARD, not a scientific criterion. The conversation grows by one
                # user turn plus one reply every turn, so an unbounded session eventually
                # exceeds max_model_len and vLLM raises mid-batch, losing the whole batch.
                # Cap turns below that limit and record the cap so a truncated session is
                # never mistaken for a completed one.
                rec["turn_cap"] = True
                finished.append(s)

        for s in finished:
            if s in live:
                live.remove(s)
                s.turn_no = turn_no
                if on_finish is not None:
                    on_finish(s)               # durable now, not when the batch ends
        if verbose:
            tt = max((x.total_trials for x in sessions), default=0)
            print(f"  turn {turn_no:3d}  live {len(live):2d}  "
                  f"trials {min((x.trial_idx for x in live), default=tt)}"
                  f"-{max((x.trial_idx for x in sessions), default=0)}/{tt}  "
                  f"{time.time()-t_start:.0f}s", flush=True)
    return sessions
