"""Centre-out navigation on a continuous plane with a discrete move alphabet.

The subject emits variable-length sequences of direction symbols (see docs/DESIGN.md §2).
Under the intuitive decoder each symbol is executed as a unit step; under a perturbed
decoder the symbol is ignored and the displacement comes from the residual read at the
preceding token. This module is agnostic to which: it consumes displacements.

The position shown to the model is rounded to integers -- the model's world is a grid --
while the internal state stays continuous, so the per-move angular-error metric keeps full
resolution instead of being quantised to the lattice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Direction alphabets. The axis symbols carry their natural meaning; the diagonal symbols
# (q/e/z/c, the keyboard corner layout) carry only weak semantics, which costs nothing --
# the decoder is fit to the move actually emitted, not to the symbol's meaning (§3.4).
_D = math.sqrt(0.5)
ALPHABETS = {
    "rlud": {"r": (1.0, 0.0), "l": (-1.0, 0.0), "u": (0.0, 1.0), "d": (0.0, -1.0)},
    "rludqezc": {
        "r": (1.0, 0.0), "l": (-1.0, 0.0), "u": (0.0, 1.0), "d": (0.0, -1.0),
        "q": (-_D, _D), "e": (_D, _D), "z": (-_D, -_D), "c": (_D, -_D),
    },
}

INVALID_EMPTY = "empty"
INVALID_SYMBOL = "bad_symbol"
INVALID_SHORT = "too_short"
INVALID_LONG = "too_long"


@dataclass
class BoardConfig:
    """Pilot geometry. Every number here is PROVISIONAL and frozen after gate G3, which
    measures the actual moves-to-acquire distribution under the intuitive decoder.

    With these values a shortest solution runs 7 moves (axis targets) to 10 (diagonal),
    median ~8.5, so ``move_budget`` is ~4x the median. The paper's ratio was ~7.5x
    (7500 ms limit against ~1000 ms typical); 4x is the declared compromise, chosen because
    a tighter ratio collapses success rate instantly under any perturbation -- which would
    destroy the initial-impairment match that G4 gates on -- while 7.5x spends context on
    long post-perturbation wandering.
    """
    alphabet: str = "rlud"
    radius: float = 8.0          # target distance from the origin
    target_radius: float = 1.5   # acquisition radius
    step: float = 1.0            # displacement per symbol under literal execution
    move_budget: int = 36        # moves per trial: the 7500 ms timeout analog, ~4x median
    n_targets: int = 8
    k_min: int = 3               # forbids degenerate 1-symbol replies, and guarantees
                                 # within-reply context variation across the read positions
    k_max: int = 12              # per-reply cap: context budget only

    @property
    def symbols(self) -> dict:
        return ALPHABETS[self.alphabet]

    def sample_target(self, rng) -> tuple[float, float]:
        """A target at a UNIFORMLY RANDOM ANGLE and the fixed radius (DESIGN 3.1).

        Random angle, because with a fixed set of 8 targets and ~32 perturbation trials the
        model sees each about four times and could recover by memorising eight lookup entries
        instead of learning the mapping -- the tabulation failure mode re-entering through the
        target space, indistinguishable from real learning in the performance curves.

        Fixed radius, because distance sets difficulty and difficulty variation feeds straight
        into moves-to-acquire and success rate, the two quantities the learning metric is built
        from. Sadtler held amplitude and target size constant for this reason (Fitts' law).
        """
        a = rng.uniform(0.0, 2.0 * math.pi)
        return (self.radius * math.cos(a), self.radius * math.sin(a))

    def bin_centres(self) -> list[tuple[float, float]]:
        """The ``n_targets`` direction-BIN CENTRES -- analysis bins, not targets (DESIGN 3.1).

        Targets are sampled continuously; these 45 deg sectors are what the cosine-tuning fit
        and the per-direction screens aggregate over.
        """
        return [
            (math.cos(2 * math.pi * i / self.n_targets),
             math.sin(2 * math.pi * i / self.n_targets))
            for i in range(self.n_targets)
        ]

    def direction_bin(self, vec: tuple[float, float]) -> int:
        """Which 45 deg sector a direction falls in, 0..n_targets-1."""
        return int(round(math.atan2(vec[1], vec[0]) / (2 * math.pi / self.n_targets))) \
            % self.n_targets


@dataclass
class Trial:
    """One centre-out trial. ``apply`` consumes a displacement, whatever produced it."""
    target: tuple[float, float]
    cfg: BoardConfig
    pos: tuple[float, float] = (0.0, 0.0)
    moves_used: int = 0
    acquired: bool = False
    angular_errors: list[float] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.acquired or self.moves_used >= self.cfg.move_budget

    @property
    def distance(self) -> float:
        return math.dist(self.pos, self.target)

    def ideal(self) -> tuple[float, float]:
        """Unit vector from the current position toward the target ((0,0) if on it)."""
        dx, dy = self.target[0] - self.pos[0], self.target[1] - self.pos[1]
        n = math.hypot(dx, dy)
        return (0.0, 0.0) if n < 1e-12 else (dx / n, dy / n)

    def literal(self, symbol: str) -> tuple[float, float]:
        """The displacement a symbol produces under literal (phase-0) execution."""
        ux, uy = self.cfg.symbols[symbol]
        return (self.cfg.step * ux, self.cfg.step * uy)

    def apply(self, dx: float, dy: float) -> None:
        """Apply one displacement: record its angular error, move, check acquisition.

        Angular error is measured against the ideal direction *before* the move, and is
        NaN when either vector is degenerate (zero displacement, or already on target).
        """
        if self.done:
            raise RuntimeError("trial already finished")
        ix, iy = self.ideal()
        n = math.hypot(dx, dy)
        if n < 1e-12 or (ix == 0.0 and iy == 0.0):
            self.angular_errors.append(float("nan"))
        else:
            cos = max(-1.0, min(1.0, (dx * ix + dy * iy) / n))
            self.angular_errors.append(math.acos(cos))
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        self.moves_used += 1
        if self.distance <= self.cfg.target_radius:
            self.acquired = True


# ------------------------------------------------------------------ reply parsing
def parse_reply(text: str, cfg: BoardConfig) -> tuple[list[str] | None, str | None]:
    """``(moves, None)`` if valid, else ``(None, reason)``.

    Validity mirrors the in-channel convention used across the program: an invalid reply
    is logged as the model's actual action, scores nothing, and is reported back to the
    model in the next user turn rather than being silently repaired.
    """
    toks = (text or "").strip().split()
    if not toks:
        return None, INVALID_EMPTY
    if any(t not in cfg.symbols for t in toks):
        return None, INVALID_SYMBOL
    if len(toks) < cfg.k_min:
        return None, INVALID_SHORT
    if len(toks) > cfg.k_max:
        return None, INVALID_LONG
    return toks, None


# ------------------------------------------------------------------ prompt rendering
def render_system(cfg: BoardConfig) -> str:
    lines = ",  ".join(f"{s} = {_name(v)}" for s, v in cfg.symbols.items())
    example = " ".join(list(cfg.symbols)[:1] * 2 + list(cfg.symbols)[1:2] + list(cfg.symbols)[:1])
    return (
        "You are moving a marker on a grid. Each turn you will be told where the marker is\n"
        "and where the target is. Reply with a sequence of moves and nothing else.\n\n"
        f"The moves are:  {lines}\n\n"
        f"Write them separated by single spaces, for example:  {example}\n"
        f"Write at least {cfg.k_min} and at most {cfg.k_max} moves in one reply.\n\n"
        "Your goal is to land the marker on the target using as few moves as possible.\n"
        "When the marker reaches the target, a new trial starts with the marker back at (0, 0)."
    )


def _name(v: tuple[float, float]) -> str:
    ns = {(1, 0): "right", (-1, 0): "left", (0, 1): "up", (0, -1): "down"}
    key = (round(v[0]), round(v[1]))
    if key in ns:
        return ns[key]
    return ("up" if v[1] > 0 else "down") + "-" + ("right" if v[0] > 0 else "left")


def render_turn(trial: Trial, trial_idx: int) -> str:
    return (
        f"Trial {trial_idx}. Marker {_show(trial.pos)}. Target {_show(trial.target)}. "
        f"Moves left in this trial: {trial.cfg.move_budget - trial.moves_used}."
    )


def render_user_turn(trial: Trial, trial_idx: int, *, first: bool = False,
                     preamble: str | None = None) -> str:
    """The ONE user message for a turn -- the single template shared by calibration and play.

    Chat turns must alternate strictly, so everything the model sees before its reply is
    concatenated into one user message: on the first turn the system instructions, and
    thereafter any ``preamble`` (a previous trial's outcome line, or an invalid-move notice).
    Calibration and play must both build their prompts through this function; that is what
    makes the two byte-identical (DESIGN 4.5), and a template divergence here would be
    indistinguishable from a decoder failure.
    """
    parts = []
    if first:
        parts.append(render_system(trial.cfg))
    if preamble:
        parts.append(preamble)
    parts.append(render_turn(trial, trial_idx))
    return "\n\n".join(parts)


def render_trial_end(trial: Trial, trial_idx: int) -> str:
    if trial.acquired:
        return f"Trial {trial_idx}: TARGET REACHED after {trial.moves_used} moves."
    return f"Trial {trial_idx}: FAILED (out of moves)."


def shortest_path(trial: Trial, rng, max_len: int | None = None) -> list[str]:
    """A shortest lattice path from the marker to the target, in randomised symbol order.

    Calibration replies are target-*reaching* paths, matching the paper's observation
    calibration where the cursor travels straight to the target (§3.5). Randomising the
    order means a given target direction is served by many different symbol sequences
    (``r r u u``, ``u r u r``, ``r u r u``), so order carries no information about the
    target and there is variety within each direction.

    Only the axis-aligned components are used, so this works for any alphabet containing
    ``r l u d``. Truncated to ``max_len`` (default ``cfg.k_max``) -- a long path simply
    takes more than one turn, exactly as in play.
    """
    cfg = trial.cfg
    dx, dy = trial.target[0] - trial.pos[0], trial.target[1] - trial.pos[1]
    nx, ny = round(dx / cfg.step), round(dy / cfg.step)
    moves = ["r" if nx > 0 else "l"] * abs(nx) + ["u" if ny > 0 else "d"] * abs(ny)
    rng.shuffle(moves)
    return moves[: (cfg.k_max if max_len is None else max_len)]


def render_invalid(reason: str, cfg: BoardConfig) -> str:
    detail = {
        INVALID_EMPTY: "Your reply was empty.",
        INVALID_SYMBOL: "Your reply contained something that is not a move.",
        INVALID_SHORT: f"Your reply had fewer than {cfg.k_min} moves.",
        INVALID_LONG: f"Your reply had more than {cfg.k_max} moves.",
    }[reason]
    return (
        f"{detail} Invalid move -- nothing happened this turn. Reply with between "
        f"{cfg.k_min} and {cfg.k_max} moves from {' '.join(cfg.symbols)}, separated by spaces."
    )


def _show(p: tuple[float, float]) -> str:
    """Integer-rounded display: the model's world is a grid (§2.1)."""
    return f"({round(p[0])}, {round(p[1])})"
