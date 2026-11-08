#!/usr/bin/env python
"""G3 / G4 / full: run closed-loop sessions (docs/DESIGN.md §7).

Each session gets its own 100-coordinate draw (§4.1), its own readout fit on the cached
calibration reads restricted to that draw, and — outside baseline-only mode — its own freshly
drawn perturbation at the screened (layer, theta) recipe, re-verified against the G2 criteria
before use.

    # G3: baseline only, freezes the geometry and the per-session cost
    python scripts/run.py --reads artifacts/reads_X.npz --screen artifacts/screen_X.json \
        --mode baseline --sessions 4 --trials 20

    # G4: matched impairment + nonzero learning at a = 1.0 vs the lowest level
    python scripts/run.py --reads ... --screen ... --mode blocks --sessions 8 \
        --alignments 1.0 0.1 --trials-baseline 8 --trials-perturb 40 --trials-washout 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nc import labels as lb                                              # noqa: E402
from nc import manifold as mf                                            # noqa: E402
from nc.session import (Block, SessionState, restore_session, run_lockstep,
                        session_state)                 # noqa: E402
from nc.task import ALPHABETS, BoardConfig                               # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import screen as scr                                                     # noqa: E402

SYM = ALPHABETS["rlud"]


def fit_for_draw(z, li, coords):
    """The readout the closed loop actually uses.

    LABEL FIXED 2026-08-18. This fit the forced-letter label until now, while the G1 gate that
    licenses the whole run certified an INTENT-fit decoder (DESIGN 7.2c, 2026-08-11). The label
    change had landed in nc/labels.py, fit_draws.py and the gate, but not here or in screen.py, so
    570 sessions ran a decoder that was never the one gated on and that scores BELOW the prompt-only
    baseline on its own label (-0.124 for the 9B, -0.218 for the 31B). Both places now call
    nc.labels.readout_label so the fitted object and the gated object cannot diverge again.
    """
    U = z["U_fit"][li][:, coords].astype(np.float64)
    G, usable = lb.readout_label(z, "fit")
    U, G = U[usable], G[usable]
    ro = mf.fit_readout(U, G, d=mf.MANIFOLD_DIM)
    mf.set_gain(ro, U, target_step=1.0)
    return ro, U, z["U_val"][li][:, coords].astype(np.float64)


def admissible_perturbation(ro, Mb, Uem, dirs, ref, a, theta, rng, tries=60,
                            target_dev_deg=None, gate_severity=True):
    """Draw a perturbation at (theta, a) that clears the G2 criteria; None if none found.

    SEVERITY TARGETING (added 2026-08-12, and the absence of it was a real defect). This used to
    return the FIRST candidate inside the 20-45 deg severity window, so the realised severity
    landed wherever the first passing sample fell. Measured on draw 5: the arms came out at 25.2,
    39.3 and 31.2 deg, a 14 deg spread, while G2 had matched that draw to 31.9/31.9/31.8. The
    whole point of the matched-difficulty control is that the arms differ in ALIGNMENT and not in
    difficulty, so an arm difference under the old behaviour could have been a severity difference.
    Every closed-loop run before this date is affected; the "severity matched 20/20 draws" line in
    the G2 report described the screen's search, never what the sessions ran.

    With ``target_dev_deg`` set, all admissible candidates are collected and the one closest to
    the target is returned, so arms within a draw are matched to a common severity.
    """
    best = None
    for _ in range(tries):
        M2P = mf.build_perturbation(ro.M2, Mb, theta, a, rng)
        sv = scr.severity(ro, Uem, M2P, ref)
        if sv is None:
            continue
        rho = mf.solution_density(Uem, ro, M2P, dirs, ref, max_angle=scr.RHO_TOL)
        gap = scr.spans_plane(ro, Uem, M2P, ref)
        sev_ok = (mf.SEVERITY_WINDOW[0] <= sv["dev"] <= mf.SEVERITY_WINDOW[1]
                  if gate_severity else True)
        # The reachability floors stay even when severity is not gated: they guarantee a solution
        # EXISTS, which is a different question from how hard that solution is to find.
        if (sev_ok and 0.3 <= sv["mag"] <= 1.5
                and rho.min() >= mf.RHO_MIN and gap < scr.MAX_GAP_DEG):
            info = {"dev_deg": math.degrees(sv["dev"]), "mag": sv["mag"],
                    "rho_min": float(rho.min()), "gap_deg": gap,
                    "alignment": mf.alignment(M2P, Mb),
                    "theta_deg": math.degrees(theta)}
            if target_dev_deg is None:
                return M2P, info
            miss = abs(info["dev_deg"] - target_dev_deg)
            if best is None or miss < best[0]:
                best = (miss, M2P, dict(info, dev_miss_deg=miss,
                                        dev_target_deg=float(target_dev_deg)))
    if best is not None:
        return best[1], best[2]
    return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reads", required=True)
    p.add_argument("--screen", default=None)
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--mode", choices=["baseline", "blocks"], default="baseline")
    p.add_argument("--sessions", type=int, default=4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--trials", type=int, default=20)               # baseline mode
    p.add_argument("--trials-baseline", type=int, default=8)
    p.add_argument("--trials-perturb", type=int, default=40)
    p.add_argument("--trials-washout", type=int, default=8)
    p.add_argument("--alignments", type=float, nargs="+", default=None,
                   help="default: the highest and lowest USABLE levels from the screen report")
    p.add_argument("--move-budget", type=int, default=None,
                   help="override BoardConfig.move_budget. Frozen at 36 by default; the grid "
                        "sweep raises it so a hard cell is not censored at the budget")
    p.add_argument("--grid-thetas", type=float, nargs="+", default=None,
                   help="GRID MODE: sweep rotation x alignment. Only the feasible pairs are run, "
                        "since alignment >= cos^2(theta) is one-sided and makes the grid "
                        "triangular. Severity is RECORDED, not gated: spanning the effective-"
                        "rotation range is the point, and the severity window truncates it")
    p.add_argument("--grid-alignments", type=float, nargs="+", default=None)
    p.add_argument("--grid-control", action="store_true",
                   help="add a no-perturbation cell, run on the same target seeds as the grid")
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--gpu-frac", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=32000)
    p.add_argument("--draws", type=int, nargs="+", default=None,
                   help="coordinate-draw indices to use. With --repeats this defines a crossed "
                        "design: every draw is run in every arm with every target seed")
    p.add_argument("--repeats", type=int, default=None,
                   help="target-sequence seeds per (draw, arm) cell. A repeat is the unit of "
                        "replication: generation is greedy, so the target order is the only "
                        "source of independent variation")
    p.add_argument("--control-arm", action="store_true",
                   help="append a no-perturbation arm to --alignments, run on the SAME target "
                        "seeds as the perturbed arms so the control is paired trial by trial")
    p.add_argument("--session-list", type=int, nargs="+", default=None,
                   help="explicit global session indices for this shard. Preferred over "
                        "--session-start/--sessions when resuming, since the missing set is not "
                        "contiguous once some sessions have completed or been skipped")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip sessions whose trials file already exists anywhere under --out. "
                        "Makes a job idempotent so a chain of them converges on a full sweep")
    p.add_argument("--deadline-min", type=float, default=None,
                   help="stop starting NEW batches after this many minutes. A batch that starts "
                        "is always finished and written, so a wall-clock kill never costs work")
    p.add_argument("--wall-limit-min", type=float, default=None,
                   help="adaptive alternative to --deadline-min for a model whose throughput has "
                        "not been measured: refuse to start a batch when 1.2x the SLOWEST observed "
                        "batch would run past this many minutes. The first batch always starts, so "
                        "the guard calibrates itself inside the slot instead of being guessed.")
    p.add_argument("--session-start", type=int, default=0,
                   help="first session index of this shard. The (draw, arm, repeat) mapping is a "
                        "pure function of the index, so shards on different GPUs stay paired")
    p.add_argument("--max-turns", type=int, default=None,
                   help="context guard: end a session after this many turns (see nc/session.py)")
    p.add_argument("--no-pair-populations", dest="pair_populations", action="store_false",
                   help="revert to the pre-2026-08-12 behaviour where the coordinate draw and the "
                        "alignment arm were both keyed on the session index, giving the arms "
                        "disjoint populations. Kept only to reproduce those runs")
    p.add_argument("--sham", action="store_true",
                   help="§6 no-perturbation control: identical 3-block schedule and labels, but\n                        the decoder is NEVER swapped. Gives the drift baseline the G4 criterion\n                        compares against -- what L_session/tail_mean look like when nothing changed.")
    p.add_argument("--tag", default="g3")
    p.add_argument("--out", default=str(ROOT / "outputs"))
    a = p.parse_args()

    z = np.load(a.reads, allow_pickle=True)
    layers = [int(x) for x in z["layers"]]
    cfg = BoardConfig(**({"move_budget": a.move_budget} if a.move_budget else {}))
    if a.move_budget:
        print(f"[run] move budget {cfg.move_budget} (default is "
              f"{BoardConfig().move_budget})")
    dirs = np.array(cfg.bin_centres(), dtype=np.float64)
    elig = mf.eligible_coords(z["U_fit"][-1][::7].astype(np.float64))

    layer = a.layer
    theta_for, usable, rep = {}, [], None
    if a.screen:
        rep = json.loads(Path(a.screen).read_text())
        layer = layer if layer is not None else int(rep["layer"])
        usable = [float(x) for x in rep.get("usable_levels", [])]
        for r in rep.get("rows", []):
            for lv, c in r.get("chosen", {}).items():
                theta_for.setdefault(float(lv), math.radians(c["theta_deg"]))
    if layer is None:
        raise SystemExit("need --layer or --screen")
    # Default the arms to the screen's USABLE levels. Hardcoding e.g. 0.1 when the screen
    # truncated the grid would make admissible_perturbation fail and silently skip every
    # session -- an empty run that looks like a completed one.
    alignments = a.alignments
    if alignments is None:
        if len(usable) >= 2:
            alignments = [max(usable), min(usable)]
        elif a.mode == "blocks":
            raise SystemExit("no usable alignment levels in the screen report; pass --alignments")
        else:
            alignments = [1.0]
    if a.mode == "blocks":
        unusable = [x for x in alignments if usable and x not in usable]
        if unusable:
            print(f"[run] WARNING: alignments {unusable} are not in the screen's usable set "
                  f"{usable}; sessions may be skipped")
        print(f"[run] alignments {alignments}")
    li = layers.index(layer)
    # Per-draw target severity: the median of what G2 matched for that draw, so the arms inside a
    # draw are held at one difficulty and only the alignment differs.
    target_dev = {}
    for row in (rep.get("rows", []) if rep else []):
        devs = [v["dev_deg"] for k, v in (row.get("chosen") or {}).items()
                if v and v.get("pass") and float(k) in usable]
        if devs:
            target_dev[int(row["draw"])] = float(np.median(devs))
    if target_dev:
        print(f"[run] severity targets per draw (from G2): "
              f"{ {k: round(v, 1) for k, v in sorted(target_dev.items())} }")
    print(f"[run] mode={a.mode} layer={layer} sessions={a.sessions} batch={a.batch}")

    from nc.backends.vllm_lens_backend import VLLMLensBackend, VLLMLensConfig
    backend = VLLMLensBackend(VLLMLensConfig(
        model_name=a.model, gpu_memory_utilization=a.gpu_frac,
        max_model_len=a.max_model_len, enable_prefix_caching=True))

    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    handles = {}

    def log(idx, rec):
        f = handles.get(idx)
        if f is None:
            # APPEND, not truncate: a resumed session has its pre-kill turns copied in below, and
            # analyze_grid reads effective rotation straight off this transcript. Opening "w" would
            # leave a resumed session with only its post-resume turns, so its effective rotation
            # would be computed on part of the session while its trials covered all of it.
            f = handles[idx] = open(out / f"sess_{idx:03d}.jsonl", "a")
        f.write(json.dumps(rec, default=float) + "\n")
        f.flush()

    # GRID MODE (2026-08-12): cells are (theta, alignment) pairs, cycling fastest, then the target
    # seed. A contiguous block of len(cells) sessions is therefore one seed across the WHOLE grid,
    # which is why the sweep is split across jobs BY SEED: a job that dies leaves a complete grid
    # at fewer seeds rather than a grid with holes.
    grid = None
    if a.grid_thetas and a.grid_alignments:
        grid = []
        for al in a.grid_alignments:
            adm = {round(math.degrees(x), 3) for x in mf.admissible_thetas(al)}
            for th in a.grid_thetas:
                if round(float(th), 3) in adm:
                    grid.append((float(al), float(th)))
        if a.grid_control:
            grid.append((None, None))
        print(f"[run] grid: {len(grid)} cells x {a.repeats} seeds = "
              f"{len(grid) * (a.repeats or 1)} sessions; this shard runs "
              f"{a.session_start}..{a.session_start + a.sessions - 1}")
        for al, th in grid:
            print(f"        cell alignment {'control' if al is None else format(al, '.2f')}"
                  f"  theta {'-' if th is None else format(th, '.0f') + ' deg'}")

    def grid_cell(i):
        """global session index -> (alignment, theta, seed). Cells cycle fastest."""
        seed, k = divmod(i, len(grid))
        al, th = grid[k]
        return al, th, seed

    # CROSSED DESIGN (2026-08-12). arms cycle fastest, then draws, then repeats, so any
    # contiguous block of len(draws) * len(arms) sessions is exactly one repeat containing every
    # condition on the identical target sequence.
    arms = list(alignments) + ([None] if a.control_arm else []) if a.mode != "baseline" else [None]
    crossed = a.draws is not None and a.repeats is not None
    if crossed:
        total = len(a.draws) * len(arms) * a.repeats
        print(f"[run] crossed design: draws {a.draws} x arms "
              f"{[('control' if x is None else x) for x in arms]} x {a.repeats} repeats "
              f"= {total} sessions; this shard runs "
              f"{a.session_start}..{a.session_start + a.sessions - 1}", flush=True)

    def cell(i):
        """global session index -> (draw index, arm, repeat)"""
        if not crossed:
            n_arms = len(alignments) if a.mode != "baseline" else 1
            return ((i // n_arms if a.pair_populations else i),
                    (alignments[i % len(alignments)] if a.mode != "baseline" else None), i)
        per_rep = len(a.draws) * len(arms)
        rep, rem = divmod(i, per_rep)
        di, ai = divmod(rem, len(arms))
        return a.draws[di], arms[ai], rep

    import time as _time
    t_start = _time.time()

    base = Path(a.out)          # the SWEEP dir; `out` below is this shard's subdir inside it

    def already_done(i):
        """Done if the session has trials, or was recorded as skipped, ANYWHERE in the sweep.

        Session indices are globally unique across the sweep, so which shard wrote a session does
        not matter; a resumed job must see work done by a different shard on a previous night.
        """
        for pat in (f"sess_{i:03d}_trials.json", f"sess_{i:03d}_skipped.json"):
            if (base / pat).exists() or any(base.glob(f"*/{pat}")):
                return True
        return False

    todo = (list(a.session_list) if a.session_list is not None
            else list(range(a.session_start, a.session_start + a.sessions)))
    if a.skip_existing:
        before = len(todo)
        todo = [i for i in todo if not already_done(i)]
        print(f"[run] resume: {before - len(todo)} of {before} sessions already done, "
              f"{len(todo)} to run", flush=True)
    if not todo:
        print("[run] nothing to do")
        return

    def _config_sha(draw_i, arm, g_th, rep, blocks, a):
        """Identity of a session's configuration. A checkpoint only resumes into the same one."""
        key = json.dumps({
            "draw": int(draw_i), "arm": None if arm is None else round(float(arm), 6),
            "theta": None if g_th is None else round(float(g_th), 6), "rep": int(rep),
            "layer": int(layer), "model": a.model, "budget": cfg.move_budget,
            "blocks": [[b.name, b.trials, None if b.alignment is None else round(float(b.alignment), 6)]
                       for b in blocks]}, sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _find_ckpt(i):
        """Checkpoints live in the shard dir that wrote them, and the next job runs under a NEW
        shard tag, so look across the whole sweep exactly as already_done() does."""
        local = out / f"sess_{i:03d}_ckpt.json"
        if local.exists():
            return local
        for c in sorted(base.glob(f"*/sess_{i:03d}_ckpt.json")):
            return c
        return None

    def _drop_ckpt(i):
        for c in list(base.glob(f"*/sess_{i:03d}_ckpt.json")) + [out / f"sess_{i:03d}_ckpt.json"]:
            c.unlink(missing_ok=True)

    ckpt_sha = {}
    meta_all = []
    batch_min = []          # observed durations, so the wall guard learns this model's batch cost
    for b0 in range(0, len(todo), a.batch):
        if a.deadline_min is not None and (_time.time() - t_start) / 60 > a.deadline_min:
            print(f"[run] deadline {a.deadline_min} min reached; "
                  f"{len(todo) - b0} sessions left for the next job", flush=True)
            break
        # WALL GUARD (2026-08-14). --deadline-min is a fixed cutoff, which needs the batch cost known
        # in advance: too low wastes most of the slot, too high loses the in-flight batch to the wall
        # (trials.json is written only after a batch completes). On a model whose throughput has never
        # been measured that trade is unguessable, so instead refuse to START a batch that the
        # SLOWEST observed batch says would not finish. The first batch always starts, which is what
        # produces the first measurement.
        if a.wall_limit_min is not None and batch_min:
            el = (_time.time() - t_start) / 60
            need = 1.2 * max(batch_min)
            if el + need > a.wall_limit_min:
                print(f"[run] wall guard: {el:.0f} min elapsed, slowest batch {max(batch_min):.0f} "
                      f"min, so a new batch would risk the {a.wall_limit_min} min limit; "
                      f"{len(todo) - b0} sessions left for the next job", flush=True)
                break
        t_batch = _time.time()
        group = todo[b0: b0 + a.batch]
        sessions = []
        for i in group:
            # POPULATION IS PAIRED ACROSS ARMS. Until 2026-08-12 the coordinate draw and the
            # alignment were both keyed on the session index, so `al = alignments[i % n]` gave the
            # two arms DISJOINT populations (measured: zero overlap across 8 sessions) and any arm
            # difference was confounded with which 100 coordinates it happened to get. Sadtler
            # compares within-manifold and outside-manifold perturbations within the same units,
            # so pair: draw index advances once per arm-cycle, and every draw is run in every arm.
            g_th = None
            if grid is not None:
                g_al, g_th, rep = grid_cell(i)
                draw_i, arm = (a.draws or [0])[0], g_al
            else:
                draw_i, arm, rep = cell(i)
            coords = mf.draw_coords(elig, mf.N_COORDS, np.random.default_rng(1000 + draw_i))
            ro, Ufit, Uem = fit_for_draw(z, li, coords)
            Mb = mf.manifold_basis(ro.beta)
            ref = mf.reference_speed(ro, Ufit)
            if a.mode == "baseline":
                blocks = [Block("baseline", a.trials)]
                pinfo = None
            else:
                al = arm if (crossed or grid is not None) else alignments[i % len(alignments)]
                if a.sham or ((crossed or grid is not None) and al is None):
                    # drift baseline: same schedule, same labels, decoder NEVER swapped
                    M2P, pinfo, al = None, {"sham": True}, None
                else:
                    if grid is not None:
                        th = math.radians(g_th)
                        # one perturbation per CELL, frozen across seeds, so seeds vary only the
                        # targets. Severity is not targeted here: spanning it is the point.
                        pert_seed = (5000 + 100 * draw_i + int(round(al * 100)) + int(g_th))
                        M2P, pinfo = admissible_perturbation(
                            ro, Mb, Uem, dirs, ref, al, th,
                            np.random.default_rng(pert_seed), gate_severity=False)
                    else:
                        th = theta_for.get(al) or min(mf.admissible_thetas(al))
                        pert_seed = (5000 + 100 * draw_i + int(round(al * 10))
                                     if crossed else 5000 + i)
                        M2P, pinfo = admissible_perturbation(
                            ro, Mb, Uem, dirs, ref, al, th, np.random.default_rng(pert_seed),
                            target_dev_deg=target_dev.get(draw_i))
                    if M2P is None:
                        print(f"  session {i}: NO admissible perturbation at a={al}; skipped")
                        # Mark it. Without this a resumable chain would retry the infeasible
                        # alignment-0.1 cells forever and never terminate.
                        (out / f"sess_{i:03d}_skipped.json").write_text(json.dumps(
                            {"session": i, "alignment": al, "theta_deg": g_th,
                             "reason": "no admissible perturbation"}))
                        continue
                blocks = [Block("baseline", a.trials_baseline),
                          Block("perturbation", a.trials_perturb, M2P, al),
                          Block("washout", a.trials_washout)]
            st = SessionState(idx=i, cfg=cfg, readout=ro, coords=coords,
                              rng=random.Random(7000 + rep), blocks=blocks)
            # RESUME (2026-08-15). A checkpoint is written between trials, so a session killed by
            # the 3 h wall restarts at its last finished trial instead of at trial 0. config_sha
            # pins the checkpoint to this exact cell: a stale one from a different design is
            # refused, not silently resumed into.
            sha = _config_sha(draw_i, arm, g_th, rep, blocks, a)
            ck = _find_ckpt(i)
            if ck is not None:
                try:
                    d = json.loads(ck.read_text())
                except Exception:                                   # noqa: BLE001
                    d = None
                if restore_session(st, d, sha):
                    # carry the transcript forward so the session ends up with ONE complete jsonl
                    prev = ck.parent / f"sess_{i:03d}.jsonl"
                    dst = out / f"sess_{i:03d}.jsonl"
                    if prev.exists() and prev.resolve() != dst.resolve():
                        dst.write_text(prev.read_text())
                    print(f"[run] session {i}: RESUMED at trial {st.trial_idx}/"
                          f"{st.total_trials} ({len(st.done_trials)} trials already done, "
                          f"transcript carried forward)", flush=True)
                else:
                    print(f"[run] session {i}: checkpoint present but not usable "
                          f"(different config or schema); starting from trial 0", flush=True)
            sessions.append(st)
            ckpt_sha[i] = sha
            meta_all.append({"session": i, "coords_sha": int(coords.sum()),
                             "draw": int(draw_i), "repeat": int(rep),
                             "cell_theta_deg": g_th, "perturbation": pinfo,
                             "blocks": [{"name": b.name, "trials": b.trials,
                                         "alignment": b.alignment} for b in blocks]})
        if not sessions:
            continue
        # Write the provenance BEFORE running: a crash mid-batch would otherwise lose the
        # record of which perturbation each session used, and the transcripts alone do not
        # carry theta, severity or reachability.
        (out / "meta.json").write_text(json.dumps(meta_all, indent=2, default=float))
        # sessions carry their own schedules, so mixed alignments can share one batch
        print(f"[run] batch {group[0]}-{group[-1]}: {len(sessions)} sessions, "
              f"alignments {[b.alignment for s in sessions for b in s.blocks if b.alignment]}")
        def _ckpt(s):
            """Atomic: write a temp file then rename, so a kill mid-write cannot leave a
            half-written checkpoint that the next job would fail to parse."""
            tmp = out / f"sess_{s.idx:03d}_ckpt.tmp"
            tmp.write_text(json.dumps(session_state(s, ckpt_sha[s.idx]), default=float))
            tmp.replace(out / f"sess_{s.idx:03d}_ckpt.json")

        def _finish(s):
            (out / f"sess_{s.idx:03d}_trials.json").write_text(
                json.dumps(s.done_trials, indent=2, default=float))
            _drop_ckpt(s.idx)                     # the session is complete; drop its checkpoint
            print(f"[run] session {s.idx} finished: {len(s.done_trials)} trials, written",
                  flush=True)

        run_lockstep(backend, sessions, layer, max_turns=a.max_turns, log=log,
                     checkpoint=_ckpt, on_finish=_finish)
        for s in sessions:                        # belt and braces for any session on_finish missed
            f = out / f"sess_{s.idx:03d}_trials.json"
            if not f.exists():
                f.write_text(json.dumps(s.done_trials, indent=2, default=float))
        batch_min.append((_time.time() - t_batch) / 60)
        print(f"[run] batch {group[0]}-{group[-1]} took {batch_min[-1]:.1f} min "
              f"({batch_min[-1] / max(1, len(sessions)):.1f} min/session at batch {a.batch}); "
              f"{(_time.time() - t_start) / 60:.0f} min into the slot", flush=True)

    (out / "meta.json").write_text(json.dumps(meta_all, indent=2, default=float))
    for f in handles.values():
        f.close()
    print(f"[run] -> {out}")


if __name__ == "__main__":
    main()
