#!/usr/bin/env python
"""Figures for docs/DESIGN.md §7.3 and §7.4.

HONESTY RULES ENCODED HERE, because earlier versions of this file broke all three:
  * Every reference line is MEASURED, never chosen. The prompt-only baseline comes from
    `prompt_only_baselines` on the same label and the same split; R2 = 0 is the constant
    mean-direction predictor; 0.707 is the commitment of an even two-way tie; the majority-class
    rate is counted. No hand-picked cut-offs appear anywhere, and none should be added.
  * No coined metric names. Axis labels say what was computed ("held-out R2, direction of the
    model's next move"), not an invented capability word. An earlier version called a variance
    ratio "controllability" and that reading did not survive scrutiny.
  * TITLES DESCRIBE THE DISPLAY, NOT THE CONCLUSION. "R2 by population size", not "100
    coordinates is past the knee". The reader draws the inference; the figure states what is
    plotted. Interpretation belongs in DESIGN.md, where it can be qualified.
  * A superseded result is not shown as current. The 2026-08-10 closed-loop sessions used the
    forced-letter decoder, which we now know did not beat the prompt, so they render only under
    --superseded and are labelled as such.

Style: serif, title only, one black y spine with the others hidden, legend outside below.
Palette: Okabe-Ito subset validated against the dataviz six checks on this surface.

    python scripts/figures.py              # current results
    python scripts/figures.py --superseded # plus the 2026-08-10 forced-letter session figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
TAG = "Qwen3.5-9B"

BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, INK2, MUTED = "#1a1a1a", "#4a4a4a", "#8a8a8a"
SURF = "#ffffff"
AXIS = "#000000"
GRID = dict(color="#e2e2e0", lw=0.8, zorder=0)

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": AXIS,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
    "font.size": 10, "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "figure.dpi": 130,
})


def legend_below(ax, ncol=3, y=-0.19):
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, y),
              ncol=ncol, borderaxespad=0, handletextpad=0.6, columnspacing=1.8)


def style(ax, ylab=None, xlab=None, title=None):
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(AXIS)
    ax.spines["left"].set_linewidth(1.7)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", length=3.5, width=1.0, color=AXIS)
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    if ylab: ax.set_ylabel(ylab)
    if xlab: ax.set_xlabel(xlab)
    if title: ax.set_title(title, loc="left", pad=10, weight="bold")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


# ------------------------------------------------------------------ data loading
def load_report(path):
    p = ROOT / path
    return json.load(open(p)) if p.exists() else None


def by_layer(rep, key, agg=np.median):
    """(depths, values) per layer, aggregated over the coordinate draws."""
    rows = [x for x in rep["rows"] if key in x and np.isfinite(x[key])]
    if not rows:
        return [], []
    nl = rep.get("num_layers") or 32
    Ls = sorted({x["layer"] for x in rows})
    return ([L / (nl - 1) for L in Ls],
            [float(agg([x[key] for x in rows if x["layer"] == L])) for L in Ls])


def spread(rep, key, lo=10, hi=90):
    rows = [x for x in rep["rows"] if key in x and np.isfinite(x[key])]
    nl = rep.get("num_layers") or 32
    Ls = sorted({x["layer"] for x in rows})
    return ([np.percentile([x[key] for x in rows if x["layer"] == L], lo) for L in Ls],
            [np.percentile([x[key] for x in rows if x["layer"] == L], hi) for L in Ls])


# --------------------------------------- fig 1: does the readout beat the prompt?
def fig_readout(v2):
    """Two views of the same readout: variance explained, and exact-match rate."""
    if not v2:
        print("  1_readout.png skipped (no artifacts/v4 draws report yet)")
        return
    b = v2["prompt_only_baselines"]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))

    ax = axes[0]
    xs, ys = by_layer(v2, "r2")
    lo, hi = spread(v2, "r2")
    ax.fill_between(xs, lo, hi, color=BLUE, alpha=0.15, lw=0, zorder=2,
                    label="range over 20 coordinate draws")
    ax.plot(xs, ys, "-o", color=BLUE, lw=2, ms=8, mec=SURF, mew=1.4, zorder=3,
            label="readout from the activations")
    ax.axhline(b["best_prompt_only"], color=INK, ls="--", lw=1.5, zorder=1)
    ax.text(0.125, b["best_prompt_only"] + 0.02,
            f"a predictor that sees only the prompt ({b['best_prompt_only']:.2f})",
            fontsize=8.5, color=INK, ha="left", va="bottom")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.text(0.905, 0.02, "always guessing the average direction",
            fontsize=8, color=MUTED, ha="right", va="bottom")
    style(ax, ylab="held-out R², direction of the next move", xlab="layer depth",
          title="Variance explained, by depth")
    handles, labels = ax.get_legend_handles_labels()

    ax = axes[1]
    xs, ys = by_layer(v2, "match_acc")
    if xs:
        mlo, mhi = spread(v2, "match_acc")
        ax.fill_between(xs, mlo, mhi, color=BLUE, alpha=0.15, lw=0, zorder=2)
        ax.plot(xs, ys, "-o", color=BLUE, lw=2, ms=8, mec=SURF, mew=1.4, zorder=3)
    if np.isfinite(b.get("match_geometry_ridge", np.nan)):
        v = b["match_geometry_ridge"]
        ax.axhline(v, color=INK, ls="--", lw=1.4, zorder=1)
        ax.text(0.125, v + 0.014, f"a predictor that sees only the prompt ({v:.2f})",
                fontsize=8.5, color=INK, ha="left", va="bottom")
    if np.isfinite(b.get("match_majority_class", np.nan)):
        v = b["match_majority_class"]
        ax.axhline(v, color=MUTED, lw=0.8, zorder=1)
        ax.text(0.905, v + 0.014, f"always guessing the commonest letter ({v:.2f})",
                fontsize=8, color=MUTED, ha="right", va="bottom")
    style(ax, ylab="share of positions named correctly", xlab="layer depth",
          title="Exact match with the model's next letter, by depth")
    ax.set_ylim(0, 1.02)
    fig.legend(handles, labels, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 0.02), ncol=2, columnspacing=2.4)
    save(fig, "1_readout.png")


# ------------------------------------------- fig 3: what the intent label contains
def fig_label():
    """Why the intent label carries anything: the model is not sitting on the fence at ties.

    Earlier this plotted |E[direction]|, the length of the probability-weighted direction. That
    quantity is right for the label (it IS the label before normalising) but it does not explain
    itself: a reader cannot tell why 0.8 means "committed". The probability the model puts on its
    single preferred move says the same thing and needs no interpretation.
    """
    p = ROOT / f"artifacts/v4/reads_{TAG}.npz"
    if not p.exists():
        print("  2_label.png skipped (no v4 reads artifact yet)")
        return
    z = np.load(p, allow_pickle=True)
    if "fit_letter_p" not in z.files:
        print("  2_label.png skipped (reads artifact has no letter distributions)")
        return
    P = z["fit_letter_p"].astype(np.float64)
    amb = z["fit_ambiguous"].astype(bool)
    top = P.max(1)
    ok = np.isfinite(top)

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    bins = np.linspace(0.25, 1, 46)
    ax.hist(top[ok & amb], bins=bins, color=VERM, alpha=0.8, lw=0, zorder=3,
            label=f"two moves are equally good ({(amb & ok).mean():.0%} of positions)")
    ax.hist(top[ok & ~amb], bins=bins, color=BLUE, alpha=0.8, lw=0, zorder=3,
            label="one move is strictly best")
    ax.axvline(0.5, color=INK, ls="--", lw=1.5, zorder=4)
    top_amb = float(np.median(top[ok & amb]))
    ax.axvline(top_amb, color=VERM, lw=1.6, zorder=4)
    ymax = ax.get_ylim()[1]
    ax.text(0.505, ymax * 0.93, "0.5 = torn between two moves,\nwhere the target direction\n"
                                "would predict the label perfectly",
            fontsize=8.5, color=INK, va="top")
    ax.text(top_amb - 0.012, ymax * 0.52,
            f"median where the two moves\nare equally good: {top_amb:.2f}",
            fontsize=8.5, color=VERM, va="center", ha="right")
    style(ax, ylab="read positions (one per letter the model wrote)",
          xlab="probability the model puts on its preferred move",
          title="Distribution of the probability on the preferred move")
    legend_below(ax, ncol=1, y=-0.20)
    save(fig, "2_label.png")


# --------------------------------------------------- closed-loop behaviour figures
def arm_style(alignment):
    """Fixed colour and marker per arm. Never keyed on sort order (see the docstring rules)."""
    if alignment is None:
        return GREEN, "o", "no perturbation"
    ladder = {1.0: (BLUE, "s"), 0.7: (INK2, "D"), 0.4: (VERM, "^"), 0.2: (MUTED, "v")}
    col, mk = ladder.get(round(float(alignment), 2), (INK2, "P"))
    return col, mk, f"alignment {float(alignment):.1f}"
def load_sessions(base):
    """[(arm, trials, summary_row)] across shards. Arm comes from meta.json, rounded.

    The v4 layout is shard*/ with all arms mixed inside each shard, so a figure cannot key off the
    directory name the way the old g4/ + sham/ split allowed.
    """
    import glob as _glob
    out = []
    dirs = sorted(_glob.glob(str(ROOT / base / "shard*"))) or [str(ROOT / base)]
    for d in dirs:
        mp = Path(d) / "meta.json"
        if not mp.exists():
            continue
        meta = {m["session"]: m for m in json.load(open(mp))}
        summ = {}
        sp = Path(d) / "summary.json"
        if sp.exists():
            # analyze.py keys rows by the session STEM ("sess_000"), not an int session field
            for r in json.load(open(sp)):
                sid = r.get("session")
                if sid is None and isinstance(r.get("idx"), str):
                    sid = int(r["idx"].split("_")[-1])
                summ.setdefault(sid, []).append(r)
        for f in sorted(_glob.glob(f"{d}/sess_*_trials.json")):
            i = int(Path(f).stem.split("_")[1])
            m = meta.get(i)
            if m is None:
                continue
            al = (m.get("perturbation") or {}).get("alignment")
            out.append({"arm": None if al is None else round(float(al), 1),
                        "draw": m.get("draw"), "session": i,
                        "trials": json.load(open(f)), "summary": summ.get(i, [])})
    return out


def by_block(sessions, arm, idx):
    """(mean success, median moves) per block for one arm, one value per session."""
    rows = {}
    for s in sessions:
        if s["arm"] != arm:
            continue
        for blk in ("baseline", "perturbation", "washout"):
            g = [t for t in s["trials"] if t["block"] == blk]
            if g:
                rows.setdefault(blk, []).append(
                    float(np.mean([t["acquired"] for t in g])) if idx == 0
                    else float(np.median([t["moves_used"] for t in g])))
    return rows


def fig_blocks(base, note):
    sessions = load_sessions(base)
    if not sessions:
        print("  4_blocks.png skipped (no sessions)")
        return
    arms = sorted({s["arm"] for s in sessions}, key=lambda x: (x is None, -(x or 0)))
    blocks = ["baseline", "perturbation", "washout"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
    w, xs = 0.8 / max(len(arms), 1), np.arange(3)
    for j, (idx, ylab, ttl) in enumerate([
            (0, "trials reaching the target", "Success rate by block"),
            (1, "median moves used", "Moves used by block")]):
        ax = axes[j]
        for k, arm in enumerate(arms):
            rows = by_block(sessions, arm, idx)
            if not rows:
                continue
            col, _, lab = arm_style(arm)
            v = [np.mean(rows[b]) if b in rows else np.nan for b in blocks]
            e = [np.std(rows[b]) / max(np.sqrt(len(rows[b])), 1) if b in rows else 0
                 for b in blocks]
            off = (k - (len(arms) - 1) / 2) * w
            ax.bar(xs + off, v, w * 0.88, yerr=e, color=col, zorder=3,
                   label=lab if j == 0 else None,
                   error_kw=dict(ecolor=MUTED, lw=1, capsize=2.5))
            for xi, vi in zip(xs + off, v):
                if np.isfinite(vi):
                    ax.text(xi, vi + (0.03 if idx == 0 else 1.2),
                            f"{vi:.2f}" if idx == 0 else f"{vi:.0f}",
                            ha="center", fontsize=7.5, color=INK2)
        ax.set_xticks(xs)
        ax.set_xticklabels([b.capitalize() for b in blocks])
        style(ax, ylab=ylab, title=ttl)
        if idx == 1:
            ax.axhline(36, color=INK, ls="--", lw=1.4, zorder=4)
            ax.text(-0.45, 37.2, "move budget, a trial hitting this failed", ha="left",
                    fontsize=8, color=INK)
            ax.set_ylim(0, 45)
        else:
            ax.set_ylim(0, 1.16)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 0.015), ncol=4, columnspacing=2.0)
    save(fig, "4_blocks.png")


def fig_metric(base, note):
    sessions = load_sessions(base)
    pts = {}
    for s in sessions:
        for r in s["summary"]:
            if r.get("metric") != "paper":
                continue
            pts.setdefault(s["arm"], []).append((r["initial_impairment"], r["l_session"]))
    if not pts:
        print("  7_metric.png skipped (no summary.json; run scripts/analyze.py --mode blocks)")
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for arm in sorted(pts, key=lambda x: (x is None, -(x or 0))):
        col, mk, lab = arm_style(arm)
        v = np.array(pts[arm])
        ax.scatter(v[:, 0], v[:, 1], s=70, color=col, marker=mk, edgecolor=SURF, lw=1.2,
                   label=f"{lab}  (n={len(v)})", zorder=3)
    ax.axhline(1.0, color=MUTED, ls=":", lw=1)
    ax.text(ax.get_xlim()[1], 1.04, "full recovery = 1.0", fontsize=8, color=MUTED, ha="right")
    ax.axhline(0, color=MUTED, lw=0.8)
    style(ax, ylab="measured amount of learning (L_session)",
          xlab="initial impairment, the metric's denominator",
          title="L_session against initial impairment, one point per session")
    legend_below(ax, ncol=4, y=-0.19)
    save(fig, "7_metric.png")


# ------------------- fig 3: are the frozen design constants (100 coordinates, d = 10) big enough?
def fig_constants():
    A = load_report(f"artifacts/v4/axes_{TAG}.json")
    if not A or not {"a4", "b5"} <= set(A):
        print("  3_constants.png skipped (no v4 axes sweep yet)")
        return
    a4 = [r for r in A["a4"] if "r2" in r]
    shown = sorted({r["layer"] for r in a4})
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3))

    ax = axes[0]
    for L in shown:
        g = [r for r in a4 if r["layer"] == L]
        ns = sorted({r["n_coords"] for r in g})
        m = [float(np.median([r["r2"] for r in g if r["n_coords"] == n])) for n in ns]
        col = BLUE if L >= 25 else (GREEN if L >= 21 else VERM)
        ax.plot(ns, m, "-o", color=col, lw=2, ms=6, mec=SURF, mew=1.2, label=f"layer {L}", zorder=3)
    ax.axvline(100, color=INK, ls="--", lw=1.3, zorder=1)
    ax.text(108, ax.get_ylim()[0] + 0.05, "100, the frozen choice", fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100, 200, 400, 800])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    style(ax, ylab="held-out R², intent direction", xlab="coordinates in the population",
          title="R² by population size")
    legend_below(ax, ncol=3, y=-0.20)

    ax = axes[1]
    b5 = [r for r in A["b5"] if "r2" in r and r["layer"] in shown]
    for L in sorted({r["layer"] for r in b5}):
        g = [r for r in b5 if r["layer"] == L]
        ds = sorted({r["d"] for r in g})
        m = [float(np.mean([r["r2"] for r in g if r["d"] == d])) for d in ds]
        col = BLUE if L >= 25 else (GREEN if L >= 21 else VERM)
        ax.plot(ds, m, "-o", color=col, lw=2, ms=6, mec=SURF, mew=1.2, zorder=3)
    ax.axvline(10, color=INK, ls="--", lw=1.3, zorder=1)
    ax.text(11, ax.get_ylim()[0] + 0.04, "d = 10, the frozen choice", fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    style(ax, ylab="held-out R², intent direction", xlab="factors kept, d",
          title="R² by factors kept")

    ax = axes[2]
    if "a1" in A:
        per = A["a1"]["per_layer"]
        Ls = sorted(int(k) for k in per)
        xs = [L / (A["num_layers"] - 1) for L in Ls]
        ax.vlines(xs, [per[str(L)]["min"] for L in Ls], [per[str(L)]["max"] for L in Ls],
                  color=BLUE, lw=6, alpha=0.25, zorder=2)
        ax.plot(xs, [per[str(L)]["median"] for L in Ls], "-o", color=BLUE, lw=2, ms=7,
                mec=SURF, mew=1.4, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.text(0.02, 0.03, f"depth accounts for {A['a1']['var_share_layer']*100:.0f}% of the "
                            f"R² variance,\nthe coordinate draw for "
                            f"{A['a1']['var_share_draw']*100:.0f}%",
                transform=ax.transAxes, fontsize=8.5, color=INK, va="bottom")
        style(ax, ylab="held-out R², range over 20 draws", xlab="layer depth",
              title="R² range across 20 coordinate draws")
    save(fig, "3_constants.png")


# ------------------------- fig 5: moves per trial, one panel per coordinate draw
def fig_v3():
    """Moves per trial across the session, per draw and arm. The plot the run is about."""
    import glob as _glob
    dirs = sorted(_glob.glob(str(ROOT / "outputs/v4/shard*")))
    if not dirs:
        print("  5_moves.png skipped (no outputs/v4)")
        return
    rows = []
    for d in dirs:
        mp = Path(d) / "meta.json"
        if not mp.exists():
            continue
        meta = {m["session"]: m for m in json.load(open(mp))}
        for f in sorted(_glob.glob(f"{d}/sess_*_trials.json")):
            i = int(Path(f).stem.split("_")[1])
            m = meta.get(i)
            if not m:
                continue
            q = m.get("perturbation") or {}
            al = q.get("alignment")
            rows.append({"draw": m["draw"], "arm": None if al is None else round(float(al), 1),
                         "moves": [t["moves_used"] for t in json.load(open(f))]})
    draws = sorted({r["draw"] for r in rows})
    fig, axes = plt.subplots(1, len(draws), figsize=(6.2 * len(draws), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, dr in zip(axes, draws):
        for arm in (None, 1.0, 0.7, 0.4):
            g = [r["moves"] for r in rows if r["draw"] == dr and r["arm"] == arm]
            if not g:
                continue
            n = max(len(x) for x in g)
            A = np.full((len(g), n), np.nan)
            for k, x in enumerate(g):
                A[k, :len(x)] = x
            with np.errstate(invalid="ignore"):
                mu = np.nanmean(A, axis=0)
            col, _, lab = arm_style(arm)
            ax.plot(np.arange(len(mu)) + 1, mu, "-", color=col, lw=1.9,
                    label=f"{lab}  (n={len(g)})", zorder=3)
        ax.axhline(36, color=INK, ls="--", lw=1.3, zorder=1)
        ax.text(1.5, 37, "move budget, a trial hitting this failed", fontsize=8, color=INK)
        for b in (8.5, 48.5):
            ax.axvline(b, color=INK, lw=1.0, zorder=1)
        ax.text(9.5, 6.5, "readout rotated", fontsize=8.5, color=INK)
        ax.text(49.5, 6.5, "restored", fontsize=8.5, color=INK)
        style(ax, ylab="moves to reach the target" if dr == draws[0] else None,
              xlab="trial number", title=f"Coordinate population {dr}")
        ax.set_ylim(5, 41)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 0.03), ncol=4, columnspacing=2.2)
    save(fig, "5_moves.png")


# ------------- fig 8: aim, separated from speed. Same layout as fig 7 so they can be compared.
def fig_v3_aim():
    """Angle between the decoded step and the straight line to the target, per trial.

    Fig 7 shows moves used, which mixes two things: how well the marker is aimed and how far it
    travels per move. The perturbation halves the step length, so most of fig 7's rise can be a
    speed effect. This panel isolates aim: it is the angular error recorded at every applied move
    (nc/task.py Trial.apply), averaged per trial, and it is blind to step size by construction.
    """
    import glob as _glob
    dirs = sorted(_glob.glob(str(ROOT / "outputs/v4/shard*")))
    if not dirs:
        print("  6_aim.png skipped (no outputs/v4)")
        return
    rows = []
    for d in dirs:
        mp = Path(d) / "meta.json"
        if not mp.exists():
            continue
        meta = {m["session"]: m for m in json.load(open(mp))}
        for f in sorted(_glob.glob(f"{d}/sess_*_trials.json")):
            i = int(Path(f).stem.split("_")[1])
            m = meta.get(i)
            if not m:
                continue
            q = m.get("perturbation") or {}
            al = q.get("alignment")
            per = []
            for tr in json.load(open(f)):
                e = [x for x in (tr.get("angular_errors") or []) if x is not None]
                per.append(float(np.degrees(np.mean(e))) if e else np.nan)
            rows.append({"draw": m["draw"], "arm": None if al is None else round(float(al), 1),
                         "ang": per})
    draws = sorted({r["draw"] for r in rows})
    fig, axes = plt.subplots(1, len(draws), figsize=(6.2 * len(draws), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, dr in zip(axes, draws):
        for arm in (None, 1.0, 0.7, 0.4):
            g = [r["ang"] for r in rows if r["draw"] == dr and r["arm"] == arm]
            if not g:
                continue
            n = max(len(x) for x in g)
            A = np.full((len(g), n), np.nan)
            for k, x in enumerate(g):
                A[k, :len(x)] = x
            with np.errstate(invalid="ignore"):
                mu = np.nanmean(A, axis=0)
            col, _, lab = arm_style(arm)
            ax.plot(np.arange(len(mu)) + 1, mu, "-", color=col, lw=1.9,
                    label=f"{lab}  (n={len(g)})", zorder=3)
        ax.axhline(90, color=INK, ls="--", lw=1.3, zorder=1)
        ax.text(1.5, 92, "90 degrees, the step carries no aim at all", fontsize=8, color=INK)
        for b in (8.5, 48.5):
            ax.axvline(b, color=INK, lw=1.0, zorder=1)
        ax.text(9.5, 14, "readout rotated", fontsize=8.5, color=INK)
        ax.text(49.5, 14, "restored", fontsize=8.5, color=INK)
        style(ax, ylab="angle to the target, degrees" if dr == draws[0] else None,
              xlab="trial number", title=f"Coordinate population {dr}")
        ax.set_ylim(10, 108)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 0.03), ncol=4, columnspacing=2.2)
    save(fig, "6_aim.png")


# ---- fig 8: effective rotation, the rotation the model actually experiences
def fig_effective_rotation():
    """Angle between each decoded step and the canonical direction of the letter that produced it.

    This is the perturbation as the MODEL experiences it. The nominal theta is a rotation applied
    to the 2D control space inside the 10-dimensional manifold; what reaches the model is the
    change in the map from its letter to the marker's motion, and the two are not the same number.
    Measured on this run: alignment 1.0 and 0.7 share a nominal 50 deg yet land at 71 and 41 deg
    effective, and alignment 0.4's nominal 60 deg lands at 42.

    It is also by far the best predictor of impairment we have (rank correlation -0.80 with success
    across 68 sessions, against -0.27 for the severity that the screen actually matches on).

    Needs `path` and `reply` from the turn records: the per-move displacement is a path diff and
    the letter that caused it is reply.split()[j], since move j applies the state that generated
    letter j.
    """
    import glob as _glob
    DIRS = {"r": (1.0, 0.0), "l": (-1.0, 0.0), "u": (0.0, 1.0), "d": (0.0, -1.0)}
    dirs = sorted(_glob.glob(str(ROOT / "outputs/v4/shard*")))
    if not dirs:
        print("  8_effective_rotation.png skipped (no outputs/v4)")
        return
    rows = []
    for d in dirs:
        mp = Path(d) / "meta.json"
        if not mp.exists():
            continue
        meta = {m["session"]: m for m in json.load(open(mp))}
        for f in sorted(_glob.glob(f"{d}/sess_*.jsonl")):
            i = int(Path(f).stem.split("_")[1])
            m = meta.get(i)
            if not m:
                continue
            q = m.get("perturbation") or {}
            al = q.get("alignment")
            per = {}
            for line in Path(f).open():
                try:
                    r = json.loads(line)
                except Exception:                                    # noqa: BLE001
                    continue
                pth, rep = r.get("path"), (r.get("reply") or "").split()
                if not pth or len(pth) < 2:
                    continue
                P = np.asarray(pth, dtype=float)
                for j in range(len(P) - 1):
                    if j >= len(rep) or rep[j] not in DIRS:
                        continue
                    v = P[j + 1] - P[j]
                    n = float(np.linalg.norm(v))
                    if n < 1e-9:
                        continue
                    c = float(np.dot(v / n, DIRS[rep[j]]))
                    per.setdefault(r["trial"], []).append(
                        float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
            if per:
                rows.append({"draw": m.get("draw"),
                             "arm": None if al is None else round(float(al), 1),
                             "theta": q.get("theta_deg"),
                             "series": [float(np.mean(per[k])) if k in per else np.nan
                                        for k in range(1, max(per) + 1)]})
    if not rows:
        print("  8_effective_rotation.png skipped (no path/reply data)")
        return
    draws = sorted({r["draw"] for r in rows})
    fig, axes = plt.subplots(1, len(draws), figsize=(6.2 * len(draws), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, dr in zip(axes, draws):
        for arm in (None, 1.0, 0.7, 0.4):
            g = [r["series"] for r in rows if r["draw"] == dr and r["arm"] == arm]
            if not g:
                continue
            n = max(len(x) for x in g)
            A = np.full((len(g), n), np.nan)
            for k, x in enumerate(g):
                A[k, :len(x)] = x
            with np.errstate(invalid="ignore"):
                mu = np.nanmean(A, axis=0)
            col, _, lab = arm_style(arm)
            th = next((r["theta"] for r in rows
                       if r["draw"] == dr and r["arm"] == arm and r["theta"]), None)
            ax.plot(np.arange(len(mu)) + 1, mu, "-", color=col, lw=1.9, zorder=3,
                    label=lab + (f", nominal {th:.0f} deg" if th else ""))
        ax.axhline(90, color=INK, ls="--", lw=1.2, zorder=1)
        ax.text(1.5, 91.5, "90 degrees, the step is orthogonal to its own letter",
                fontsize=8, color=INK)
        for b in (8.5, 48.5):
            ax.axvline(b, color=INK, lw=1.0, zorder=1)
        ax.text(9.5, 6, "readout rotated", fontsize=8.5, color=INK)
        ax.text(49.5, 6, "restored", fontsize=8.5, color=INK)
        style(ax, ylab="angle to the letter's own direction, degrees" if dr == draws[0] else None,
              xlab="trial number", title=f"Coordinate population {dr}")
        ax.set_ylim(0, 100)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 0.03), ncol=4, columnspacing=1.8)
    save(fig, "8_effective_rotation.png")


# ---- fig 9: the rotation x alignment grid, and the hypothesis test it enables
def fig_grid():
    """Success against effective rotation, and what alignment adds once that is removed.

    Alignment is ORDINAL (1.0, 0.7, 0.4, 0.2, 0.1), so it gets a single-hue light-to-dark ramp
    with a colourbar rather than five categorical colours: cycling hues would imply the levels are
    unordered kinds. Reference lines are measured, not chosen: the control arm's own success, and
    zero residual.
    """
    A = load_report("artifacts/v5_grid.json")
    if not A or not A.get("sessions"):
        print("  9_grid.png skipped (run scripts/analyze_grid.py first)")
        return
    S = [r for r in A["sessions"]
         if r.get("alignment") is not None and r.get("n_trials", 0) >= 40
         and np.isfinite(r.get("effective", np.nan))]
    ctrl = [r for r in A["sessions"] if r.get("alignment") is None]
    if len(S) < 8:
        print(f"  9_grid.png skipped (only {len(S)} usable perturbed sessions)")
        return
    x = np.array([r["effective"] for r in S], float)
    y = np.array([r["success"] for r in S], float)
    al = np.array([r["alignment"] for r in S], float)

    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))

    ax = axes[0]
    if ctrl:
        c = float(np.mean([r["success"] for r in ctrl]))
        ax.axhline(c, color=MUTED, lw=0.9)
        ax.text(x.max(), c + 0.015, f"no perturbation ({c:.2f})", fontsize=8.5,
                color=MUTED, ha="right", va="bottom")
    sc = ax.scatter(x, y, c=al, cmap=cmap, norm=norm, s=70, edgecolor=SURF, lw=1.0, zorder=3)
    # cubic in effective rotation only, the same fit the hypothesis test residualises against
    o = np.argsort(x)
    M = np.column_stack([np.ones_like(x), x, x ** 2, x ** 3])
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    ax.plot(x[o], (M @ coef)[o], "-", color=INK, lw=1.8, zorder=4,
            label="cubic fit on effective rotation alone")
    style(ax, ylab="trials reaching the target", xlab="effective rotation, degrees",
          title="Success against effective rotation")
    ax.set_ylim(-0.03, 1.06)
    legend_below(ax, ncol=1, y=-0.20)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("alignment", fontsize=9)

    ax = axes[1]
    resid = y - M @ coef
    ax.axhline(0, color=INK, lw=1.4, ls="--", zorder=1)
    ax.text(0.03, 0.02, "zero = fully explained by effective rotation", fontsize=8.5,
            color=INK, va="bottom")
    for a_ in sorted(set(al.tolist())):
        m = al == a_
        ax.scatter(np.full(m.sum(), a_), resid[m], s=70, color=cmap(norm(a_)),
                   edgecolor=SURF, lw=1.0, zorder=3)
        ax.plot([a_ - 0.035, a_ + 0.035], [resid[m].mean()] * 2, color=INK, lw=2.2, zorder=4)
    style(ax, ylab="success minus the fit", xlab="alignment",
          title="Residual success by alignment, effective rotation removed")
    save(fig, "9_grid.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/v4")
    args = ap.parse_args()

    print("writing figures to", FIG)
    v4 = load_report(f"artifacts/v4/draws_{TAG}.json")
    fig_readout(v4)
    fig_label()
    fig_constants()
    fig_v3()
    fig_v3_aim()
    fig_effective_rotation()
    fig_grid()
    fig_blocks(args.base, "")
    fig_metric(args.base, "")
