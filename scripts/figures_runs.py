#!/usr/bin/env python
"""One folder per RUN (model x coordinate population), plus aggregate figures across runs.

WHY A TREE. The campaign produced four runs: Qwen3.5-9B on populations 16 and 14, gemma-4-31B-it on
populations 9 and 17. The old flat figures/ directory could only ever show one of them, and in fact
showed a fifth thing: it was still wired to outputs/v4, the superseded four-arm run. A reader could
not tell which run a figure came from. Here every figure lives under the run it describes, the folder
name carries model, layer and population, and PROVENANCE.txt records the sweep directory and counts.

    figures/runs/<model>_L<layer>_pop<draw>/
        01_dynamics.png      2x2: success, moves, aim, effective rotation, all against trial number
        02_readout.png       what the decoder could read at this layer, this population highlighted
        trajectories/        one page per sampled arm, drawn by plot_trajectories.py
        PROVENANCE.txt
    figures/aggregate/
        01_replication.png   the four runs side by side on the two numbers that matter
        02_dose_response.png success against effective rotation, every run overlaid

HONESTY RULES inherited from figures.py and re-stated because they are easy to lose in a rewrite:
titles describe the display and never the conclusion; every reference line is measured, not chosen;
no coined metric names. One addition specific to this tree: the grid is TRIANGULAR, so low alignment
exists only at high rotation. Curves grouped by alignment are therefore NOT difficulty-matched, and
PROVENANCE.txt says so for every run. The matched-band test in compare_draws.py is what handles that
confound; a time-course figure cannot.

    python scripts/figures_runs.py                 # everything
    python scripts/figures_runs.py --only dynamics # one figure kind
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

from figures import (AXIS, BLUE, GREEN, GRID, INK, INK2, MUTED, SURF,    # noqa: E402
                     VERM, legend_below, style)
import nc.manifold as mf                                                 # noqa: E402
from analyze_grid import load as load_grid                               # noqa: E402

FIG = ROOT / "figures"
DIRS = {"r": (1.0, 0.0), "l": (-1.0, 0.0), "u": (0.0, 1.0), "d": (0.0, -1.0)}
MIN_TRIALS = 40          # a session needs this many perturbation trials to be analysable
PHASE_TINT = {"baseline": (GREEN, 0.10), "perturbation": (VERM, 0.13),
              "washout": (BLUE, 0.10)}

# The DECODER LABEL is part of a run's identity and therefore part of its folder name. Everything
# before 2026-08-18 ran a readout fit to the forced next letter, which the 2026-08-11 label change
# had superseded and which scores BELOW the prompt-only baseline on its own label; the gate had
# certified an intent-fit readout that the loop never used (DESIGN 7.2c, 7.2i). Those runs measure a
# decoder the model could barely steer, so a reader must not have to dig to find out which is which.
RUNS = [
    dict(slug="qwen3.5-9b_L28_pop16_letterfit", dir="outputs/v5", model="Qwen3.5-9B",
         tag="Qwen3.5-9B", calib="artifacts/v4", layer=28, draw=16, decoder="forced letter",
         note="SUPERSEDED DECODER: the original grid result"),
    dict(slug="qwen3.5-9b_L28_pop14_letterfit", dir="outputs/v6q", model="Qwen3.5-9B",
         tag="Qwen3.5-9B", calib="artifacts/v4", layer=28, draw=14, decoder="forced letter",
         note="SUPERSEDED DECODER: second population"),
    dict(slug="gemma-4-31b_L53_pop09_letterfit", dir="outputs/v6g_d9", model="gemma-4-31B-it",
         tag="gemma-4-31B-it", calib="artifacts/v6g", layer=53, draw=9, decoder="forced letter",
         note="SUPERSEDED DECODER: second model"),
    dict(slug="gemma-4-31b_L53_pop17_letterfit", dir="outputs/v6g_d17", model="gemma-4-31B-it",
         tag="gemma-4-31B-it", calib="artifacts/v6g", layer=53, draw=17, decoder="forced letter",
         note="SUPERSEDED DECODER: second model, second population"),
    dict(slug="qwen3.5-9b_L28_pop13_intent", dir="outputs/v7q_d13", model="Qwen3.5-9B",
         tag="Qwen3.5-9B", calib="artifacts/v7q", layer=28, draw=13, decoder="model intent",
         note="corrected decoder, first population"),
    dict(slug="qwen3.5-9b_L28_pop00_intent", dir="outputs/v7q_d0", model="Qwen3.5-9B",
         tag="Qwen3.5-9B", calib="artifacts/v7q", layer=28, draw=0, decoder="model intent",
         note="corrected decoder, second population"),
]


def run_dir(r, *parts):
    p = FIG / "runs" / r["slug"]
    for x in parts:
        p = p / x
    p.parent.mkdir(parents=True, exist_ok=True) if parts else p.mkdir(parents=True, exist_ok=True)
    return p


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(FIG)}")


# Alignment is ORDERED, so it gets a sequential scale and a colourbar rather than five unrelated
# hues: the reader can then see 0.4 sitting between 0.2 and 0.7 without consulting a key. Same
# colormap and normalisation as fig_grid in figures.py, so alignment means the same colour project
# wide. Control is not a point on the scale (no perturbation was applied at all), so it keeps its
# own colour and its own legend entry.
# Viridis TRUNCATED at both ends. Full viridis puts alignment 1.0 on pale yellow, which is the
# least legible colour on this near-white surface and happens to be the most important arm, and
# alignment 0.1 on a purple that is hard to tell from the black spine. Trimming to [0.06, 0.86]
# keeps the ordered ramp and its perceptual uniformity while staying readable as thin lines.
ALIGN_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "alignment", plt.get_cmap("viridis")(np.linspace(0.06, 0.86, 256)))
ALIGN_NORM = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)


def arm_colour(al):
    # The control is NOT a point on the alignment scale, so it must not read as one: black keeps it
    # outside the ramp entirely. An earlier version used the palette green, which collided with
    # viridis's mid-green around alignment 0.7.
    if al is None:
        return INK, "no perturbation"
    return ALIGN_CMAP(ALIGN_NORM(float(al))), f"alignment {float(al):.1f}"


def align_colourbar(fig, handle, arms, **kw):
    """Colourbar ticked at the alignment values the grid actually contains, not a round scale."""
    ticks = sorted(a for a in arms if a is not None)
    cb = fig.colorbar(handle, ax=fig.axes, ticks=ticks, **kw)
    cb.set_ticklabels([f"{t:.1f}" for t in ticks])
    cb.set_label("alignment", fontsize=9.5)
    cb.ax.tick_params(labelsize=9)
    return cb


# ------------------------------------------------------------------ per-trial series
def per_trial_effective(jsonl: Path):
    """{trial: mean angle between each applied step and the letter that caused it}.

    Same pairing as analyze_grid: move j applies the state that generated letter j, so the
    displacement path[j+1]-path[j] belongs to reply.split()[j].
    """
    acc = {}
    if not jsonl.exists():
        return acc
    for line in jsonl.open():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:                                                # noqa: BLE001
            continue
        p, rep = r.get("path"), (r.get("reply") or "").split()
        if not p or len(p) < 2:
            continue
        P = np.asarray(p, dtype=float)
        for j in range(len(P) - 1):
            if j >= len(rep) or rep[j] not in DIRS:
                continue
            v = P[j + 1] - P[j]
            n = float(np.linalg.norm(v))
            if n < 1e-9:
                continue
            c = float(np.dot(v / n, DIRS[rep[j]]))
            acc.setdefault(r.get("trial"), []).append(
                math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def load_run(r):
    """[{alignment, theta, trials, eff_by_trial}] for every session in the sweep."""
    out = []
    for d in sorted(glob.glob(str(ROOT / r["dir"] / "*"))):
        mp = Path(d) / "meta.json"
        if not mp.exists():
            continue
        meta = {m["session"]: m for m in json.loads(mp.read_text())}
        for f in sorted(glob.glob(f"{d}/sess_*_trials.json")):
            i = int(Path(f).stem.split("_")[1])
            m = meta.get(i)
            if m is None:
                continue
            q = m.get("perturbation") or {}
            al = q.get("alignment")
            out.append({
                "session": i, "alignment": None if al is None else round(float(al), 2),
                "theta": m.get("cell_theta_deg"),
                "trials": json.loads(Path(f).read_text()),
                "eff": per_trial_effective(Path(d) / f"sess_{i:03d}.jsonl"),
            })
    return out


def trial_series(sessions, metric, al):
    """(trial indices, mean over sessions) for one alignment level."""
    per = {}
    for s in sessions:
        if s["alignment"] != al:
            continue
        for t in s["trials"]:
            k = t["trial"]
            if metric == "success":
                v = float(t["acquired"])
            elif metric == "moves":
                v = float(t["moves_used"])
            elif metric == "aim":
                # Trial.apply stores acos(cos), i.e. RADIANS. Convert here so this panel shares
                # units with the effective-rotation panel beside it.
                e = [x for x in (t.get("angular_errors") or []) if x is not None]
                v = float(np.degrees(np.mean(e))) if e else None
            else:
                v = s["eff"].get(k)
            if v is not None and np.isfinite(v):
                per.setdefault(k, []).append(v)
    ks = sorted(per)
    return np.array(ks), np.array([float(np.mean(per[k])) for k in ks])


def block_spans(sessions):
    """(block, first trial, last trial) from the recorded labels, not from assumed counts."""
    span = {}
    for s in sessions:
        for t in s["trials"]:
            b = t["block"]
            lo, hi = span.get(b, (10**9, -10**9))
            span[b] = (min(lo, t["trial"]), max(hi, t["trial"]))
    return [(b, *span[b]) for b in ("baseline", "perturbation", "washout") if b in span]


# ------------------------------------------------------------------ 1. the 2x2
PANELS = [("success", "trials reaching\nthe target", "Trials reaching the target"),
          ("moves", "moves to reach\nthe target", "Moves to reach the target"),
          ("aim", "angle to the target,\ndegrees", "Angle to the target"),
          ("eff", "angle to the letter's\nown direction, degrees",
           "Angle to the letter's own direction")]


MIN_SESSIONS_FOR_CURVES = 12   # below this a per-trial mean is one session, not an average


def fig_dynamics(r, sessions):
    if len(sessions) < MIN_SESSIONS_FOR_CURVES:
        print(f"  01_dynamics skipped for {r['slug']}: {len(sessions)} sessions, need "
              f"{MIN_SESSIONS_FOR_CURVES} before a per-trial mean means anything")
        return
    arms = sorted({s["alignment"] for s in sessions},
                  key=lambda a: (a is not None, -(a or 0)))
    spans = block_spans(sessions)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))
    for ax, (metric, ylab, ttl) in zip(axes.ravel(), PANELS):
        for b, lo, hi in spans:
            col, alpha = PHASE_TINT[b]
            ax.axvspan(lo - 0.5, hi + 0.5, color=col, alpha=alpha, lw=0, zorder=0)
        for al in arms:
            x, y = trial_series(sessions, metric, al)
            if x.size < 3:
                continue
            col, lab = arm_colour(al)
            # perturbed arms are identified by the colourbar, so only the control needs a label
            ax.plot(x, y, "-", color=col, lw=2.0 if al is None else 1.5,
                    zorder=4 if al is None else 3,
                    label=lab if al is None else None)
        style(ax, ylab=ylab, xlab="trial number", title=ttl)
        if metric == "success":
            ax.set_ylim(-0.03, 1.06)
    fig.suptitle(f"{r['model']}, layer {r['layer']}, population {r['draw']}, "
                 f"{r['decoder']} decoder", x=0.02, ha="left", weight="bold", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.04, 0.90, 0.95))
    sm = matplotlib.cm.ScalarMappable(norm=ALIGN_NORM, cmap=ALIGN_CMAP)
    align_colourbar(fig, sm, arms, location="right", fraction=0.030, pad=0.015, aspect=34)
    h, l = axes[0, 0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, frameon=False, loc="lower center", ncol=len(l),
                   bbox_to_anchor=(0.45, -0.01))
    save(fig, run_dir(r, "01_dynamics.png"))


# ------------------------------------------------------------------ 1b. grid coverage
def fig_cells(r):
    """Success per (alignment, rotation) cell, with the cells the design cannot build marked.

    The grid is TRIANGULAR because alignment >= cos^2(theta) is one-sided, and a cell can also be
    dropped because no candidate perturbation at that corner cleared the severity, reachability and
    extrapolation screens. Those two absences mean different things, so they are drawn differently
    rather than both left blank: hatched means the geometry forbids it, "skip" means the geometry
    allows it but nothing admissible was found. A reader can then see coverage and difficulty in one
    place, which no scatter of sessions shows.
    """
    rows = load_grid(r["dir"])
    if not rows:
        print(f"  03_cells skipped for {r['slug']} (no sessions yet)")
        return
    pert = [x for x in rows if x["alignment"] is not None]
    ctrl = [x for x in rows if x["alignment"] is None]
    if not pert:
        print(f"  03_cells skipped for {r['slug']} (no perturbed sessions)")
        return
    # ASCENDING, so alignment increases upward like any y axis. Descending put 1.0 at the
    # bottom and read as though more alignment were less.
    als = sorted({x["alignment"] for x in pert})
    ths = sorted({x["theta"] for x in pert})
    n_skip = {}
    for f in glob.glob(str(ROOT / r["dir"] / "*" / "sess_*_skipped.json")):
        d = json.loads(Path(f).read_text())
        a_, t_ = d.get("alignment"), d.get("theta_deg")
        if a_ is not None and t_ is not None:
            n_skip[(round(float(a_), 2), float(t_))] = n_skip.get((round(float(a_), 2), float(t_)), 0) + 1

    fig, (ax, cax) = plt.subplots(1, 2, figsize=(1.85 * len(ths) + 5.0, 1.30 * len(als) + 4.0),
                                  gridspec_kw={"width_ratios": [len(ths), 1]})
    cmap = plt.get_cmap("Blues")
    norm = matplotlib.colors.Normalize(0.0, 1.0)
    for i, al in enumerate(als):
        adm = {round(math.degrees(x), 3) for x in mf.admissible_thetas(al)}
        for j, th in enumerate(ths):
            g = [x for x in pert if x["alignment"] == al and x["theta"] == th]
            use = [x for x in g if x["n_trials"] >= MIN_TRIALS]
            feasible = round(float(th), 3) in adm
            if use:
                v = float(np.mean([x["success"] for x in use]))
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=cmap(norm(v)), lw=0.8,
                                           edgecolor=SURF))
                ax.text(j + 0.5, i + 0.60, f"{v:.2f}", ha="center", va="center", fontsize=19,
                        color=SURF if v > 0.55 else INK, weight="bold")
                ax.text(j + 0.5, i + 0.27, f"{len(use)}/{len(g)}", ha="center", va="center",
                        fontsize=12, color=SURF if v > 0.55 else INK2)
            elif not feasible:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor="#efefee", lw=0.8,
                                           edgecolor=SURF, hatch="///"))
            else:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor="#efefee", lw=0.8,
                                           edgecolor=SURF))
                lab = "skip" if n_skip.get((al, float(th))) else "no\ndata"
                ax.text(j + 0.5, i + 0.5, lab, ha="center", va="center", fontsize=13, color=MUTED)
    ax.set_xlim(0, len(ths)); ax.set_ylim(0, len(als))
    ax.set_xticks(np.arange(len(ths)) + 0.5)
    ax.set_xticklabels([f"{t:.0f}" for t in ths], fontsize=18)
    ax.set_yticks(np.arange(len(als)) + 0.5)
    ax.set_yticklabels([f"{a:.1f}" for a in als], fontsize=18)
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel(r"nominal rotation $\theta$, degrees", fontsize=21)
    ax.set_ylabel(r"alignment $a$", fontsize=21)
    ax.set_title("Trials reaching the target, per cell", loc="left", pad=14, weight="bold",
                 fontsize=20)

    if ctrl:
        v = float(np.mean([x["success"] for x in ctrl]))
        cax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=cmap(norm(v)), lw=0.8, edgecolor=SURF))
        cax.text(0.5, 0.60, f"{v:.2f}", ha="center", va="center", fontsize=19,
                 color=SURF if v > 0.55 else INK, weight="bold")
        cax.text(0.5, 0.27, f"n={len(ctrl)}", ha="center", va="center", fontsize=12,
                 color=SURF if v > 0.55 else INK2)
    cax.set_xlim(0, 1); cax.set_ylim(0, len(als))
    cax.set_xticks([0.5]); cax.set_xticklabels(["control"], fontsize=18); cax.set_yticks([])
    for sp in ("top", "right", "bottom", "left"):
        cax.spines[sp].set_visible(False)
    cax.tick_params(length=0)
    cax.set_title("no perturbation", loc="left", pad=14, fontsize=16)

    hatch = matplotlib.patches.Patch(
        facecolor="#efefee", hatch="///", edgecolor=SURF,
        label=r"cannot exist: $a < \cos^{2}\theta$, or too little in-manifold rotation")
    blank = matplotlib.patches.Patch(facecolor="#efefee", edgecolor=SURF,
                                     label="no admissible perturbation found")
    fig.legend(handles=[hatch, blank], frameon=False, loc="lower center", ncol=1,
               bbox_to_anchor=(0.5, -0.10), fontsize=17, handlelength=2.0,
               labelspacing=0.6)
    fig.suptitle(f"{r['model']}, layer {r['layer']}, population {r['draw']}, "
                 f"{r['decoder']} decoder", x=0.02, ha="left", weight="bold", fontsize=22)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    save(fig, run_dir(r, "03_cells.png"))


# ------------------------------------------------------------------ 2. the readout
def fig_readout(r):
    p = ROOT / r["calib"] / f"draws_{r['tag']}.json"
    if not p.exists():
        print(f"  02_readout skipped for {r['slug']} (no {p.name})")
        return
    rep = json.loads(p.read_text())
    rows = [x for x in rep["rows"] if "r2" in x]
    layers = sorted({x["layer"] for x in rows})
    base = rep["prompt_only_baselines"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))

    for ax, (key, ylab, ttl) in zip(axes, [
            ("r2", "held-out R2, direction of\nthe model's next move",
             "Variance explained, by depth"),
            ("match_acc", "exact match with\nthe model's next letter",
             "Exact match with the model's next letter, by depth")]):
        med = [float(np.median([x[key] for x in rows if x["layer"] == L])) for L in layers]
        lo = [float(np.min([x[key] for x in rows if x["layer"] == L])) for L in layers]
        hi = [float(np.max([x[key] for x in rows if x["layer"] == L])) for L in layers]
        ax.fill_between(layers, lo, hi, color=MUTED, alpha=0.22, lw=0, zorder=1,
                        label="range across populations")
        ax.plot(layers, med, "-o", color=INK, lw=1.7, ms=4.5, zorder=3,
                label="median across populations")
        mine = [(x["layer"], x[key]) for x in rows if x["draw"] == r["draw"]]
        if mine:
            xs, ys = zip(*sorted(mine))
            ax.plot(xs, ys, "-s", color=BLUE, lw=1.9, ms=5.5, zorder=4,
                    label=f"population {r['draw']}, this run")
        # Both panels get a prompt-only reference, because both are scores a prompt-only predictor
        # can also achieve. The match baselines were computed all along (nc.labels.prompt_only_
        # baselines writes match_* whenever the letter distributions are available) and simply were
        # not plotted. Labelled ON the line: a reference value is not a series and should not
        # compete with the three series in the legend. The trivial floors (R2 = 0, majority class)
        # are deliberately absent -- they carry no interpretation worth the ink.
        if key == "r2":
            b = base["best_prompt_only"]
        else:
            cands = {k: v for k, v in base.items()
                     if k.startswith("match_") and not k.endswith(("_ambiguous",
                                                                   "_where_prompt_disagrees"))
                     and k != "match_majority_class"}
            b = max(cands.values()) if cands else None
        if b is not None:
            ax.axhline(b, color=VERM, lw=1.5, ls="--", zorder=2)
            ax.text(layers[0], b, "best prompt-only baseline", fontsize=8.5, color=VERM,
                    va="bottom", ha="left")
        style(ax, ylab=ylab, xlab="layer", title=ttl)
    h, l = axes[0].get_legend_handles_labels()
    fig.suptitle(f"{r['model']}, layer {r['layer']}, population {r['draw']}, "
                 f"{r['decoder']} decoder", x=0.02, ha="left", weight="bold", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.legend(h, l, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02),
               columnspacing=2.4, handletextpad=0.6)
    save(fig, run_dir(r, "02_readout.png"))


# ------------------------------------------------------------------ 3. trajectories
def make_trajectories(r):
    out = run_dir(r, "trajectories")
    out.mkdir(parents=True, exist_ok=True)
    for arm in ("1.0", "0.4", "control"):
        cmd = [sys.executable, str(ROOT / "scripts/plot_trajectories.py"),
               "--dir", r["dir"], "--out", str(out.relative_to(ROOT)), "--arm", arm]
        s = subprocess.run(cmd, capture_output=True, text=True)
        if s.returncode != 0:
            print(f"  trajectories arm {arm}: failed ({s.stderr.strip().splitlines()[-1:]})")
    for f in sorted(out.glob("*.png")):
        print(f"  {f.relative_to(FIG)}")


# ------------------------------------------------------------------ provenance
def provenance(r, sessions):
    grid = load_grid(r["dir"])
    pert = [x for x in grid if x["alignment"] is not None]
    use = [x for x in pert if x["n_trials"] >= 40 and np.isfinite(x["effective"])]
    txt = f"""run: {r['slug']}
model: {r['model']}
decoder fit to: {r['decoder']}
layer: {r['layer']}
coordinate population (draw): {r['draw']}
role: {r['note']}
sweep directory: {r['dir']}
calibration artifacts: {r['calib']}

sessions written: {len(sessions)}
perturbed: {len(pert)}   control: {len(grid) - len(pert)}
usable for the grid analysis (>= 40 perturbation trials): {len(use)}
truncated below that: {len(pert) - len(use)}

CAVEAT ON ANY CURVE GROUPED BY ALIGNMENT
The grid is triangular for two reasons, both hatched in 03_cells.png. A cell needs alignment >=
cos^2(theta), so low alignment exists only at large rotation; and it needs cos^2(psi) >= 0.30^2, a
floor that stops the whole rotation tilting out of the manifold and leaving a merely SCALED copy of
the original mapping, which measures ~3 degrees of velocity change and is not a perturbation at all.
At alignment 0.1 and rotation 78 the first condition holds and the second does not. Curves grouped by alignment in 01_dynamics.png therefore pool different rotations
and are NOT difficulty-matched. Alignment effects are tested in the matched effective-rotation
bands reported by scripts/compare_draws.py, which is the only place that confound is handled.
"""
    p = run_dir(r, "PROVENANCE.txt")
    p.write_text(txt)
    print(f"  {p.relative_to(FIG)}")


# ------------------------------------------------------------------ aggregate
def load_replication():
    p = ROOT / "artifacts/v6_replication.json"
    if not p.exists():
        return None
    return [x for x in json.loads(p.read_text()) if x.get("summary")]


def fig_replication(rep):
    labels = [f"{x['model'].split('-')[0]}\n{x['draw']}" for x in rep]
    eff = [x["summary"]["rank_eff"] for x in rep]
    res = [x["summary"]["resid_align"] for x in rep]
    med = [x["summary"]["band_median"] for x in rep]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    xs = np.arange(len(rep))

    ax = axes[0]
    ax.bar(xs, eff, width=0.55, color=INK, zorder=3)
    ax.axhline(0, color=AXIS, lw=1.2, zorder=2)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    style(ax, ylab="rank correlation with success", title="Effective rotation against success")
    ax.set_ylim(-1.0, 0.1)

    ax = axes[1]
    ax.bar(xs - 0.17, res, width=0.32, color=BLUE, zorder=3,
           label="residual correlation, cubic removed")
    ax.bar(xs + 0.17, med, width=0.32, color=VERM, zorder=3,
           label="median across matched rotation bands")
    ax.axhline(0, color=AXIS, lw=1.2, zorder=2)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    style(ax, ylab="correlation of alignment with success", title="Alignment beyond effective rotation")
    legend_below(ax, ncol=1, y=-0.24)
    fig.tight_layout()
    save(fig, FIG / "aggregate" / "01_replication.png")


def fig_dose_response():
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    cols = [BLUE, INK2, VERM, GREEN]
    for r, col in zip(RUNS, cols):
        rows = [x for x in load_grid(r["dir"])
                if x["alignment"] is not None and x["n_trials"] >= 40
                and np.isfinite(x["effective"])]
        if len(rows) < 8:
            continue
        x = np.array([q["effective"] for q in rows])
        y = np.array([q["success"] for q in rows])
        ax.scatter(x, y, s=26, color=col, alpha=0.55, lw=0, zorder=3,
                   label=f"{r['model']}, population {r['draw']} (n={len(rows)})")
        M = np.column_stack([np.ones_like(x), x, x ** 2, x ** 3])
        coef, *_ = np.linalg.lstsq(M, y, rcond=None)
        o = np.argsort(x)
        ax.plot(x[o], (M @ coef)[o], "-", color=col, lw=2.0, zorder=4)
    style(ax, ylab="trials reaching the target",
          xlab="effective rotation, degrees", title="Success against effective rotation")
    ax.set_ylim(-0.03, 1.06)
    legend_below(ax, ncol=1, y=-0.22)
    fig.tight_layout()
    save(fig, FIG / "aggregate" / "02_dose_response.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    choices=["dynamics", "cells", "readout", "trajectories", "aggregate", "provenance"])
    a = ap.parse_args()
    want = lambda k: a.only in (None, k)                                  # noqa: E731

    for r in RUNS:
        if not (ROOT / r["dir"]).exists():
            print(f"[skip] {r['slug']}: {r['dir']} missing")
            continue
        print(f"\n[{r['slug']}]  {r['dir']}")
        sessions = load_run(r) if want("dynamics") or want("provenance") else []
        if want("dynamics") and sessions:
            fig_dynamics(r, sessions)
        if want("cells"):
            fig_cells(r)
        if want("readout"):
            fig_readout(r)
        if want("trajectories"):
            make_trajectories(r)
        if want("provenance") and sessions:
            provenance(r, sessions)

    if want("aggregate"):
        print("\n[aggregate]")
        rep = load_replication()
        if rep:
            fig_replication(rep)
        else:
            print("  01_replication skipped (run scripts/compare_draws.py first)")
        fig_dose_response()


if __name__ == "__main__":
    main()
