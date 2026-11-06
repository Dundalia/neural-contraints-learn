#!/usr/bin/env python
"""Draw single-trial marker trajectories, one arrow per applied move.

WHY THIS EXISTS. Every other figure in this project aggregates: success rates, medians, slopes. A
trajectory shows what actually happened to the marker on one trial, which is the only view that
makes a failure legible. A trial that spirals, one that stalls just outside the acquisition disk,
and one that marches off in a fixed wrong direction all produce the same "36 moves, not acquired".

READS `path` FROM THE TURN RECORDS (added 2026-08-12). pos_before and pos_after only bracket a
whole reply, so per-step arrows need the per-move positions. A trial spans several replies, and its
trajectory is those paths concatenated in turn order.

COLOUR IS SEQUENTIAL, NOT CATEGORICAL. Steps are grouped by the reply that produced them, and reply
order is an ordinal sequence, so a single-hue light-to-dark ramp is the correct encoding: early
replies light, late replies dark. Cycling categorical hues here would imply the replies are
unordered kinds, which they are not.

    python scripts/plot_trajectories.py --dir outputs/v4 --per-condition 3
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SURF, INK, INK2, MUTED = "#ffffff", "#1a1a1a", "#4a4a4a", "#8a8a8a"
GOOD, BAD = "#009E73", "#D55E00"
# Phase tints. The whole figure is ONE arm, so colour is free to encode the phase instead. Light
# and very transparent: the background must separate the phases at a glance without competing with
# the arrows, which carry the actual data.
PHASE_TINT = {"baseline": ("#009E73", 0.10), "perturbation": ("#D55E00", 0.13),
              "washout": ("#0072B2", 0.10)}
RADIUS, ACQ = 8.0, 1.5

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "axes.titlesize": 16,
    "legend.fontsize": 16, "axes.labelsize": 16,
    "font.size": 15, "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "figure.dpi": 130,
})


def load_trials(dirs):
    """One entry per trial: its concatenated per-move path plus full provenance."""
    out = []
    for d in dirs:
        mp = Path(d) / "meta.json"
        if not mp.exists():
            continue
        meta = {m["session"]: m for m in json.loads(mp.read_text())}
        for f in sorted(glob.glob(f"{d}/sess_*.jsonl")):
            idx = int(Path(f).stem.split("_")[1])
            m = meta.get(idx)
            if m is None:
                continue
            pert = m.get("perturbation") or {}
            per = defaultdict(list)
            for line in Path(f).open():
                try:
                    r = json.loads(line)
                except Exception:                                     # noqa: BLE001
                    continue
                if r.get("path"):
                    per[r["trial"]].append(r)
            for trial, turns in per.items():
                blk = turns[0]["block"]
                # a washout trial is only interpretable next to what preceded it
                prev = pert.get("alignment") if blk == "washout" else None
                out.append({
                    "dir": Path(d).name, "session": idx, "trial": trial, "block": blk,
                    "draw": m.get("draw"), "repeat": m.get("repeat"),
                    "alignment": pert.get("alignment") if blk == "perturbation" else None,
                    "after_alignment": prev,
                    "theta_deg": pert.get("theta_deg"), "dev_deg": pert.get("dev_deg"),
                    "mag": pert.get("mag"), "sham": bool(pert.get("sham")),
                    "target": turns[0]["target"],
                    "replies": [t["reply"] for t in turns],
                    "paths": [t["path"] for t in turns],
                    "acquired": bool(turns[-1].get("acquired")),
                    "moves": sum(t["n_moves"] for t in turns),
                })
    return out


def title_for(t):
    """Provenance, in reading order: which run, which condition, what happened."""
    where = f"{t['dir']} session {t['session']}, draw {t['draw']}, seed {t['repeat']}, trial {t['trial']}"
    if t["block"] == "perturbation" and t["alignment"] is None:
        # a no-perturbation session still HAS a perturbation block; nothing was swapped in it
        cond = "perturbation block of a no-perturbation session, readout never swapped"
    elif t["block"] == "perturbation":
        cond = (f"perturbed: alignment {t['alignment']:.2f}, rotation {t['theta_deg']:.0f} deg, "
                f"severity {t['dev_deg']:.1f} deg, speed {t['mag']:.2f} of normal")
    elif t["block"] == "washout":
        cond = ("washout, original readout restored" +
                (f" after alignment {t['after_alignment']:.2f}"
                 if t["after_alignment"] is not None else " after the no-perturbation block"))
    else:
        cond = "baseline, original readout" + (", no-perturbation session" if t["sham"] else "")
    got = (f"reached the target in {t['moves']} moves" if t["acquired"]
           else f"never arrived, {t['moves']} moves used")
    # wrap: an unwrapped provenance line is wider than the axes and bleeds into the next panel
    return "\n".join(textwrap.fill(s, 52) for s in (where, cond, got))


def panel(ax, t, tint=True, lim=None, labels=True):
    """The template: one trial, one arrow per applied move, colour by reply order."""
    if tint:
        hexcol, alpha = PHASE_TINT.get(t["block"], ("#8a8a8a", 0.08))
        ax.set_facecolor(matplotlib.colors.to_rgba(hexcol, alpha))
    gx, gy = t["target"]
    ax.add_patch(plt.Circle((0, 0), RADIUS, fill=False, ec=MUTED, lw=0.8, ls=(0, (4, 4)),
                            zorder=1))
    ax.add_patch(plt.Circle((gx, gy), ACQ, fill=True, fc=GOOD, alpha=0.16, ec=GOOD, lw=1.2,
                            zorder=2))
    ax.plot([gx], [gy], marker="*", ms=11, color=GOOD, zorder=5)
    ax.plot([0], [0], marker="o", ms=5, color=INK, zorder=5)

    n = max(len(t["paths"]), 1)
    cmap = plt.get_cmap("Blues")
    for i, path in enumerate(t["paths"]):
        col = cmap(0.35 + 0.6 * (i / max(n - 1, 1)))
        p = np.asarray(path, dtype=float)
        for a, b in zip(p[:-1], p[1:]):
            d = b - a
            if np.hypot(*d) < 1e-9:                 # a zero-length move still counts as a move
                ax.plot([a[0]], [a[1]], marker="x", ms=4, color=col, zorder=4)
                continue
            ax.annotate("", xy=b, xytext=a, zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2,
                                        shrinkA=0, shrinkB=0, mutation_scale=9))
    end = np.asarray(t["paths"][-1][-1], dtype=float)
    ax.plot([end[0]], [end[1]], marker="s", ms=5,
            color=GOOD if t["acquired"] else BAD, zorder=6)

    if lim is None:
        lim = max(RADIUS + 3.0,
                  np.abs(np.concatenate([np.asarray(p) for p in t["paths"]])).max() + 1.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ticks = [x for x in (-10, -5, 0, 5, 10) if abs(x) <= lim]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if not labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(color="#e6e6e2", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_title(t.get("short_title") or title_for(t), loc="left", pad=7, fontsize=15.5)


def phase_of(t):
    return t["block"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/v4")
    ap.add_argument("--out", default="figures/trajectories")
    ap.add_argument("--arm", default="1.0",
                    help="which arm to follow: an alignment like 1.0, or 'control'")
    ap.add_argument("--session", type=int, default=None,
                    help="a specific session; default is the first complete session of that arm")
    ap.add_argument("--n-baseline", type=int, default=5)
    ap.add_argument("--n-perturb", type=int, default=10)
    ap.add_argument("--n-washout", type=int, default=5,
                    help="trials sampled per phase, spread evenly across it rather than at "
                         "random, so the panels step through the phase in order")
    ap.add_argument("--cols", type=int, default=5)
    a = ap.parse_args()

    # any subdirectory holding a meta.json: the v4 layout used shard*, the v5/v6 campaign
    # writes c<chain>g<gpu>, and hardcoding "shard*" silently found nothing under the newer runs
    dirs = [d for d in sorted(glob.glob(str(ROOT / a.dir / "*")))
            if (Path(d) / "meta.json").exists()] or [str(ROOT / a.dir)]
    trials = load_trials(dirs)
    if not trials:
        raise SystemExit(f"no trials with a logged path under {a.dir}")

    want = None if a.arm == "control" else round(float(a.arm), 1)

    def arm_of(t):
        return None if t["block"] != "perturbation" else (
            None if t["alignment"] is None else round(t["alignment"], 1))

    # a session belongs to an arm by what its perturbation block used, which every trial in the
    # session shares; read it off the perturbation trials
    sess_arm = {}
    for t in trials:
        if t["block"] == "perturbation":
            sess_arm[(t["dir"], t["session"])] = (
                None if t["alignment"] is None else round(t["alignment"], 1))
    cands = sorted(k for k, v in sess_arm.items() if v == want)
    if not cands:
        raise SystemExit(f"no session found for arm {a.arm}")
    if a.session is not None:
        cands = [k for k in cands if k[1] == a.session] or cands[:1]
    # prefer a session with all three phases present
    key = None
    for c in cands:
        blocks = {t["block"] for t in trials if (t["dir"], t["session"]) == c}
        if {"baseline", "perturbation", "washout"} <= blocks:
            key = c
            break
    key = key or cands[0]
    mine = sorted((t for t in trials if (t["dir"], t["session"]) == key),
                  key=lambda t: t["trial"])
    label = "no perturbation" if want is None else f"alignment {want:.1f}"
    print(f"[traj] following {label}: {key[0]} session {key[1]}, {len(mine)} trials")

    # sample evenly WITHIN each phase, keeping global trial order
    picked = []
    counts = {"baseline": a.n_baseline, "perturbation": a.n_perturb, "washout": a.n_washout}
    for phase in ("baseline", "perturbation", "washout"):
        g = [t for t in mine if t["block"] == phase]
        if not g:
            continue
        n = min(counts[phase], len(g))
        idx = np.linspace(0, len(g) - 1, n).round().astype(int)
        take = [g[i] for i in sorted(set(idx))]
        print(f"  {phase:13s} {len(g):3d} trials -> panels at trials "
              f"{[x['trial'] for x in take]}  ({sum(x['acquired'] for x in take)} reached)")
        picked += take
    picked.sort(key=lambda t: t["trial"])

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    for t in picked:
        t["short_title"] = (f"trial {t['trial']}, {t['block']}\n"
                            + ("reached in %d moves" % t["moves"] if t["acquired"]
                               else "never arrived, %d moves" % t["moves"]))

    cols = min(a.cols, len(picked))
    rows = int(np.ceil(len(picked) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    # ONE shared limit for every panel, so the single labelled panel speaks for all of them
    lim = max(RADIUS + 3.0, max(np.abs(np.concatenate([np.asarray(q) for q in x["paths"]])).max()
                                for x in picked) + 1.5)
    label_at = cols * (rows - 1)                     # bottom-left, the usual reference position
    for i, (ax, t) in enumerate(zip(axes, picked)):
        panel(ax, t, lim=lim, labels=(i == label_at))
    for ax in axes[len(picked):]:
        ax.axis("off")

    first = picked[0]
    head = (f"{label}, {key[0]} session {key[1]}, draw {first['draw']}, target seed "
            f"{first['repeat']}")
    if want is not None:
        head += (f"   |   rotation {first['theta_deg']:.0f} deg, severity "
                 f"{[x for x in picked if x['block']=='perturbation'][0]['dev_deg']:.1f} deg")
    handles = [plt.Rectangle((0, 0), 1, 1,
                             fc=matplotlib.colors.to_rgba(PHASE_TINT[b][0], PHASE_TINT[b][1] * 3),
                             ec="none")
               for b in ("baseline", "perturbation", "washout")]
    fig.legend(handles, ["baseline, original readout", "perturbation, readout rotated",
                         "washout, original readout restored"],
               frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.022), ncol=3,
               columnspacing=2.6, fontsize=17)
    fig.suptitle(head, x=0.006, ha="left", weight="bold", fontsize=21, y=0.998)
    fig.tight_layout()
    name = f"sequence_{'control' if want is None else ('a%.1f' % want)}_s{key[1]:03d}.png"
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)
    print(f"[traj] wrote {out / name} ({len(picked)} panels in trial order)")


if __name__ == "__main__":
    main()
