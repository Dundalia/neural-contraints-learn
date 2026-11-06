#!/usr/bin/env python
"""Merge fit_draws shards into one report, recomputing every derived number from merged rows.

Rows carry their own draw and layer index, so concatenating row lists cannot misalign anything.
What CANNOT be merged by concatenation is the derived summary: the best layer and the gates are
medians over draws, so a shard's own values describe only its own draws. This recomputes them
from the merged rows rather than trusting whichever shard happened to be first.

    python scripts/merge_draw_shards.py --shards artifacts/v4/draws_shard*.json --out merged.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    reps = [json.loads(Path(s).read_text()) for s in sorted(a.shards)]
    if not reps:
        raise SystemExit("no shards")

    # the baselines depend only on the label and the split, so every shard must agree; if they
    # disagree the shards were not run on the same artifact and merging would be meaningless
    base = reps[0]["prompt_only_baselines"]
    for r in reps[1:]:
        b = r["prompt_only_baselines"]
        if abs(b["best_prompt_only"] - base["best_prompt_only"]) > 1e-9:
            raise SystemExit(f"shards disagree on the prompt-only baseline: "
                             f"{b['best_prompt_only']} vs {base['best_prompt_only']}")
    if len({r.get("label") for r in reps}) != 1:
        raise SystemExit("shards used different labels")

    rows = [x for r in reps for x in r["rows"]]
    draws = sorted({x["draw"] for x in rows if "draw" in x})
    ok = [x for x in rows if "r2" in x]
    if not ok:
        raise SystemExit("no shard produced a finite R2")

    layers = sorted({x["layer"] for x in ok})
    med = {L: float(np.median([x["r2"] for x in ok if x["layer"] == L])) for L in layers}
    best = max(med, key=med.get)
    g = [x for x in ok if x["layer"] == best]

    def bf(k):
        v = [x[k] for x in g if k in x and np.isfinite(x[k])]
        return float(np.median(v)) if v else float("nan")

    gates = {"r2_beats_mean_direction": bool(bf("r2") > 0),
             "beats_prompt_only": bool(bf("r2") > base["best_prompt_only"])}
    out = dict(reps[0])
    out.update({"rows": rows, "draws": len(draws), "draw_indices": draws,
                "best_layer": int(best), "gates": gates,
                "G1_PASS": bool(all(gates.values())),
                "merged_from": [str(Path(s).name) for s in sorted(a.shards)]})
    out.pop("draw_range", None)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"  merged {len(reps)} shards, draws {draws[0]}..{draws[-1]} ({len(draws)}), "
          f"{len(rows)} rows -> best layer {best} R2 {med[best]:+.3f}, gates {gates}")


if __name__ == "__main__":
    main()
