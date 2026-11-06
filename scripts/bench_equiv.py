#!/usr/bin/env python
"""Is the prefix-cached read equivalent to full recompute, at production context lengths?

WHY THIS DECIDES EVERYTHING. bench_read measured the span-cached read at 17.5x on a 7k context, which
would take a gemma session from 21.7 to 1.2 GPU-min. But a faster read that returns DIFFERENT
activations is worthless, and the first measurement showed a raw max|delta| of 0.5 with min cosine
0.99990 -- small, yet not zero. Raw tensor deltas are the wrong unit to judge that in. The closed
loop consumes exactly one thing: the 2D velocity the readout decodes from the activation. So this
measures the angular difference in THAT, through the REAL fitted readout for the draw actually being
run, over many reads spanning the context lengths a real session traverses.

THE BAR IS SET BY THE MANIPULATION. Perturbations in the grid rotate the decoder by 50 to 87 degrees
and effective rotation spans 36 to 127. A read-path difference is negligible if it is orders of
magnitude below that. It is reported as percentiles, not a mean, because what would matter is a tail
of large deviations, not the average.

CONTEXT LENGTH IS SWEPT because the cached read is O(span) while the full read is O(context): the
speedup grows with conversation depth, and so might any divergence. A single depth would report one
point on a curve as though it were the whole story.

    python scripts/bench_equiv.py --depths 60 150 300 450 --samples 24
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "llm-neurofeedback"))

import nc.manifold as mf                                                     # noqa: E402
from nc.task import BoardConfig, Trial, render_trial_end, render_user_turn   # noqa: E402
from run import fit_for_draw                                                 # noqa: E402


def conversation(n_turns, cfg, rng):
    conv, tr = [], Trial(cfg=cfg, target=(5.0, 3.0))
    for t in range(n_turns):
        conv.append({"role": "user", "content": render_user_turn(
            tr, t, first=(t == 0),
            preamble=None if t == 0 else render_trial_end(tr, t - 1))})
        conv.append({"role": "assistant", "content": " ".join(
            rng.choice(list("rlud"), int(rng.integers(3, 13))))})
    conv.append({"role": "user", "content": render_user_turn(tr, n_turns, first=False)})
    return conv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--reads", default=str(ROOT / "artifacts/v6g/reads_gemma-4-31B-it.npz"))
    ap.add_argument("--screen", default=str(ROOT / "artifacts/v6g/screen_gemma-4-31B-it.json"))
    ap.add_argument("--draw", type=int, default=9)
    ap.add_argument("--depths", type=int, nargs="+", default=[60, 150, 300, 450])
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--gpu-frac", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--out", default=str(ROOT / "artifacts/bench_equiv.json"))
    a = ap.parse_args()

    from neurofeedback.backends.vllm_lens_backend import (                   # noqa: PLC0415
        VLLMLensBackend, VLLMLensConfig)

    # the REAL readout for this draw, so the angle reported is the quantity the loop consumes
    z = np.load(a.reads, allow_pickle=True)
    screen = json.loads(Path(a.screen).read_text())
    layer = int(screen["layer"])
    li = list(z["layers"]).index(layer)
    elig = mf.eligible_coords(z["U_fit"][-1][::7].astype(np.float64))   # exactly run.py:172
    coords = mf.draw_coords(elig, mf.N_COORDS, np.random.default_rng(1000 + a.draw))
    ro, _, _ = fit_for_draw(z, li, coords)
    print(f"[equiv] layer {layer}, draw {a.draw}, {len(coords)} coordinates", flush=True)

    backend = VLLMLensBackend(VLLMLensConfig(
        model_name=a.model, gpu_memory_utilization=a.gpu_frac,
        max_model_len=a.max_model_len, enable_prefix_caching=True))

    cfg = BoardConfig()
    rows, res = [], {"model": a.model, "layer": layer, "draw": a.draw, "depths": {}}
    for depth in a.depths:
        rng = np.random.default_rng(depth)
        ang, mag, dfull, dcache, ntok = [], [], [], [], None
        for s in range(a.samples):
            conv = conversation(depth, cfg, rng)
            reply = " ".join(rng.choice(list("rlud"), int(rng.integers(3, 13))))
            item = [(conv, reply)]
            backend.prefix_cache_reads = False
            t0 = time.time()
            f = backend.read_residual_multi_batch(item, [layer], position=None,
                                                  extra_left=1)[0][0].numpy().astype(np.float64)
            dfull.append(time.time() - t0)
            backend.prefix_cache_reads = True
            t0 = time.time()
            c = backend.read_residual_multi_batch(item, [layer], position=None, extra_left=1,
                                                  allow_prefix_cache=True)[0][0].numpy().astype(np.float64)
            dcache.append(time.time() - t0)
            backend.prefix_cache_reads = False
            if f.shape != c.shape:
                print(f"  depth {depth} sample {s}: SHAPE MISMATCH {f.shape} vs {c.shape}")
                continue
            vf, vc = ro.velocity(f[:, coords]), ro.velocity(c[:, coords])
            nf, nc = np.linalg.norm(vf, axis=-1), np.linalg.norm(vc, axis=-1)
            cos = (vf * vc).sum(-1) / np.maximum(nf * nc, 1e-12)
            ang.extend(np.degrees(np.arccos(np.clip(cos, -1, 1))).tolist())
            mag.extend((nc / np.maximum(nf, 1e-12)).tolist())
        ang, mag = np.array(ang), np.array(mag)
        d = {"n_steps": int(ang.size),
             "full_s": float(np.median(dfull)), "cached_s": float(np.median(dcache)),
             "speedup": float(np.median(dfull) / max(np.median(dcache), 1e-9)),
             "angle_deg": {"median": float(np.median(ang)), "p95": float(np.percentile(ang, 95)),
                           "p99": float(np.percentile(ang, 99)), "max": float(ang.max())},
             "mag_ratio": {"median": float(np.median(mag)), "p01": float(np.percentile(mag, 1)),
                           "p99": float(np.percentile(mag, 99))}}
        res["depths"][str(depth)] = d
        print(f"  depth {depth:4d} turns: full {d['full_s']:5.2f}s  cached {d['cached_s']:5.2f}s  "
              f"{d['speedup']:5.1f}x | decoded angle med {d['angle_deg']['median']:.3f} "
              f"p99 {d['angle_deg']['p99']:.3f} max {d['angle_deg']['max']:.3f} deg "
              f"({d['n_steps']} steps)", flush=True)

    print("\n[equiv] VERDICT INPUTS")
    print("  the manipulation being measured rotates the decoder by 50-87 deg;")
    print("  a read-path difference matters only if it is comparable to that.")
    worst = max((d["angle_deg"]["max"] for d in res["depths"].values()), default=float("nan"))
    print(f"  worst decoded-direction difference across every depth and step: {worst:.3f} deg")
    Path(a.out).write_text(json.dumps(res, indent=2, default=float))
    print(f"[equiv] wrote {a.out}")


if __name__ == "__main__":
    main()
