# Adopting the prefix-cached read: gate declared before the result

Written 2026-08-15 while `bench_equiv` (job 10381527) was still running, so the threshold is chosen
against the manipulation rather than against the number that comes back.

## What is being decided

Whether the closed loop reads activations via `span_prefix_cache` (reuse the cached context, recompute
and capture only the response span) instead of full recompute. Measured at 7,151 tokens of context on
gemma-4-31B: **0.19 s versus 3.31 s, a 17.5x speedup**, and the cached read is O(span) while the full
read is O(context), so the gap widens with conversation depth. Projected session cost 21.7 -> 1.2
GPU-min for the read component.

## Why a gate is needed at all

A faster read that returns different activations is worthless. The loop consumes exactly one quantity:
the 2D velocity the readout decodes. So the gate is stated on that, not on raw tensor deltas.

## The gate (pre-declared)

Adopt the cached read if, across all four context depths (60/150/300/450 turns) through the real
fitted readout for the draw being run:

* **p99 of the decoded-direction difference <= 1.0 deg**, and
* **max <= 3.0 deg**, and
* no shape mismatch and no span-aliasing fallback on any sample.

Otherwise keep full recompute and take only the free scheduling wins.

**Why these numbers.** The grid manipulates the decoder by 50 to 87 deg and effective rotation spans
36 to 127 deg. A 3 deg worst-case perturbation is under 8% of the *smallest* manipulated angle and
under 3% of the typical one, and it applies equally to every cell, so it cannot create an alignment
effect where none exists. Anything above that starts to be a fraction of the signal rather than a
rounding error.

## Consequences if it passes

* Gemma re-runs into **fresh directories**, not into `outputs/v6g_d*`. Within one sweep the read path
  must be homogeneous, otherwise a session's cost of measurement varies across cells for no reason.
* The 42 existing full-recompute gemma sessions are kept and become the **outcome-level control**: if
  cached-read sessions in the same cells give the same success and moves distributions, that is
  stronger evidence than any tensor comparison.
* Qwen (v5, v6q) is NOT re-run. Both are complete under full recompute, and the paper's test is
  *within* a model and draw, where the read path is constant and therefore cannot confound it. The
  cross-model read-path difference is a documented caveat, not a confound.

## What is not on the table autonomously

Changing the experiment: cells, seeds, trial counts, move budget, alignment levels, the label, the
read convention (one activation per emitted letter, DESIGN 7.2d), or the stuck/repeat detectors.
Committing or pushing. Deleting any existing outputs.
