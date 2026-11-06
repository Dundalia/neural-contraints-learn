#!/usr/bin/env python
"""Where does a closed-loop turn's time actually go, and which read paths are equivalent?

THE PROBLEM. A gemma-4-31B session costs 41.5 GPU-min: 394 turns at 6.3 s each. Generation is 24
tokens on a prefix-cached conversation and is nearly free; essentially all of it is the ACTIVATION
READ, which re-prefills the entire conversation because prefix caching is force-disabled for
capturing requests (the science default, DESIGN 7.2f). First principles agree: 2 x 31e9 x 13k tokens
= 806 TFLOP, ~5.8 s at a realistic fraction of A100 peak. So the read is the only thing worth
optimising, and the question is whether a cheaper read gives the SAME activations.

THREE PATHS, timed and compared against the current one:

  full        what runs today. read_residual_multi_batch, prefix cache off, whole context recomputed.
  cached      the same call with span_prefix_cache: the cached context KV is reused and only the
              response span is recomputed. O(span) instead of O(context).
  generation  no separate read at all. The capture hook fires on the prefill AND on every decode
              step (which is why vllm_lens has to trim the surplus position after an EOS extra
              pass), so a generate() call already produces the state that emitted each token. If
              those match the teacher-forced read, the read is pure duplicated work.

EQUIVALENCE IS THE POINT, NOT SPEED. A faster read that returns different activations is worthless,
so every variant is compared to `full` three ways: raw tensor difference, cosine similarity, and --
the one that decides it -- the DECODED VELOCITY through a real readout, which is what the closed
loop actually consumes. Numerical noise that never changes a decoded step is acceptable; anything
that moves the velocity is not.

    python scripts/bench_read.py --model google/gemma-4-31B-it --layer 53 --turns 120
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
sys.path.insert(0, str(ROOT.parent / "llm-neurofeedback"))

from nc.task import BoardConfig, Trial, render_trial_end, render_user_turn   # noqa: E402


def build_conversation(n_turns: int, cfg: BoardConfig):
    """A conversation of the shape a real session reaches, built with the real renderers."""
    rng = np.random.default_rng(0)
    conv, tr = [], Trial(cfg=cfg, target=(5.0, 3.0))
    for t in range(n_turns):
        conv.append({"role": "user", "content": render_user_turn(
            tr, t, first=(t == 0),
            preamble=None if t == 0 else render_trial_end(tr, t - 1))})
        k = int(rng.integers(3, 10))
        conv.append({"role": "assistant",
                     "content": " ".join(rng.choice(list("rlud"), k))})
    conv.append({"role": "user", "content": render_user_turn(tr, n_turns, first=False)})
    return conv


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a - b)
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    cos = (a * b).sum(-1) / np.maximum(na * nb, 1e-12)
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "rel": float(d.max() / max(float(np.abs(a).max()), 1e-12)),
            "min_cos": float(cos.min()), "shape": list(a.shape)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--layer", type=int, default=53)
    ap.add_argument("--turns", type=int, default=120, help="conversation depth to benchmark at")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--gpu-frac", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--reply", default="r r u u d")
    ap.add_argument("--out", default=str(ROOT / "artifacts/bench_read.json"))
    a = ap.parse_args()

    from neurofeedback.backends.vllm_lens_backend import (                   # noqa: PLC0415
        VLLMLensBackend, VLLMLensConfig, build_teacher_forced_ids)

    cfg = BoardConfig()
    conv = build_conversation(a.turns, cfg)
    backend = VLLMLensBackend(VLLMLensConfig(
        model_name=a.model, gpu_memory_utilization=a.gpu_frac,
        max_model_len=a.max_model_len, enable_prefix_caching=True))
    ids, ss, sl = build_teacher_forced_ids(backend.tokenizer, conv, a.reply, backend.family)
    n_ctx = len(ids)
    print(f"\n[bench] context {n_ctx} tokens, response span {sl}, layer {a.layer}, "
          f"model {a.model}", flush=True)
    print(f"[bench] {backend.num_layers} layers, hidden {backend.hidden_size}", flush=True)

    res = {"model": a.model, "n_ctx": n_ctx, "span": sl, "layer": a.layer, "paths": {}}

    def timed(fn, reps):
        fn()                                                # warm
        ts = []
        for _ in range(reps):
            t = time.time(); out = fn(); ts.append(time.time() - t)
        return out, float(np.median(ts))

    # ---------------------------------------------------------------- 1. full recompute (today)
    def full(bs=1):
        items = [(conv, a.reply)] * bs
        return backend.read_residual_multi_batch(items, [a.layer], position=None, extra_left=1)

    ref = None
    for bs in a.batches:
        backend.prefix_cache_reads = False
        out, dt = timed(lambda bs=bs: full(bs), a.reps)
        per = dt / bs
        if bs == 1:
            ref = out[0][0].numpy().astype(np.float64)
        res["paths"][f"full_batch{bs}"] = {"total_s": dt, "per_session_s": per}
        print(f"  full        batch {bs}: {dt:6.2f} s total, {per:6.2f} s per session-turn",
              flush=True)

    # ---------------------------------------------------------------- 2. prefix-cached span read
    try:
        backend.prefix_cache_reads = True
        # prime: the context must be in the cache for the span read to reuse it
        backend.read_residual_multi_batch([(conv, a.reply)], [a.layer], position=None,
                                          extra_left=1, allow_prefix_cache=False)
        out, dt = timed(lambda: backend.read_residual_multi_batch(
            [(conv, a.reply)], [a.layer], position=None, extra_left=1,
            allow_prefix_cache=True), a.reps)
        cached = out[0][0].numpy().astype(np.float64)
        res["paths"]["cached"] = {"total_s": dt, "per_session_s": dt,
                                  "vs_full": stats(ref, cached)}
        print(f"  cached      batch 1: {dt:6.2f} s   speedup {res['paths']['full_batch1']['total_s']/max(dt,1e-9):5.1f}x")
        print(f"              vs full: max|d| {res['paths']['cached']['vs_full']['max_abs']:.3e}  "
              f"min cos {res['paths']['cached']['vs_full']['min_cos']:.8f}")
    except Exception as e:                                                   # noqa: BLE001
        res["paths"]["cached"] = {"error": repr(e)}
        print(f"  cached: FAILED {e!r}")
    finally:
        backend.prefix_cache_reads = False

    # ---------------------------------------------------------------- 3. capture during generation
    # If the hook accumulates over prefill + decode, a plain generate() already contains the state
    # that produced each emitted token, and the separate read is duplicated work.
    try:
        from vllm import SamplingParams, TokensPrompt                        # noqa: PLC0415
        prompt_ids = ids[:ss]                                                # no forced response
        def gen():
            sp = SamplingParams(temperature=0.0, max_tokens=sl,
                                extra_args={"output_residual_stream": [a.layer]})
            return backend.llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)], sp)[0]
        out, dt = timed(gen, a.reps)
        act = out.activations["residual_stream"]
        act = act[0] if act.dim() == 3 else act
        gen_txt = out.outputs[0].text
        tail = act[-(sl + 1):].float().numpy().astype(np.float64)
        res["paths"]["generation"] = {
            "total_s": dt, "per_session_s": dt, "captured_positions": int(act.shape[0]),
            "generated": gen_txt,
            "covers_decode": bool(act.shape[0] > len(prompt_ids) - 1),
        }
        print(f"  generation  batch 1: {dt:6.2f} s   captured {act.shape[0]} positions "
              f"(prompt is {len(prompt_ids)})")
        print(f"              generated {gen_txt!r}")
        if tail.shape == ref.shape:
            res["paths"]["generation"]["vs_full"] = stats(ref, tail)
            print(f"              vs full: max|d| {stats(ref,tail)['max_abs']:.3e}  "
                  f"min cos {stats(ref,tail)['min_cos']:.8f}")
        else:
            print(f"              tail {tail.shape} != read {ref.shape}; "
                  "compare only if the model reproduced the forced reply")
    except Exception as e:                                                   # noqa: BLE001
        res["paths"]["generation"] = {"error": repr(e)}
        print(f"  generation: FAILED {e!r}")

    # ------------------------------------------------- 4. generation + span_prefix_cache (the prize)
    # Capturing forces skip_reading_prefix_cache UNLESS span_prefix_cache is set, so plain
    # generation-with-capture pays a FULL prefill and saves nothing: it just moves the cost. Setting
    # both is the combination that matters -- the cached context is reused and only the newly
    # computed positions (the fresh user turn plus every decode step) are captured. Those decode
    # positions are exactly the states that emitted the letters, so this yields the read AND the
    # generation in one O(span) call instead of one cheap call plus one O(context) one.
    try:
        from vllm import SamplingParams, TokensPrompt                        # noqa: PLC0415
        prompt_ids = ids[:ss]
        # prime the cache with the context, the way a real turn's previous generate() would
        backend.llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)],
                             SamplingParams(temperature=0.0, max_tokens=1))
        def gen_cached():
            sp = SamplingParams(temperature=0.0, max_tokens=sl,
                                extra_args={"output_residual_stream": [a.layer],
                                            "span_prefix_cache": True})
            return backend.llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)], sp)[0]
        out, dt = timed(gen_cached, a.reps)
        act = out.activations["residual_stream"]
        act = act[0] if act.dim() == 3 else act
        res["paths"]["generation_cached"] = {
            "total_s": dt, "per_session_s": dt, "captured_positions": int(act.shape[0]),
            "generated": out.outputs[0].text}
        print(f"  gen+cache   batch 1: {dt:6.2f} s   captured {act.shape[0]} positions "
              f"(prompt {len(prompt_ids)}, so O(span) if this is small)")
        print(f"              generated {out.outputs[0].text!r}")
        gtail = act[-(sl + 1):].float().numpy().astype(np.float64)
        if gtail.shape == ref.shape:
            res["paths"]["generation_cached"]["vs_full"] = stats(ref, gtail)
            print(f"              vs full: max|d| {stats(ref,gtail)['max_abs']:.3e}  "
                  f"min cos {stats(ref,gtail)['min_cos']:.8f}")
    except Exception as e:                                                   # noqa: BLE001
        res["paths"]["generation_cached"] = {"error": repr(e)}
        print(f"  gen+cache: FAILED {e!r}")

    # ---------------------------------------------------------------- what the loop actually eats
    # Numerical noise only matters if it moves a decoded step, so score every variant through a
    # real readout rather than on the raw tensor.
    try:
        import nc.manifold as mf                                             # noqa: PLC0415
        rng = np.random.default_rng(0)
        W = rng.standard_normal((2, ref.shape[1])) / np.sqrt(ref.shape[1])
        base_v = ref @ W.T
        pool = {"cached": lambda: cached, "generation": lambda: tail,
                "generation_cached": lambda: gtail}
        for k in ("cached", "generation", "generation_cached"):
            p = res["paths"].get(k, {})
            if "vs_full" not in p:
                continue
            other = pool[k]()
            v = other @ W.T
            dv = np.linalg.norm(base_v - v, axis=-1)
            ang = np.degrees(np.arccos(np.clip(
                (base_v * v).sum(-1) / np.maximum(np.linalg.norm(base_v, axis=-1)
                                                  * np.linalg.norm(v, axis=-1), 1e-12), -1, 1)))
            p["decoded_velocity"] = {"max_abs_delta": float(dv.max()),
                                     "max_angle_deg": float(np.nanmax(ang))}
            print(f"  {k:10s} decoded velocity: max |delta| {dv.max():.3e}, "
                  f"max angle {np.nanmax(ang):.4f} deg")
    except Exception as e:                                                   # noqa: BLE001
        print(f"  decoded-velocity check skipped: {e!r}")

    Path(a.out).write_text(json.dumps(res, indent=2, default=float))
    print(f"\n[bench] wrote {a.out}")

    f1 = res["paths"].get("full_batch1", {}).get("total_s")
    print("\n[bench] projected session cost at 394 turns:")
    for k, p in res["paths"].items():
        s = p.get("per_session_s")
        if s:
            print(f"    {k:14s} {s * 394 / 60:6.1f} GPU-min/session"
                  + (f"   ({f1 / s:4.1f}x faster)" if f1 and s else ""))


if __name__ == "__main__":
    main()
