#!/usr/bin/env python
"""G1 stage 1: capture calibration reads and cache them raw (docs/DESIGN.md §7.0).

This script ONLY reads. It captures the full residual stream at several depths and saves it,
so that the 100-coordinate draws, the factor analysis, the decoder and the whole G2 screen can
be done offline on CPU as many times as we like (see `fit_draws.py`, `screen.py`). The GPU pass
is the expensive part and it happens once.

No generation anywhere -- every reply is prescribed, so each chunk is one batched teacher-forced
prefill. Two forced-reply sets:

  * TARGET-REACHING (the fit set): shortest lattice path toward the target, symbol order
    randomised, truncated to a random length -- the analog of the paper's observation
    calibration, where the cursor travels to the target (§4.5).
  * RANDOM PATHS: the EMISSION SAMPLE for the reachability screen (§4.6). NOT a decodability
    validation set: with the label shift below, a state cannot know that the next letter is
    random, so it would score zero by construction. The intent-vs-target validation is the
    ambiguous-versus-unambiguous contrast instead, computed in `fit_draws.py`.

READ SITE (§4.2, re-indexed 2026-08-12): for each EMITTED letter, the activation that generated
it. Row j of the span is the state the model was in when it produced letter j, which for j = 0 is
the last PROMPT position (hence extra_left=1). A k-letter reply yields k samples and, in the
closed loop, k applied moves. The previous convention read letter j's token and labelled it with
letter j+1, so the first letter's intent was never read and its move was never applied.

    python scripts/calibrate.py --model Qwen/Qwen3.5-9B --n-fit 1200 --n-val 600
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nc.task import (BoardConfig, Trial, render_trial_end, render_user_turn,  # noqa: E402
                     shortest_path)

N_DEPTHS = 8
SYMS = ("r", "l", "u", "d")          # column order of the letter distributions


# ------------------------------------------------------------------ turn harvesting
def simulate_turns(cfg: BoardConfig, n_turns: int, rng: random.Random, mode: str):
    """Simulate sessions under literal execution and harvest (conversation, reply) pairs.

    Contexts come from real simulated play, harvested at every turn depth, so the calibration
    context distribution matches play in length as well as template. Pure CPU.

    One label per emitted letter: at iteration j the simulated marker is where the model is when
    it generates letter j, so a reply of k letters produces k labelled samples. ``ambiguous`` marks positions where two symbols are equally optimal
    (a diagonal required direction) -- there the state genuinely cannot know which symbol the
    randomised order picked, which is what makes the intent-vs-target contrast possible.
    """
    out = []
    while len(out) < n_turns:
        conv, preamble, first = [], None, True
        for t_idx in range(1, 40):
            trial = Trial(target=cfg.sample_target(rng), cfg=cfg)
            while not trial.done and len(out) < n_turns:
                if mode == "fit":
                    full = shortest_path(trial, rng)
                    k = rng.randint(cfg.k_min, max(cfg.k_min, min(len(full), cfg.k_max)))
                    moves = full[:k]
                else:
                    moves = [rng.choice(list(cfg.symbols))
                             for _ in range(rng.randint(cfg.k_min, cfg.k_max))]
                while len(moves) < cfg.k_min:
                    moves.append(rng.choice(list(cfg.symbols)))
                turn = render_user_turn(trial, t_idx, first=first, preamble=preamble)
                first, preamble = False, None
                reply = " ".join(moves)

                # Walk a copy to record, per read position j, the state BEFORE move j and the
                # label = direction of move j+1.
                labels, sim = [], Trial(target=trial.target, cfg=cfg, pos=trial.pos,
                                        moves_used=trial.moves_used)
                # ONE SAMPLE PER EMITTED LETTER (2026-08-12). At iteration j the simulated
                # marker sits exactly where the model sits when it generates letter j: moves
                # 0..j-1 applied. So the label is letter j itself, and `req` is measured from
                # that same position. The previous convention paired the state at letter j's
                # token with letter j+1, which meant the state that GENERATED THE FIRST LETTER
                # (the last prompt position) was never read and the first move was never
                # applied -- roughly 15% of the model's plan, silently dropped from the front.
                for j in range(len(moves)):
                    ix, iy = sim.ideal()
                    labels.append({"next_move": moves[j],
                                   "req": (ix, iy),
                                   "req_bin": cfg.direction_bin((ix, iy)),
                                   "ambiguous": bool(abs(ix) > 0.2 and abs(iy) > 0.2)})
                    if sim.done:
                        break
                    sim.apply(*sim.literal(moves[j]))
                if labels:
                    out.append({"conv": conv + [{"role": "user", "content": turn}],
                                "reply": reply, "labels": labels, "depth": len(conv),
                                "trial": t_idx})
                conv = conv + [{"role": "user", "content": turn},
                               {"role": "assistant", "content": reply}]
                for m in moves:
                    if trial.done:
                        break
                    trial.apply(*trial.literal(m))
            preamble = render_trial_end(trial, t_idx)
            if len(out) >= n_turns or len(conv) > 60:
                break
    rng.shuffle(out)
    return out[:n_turns]


# ------------------------------------------------------------------ reads
def read_chunked(backend, turns, layers, chunk: int):
    """Return ``(U float16 [n_layers, n_samples, hidden], meta)``.

    Asserts one-token-per-symbol: if a tokenizer merges a run, span_len != n_symbols and every
    label would silently misalign with a different letter.
    """
    L = len(layers)
    cols, meta = [[] for _ in range(L)], []
    t0 = time.time()
    for s in range(0, len(turns), chunk):
        batch = turns[s: s + chunk]
        # extra_left=1 pulls in the last PROMPT position, which is the state that generated the
        # first letter. Row j of the span is then the state that generated letter j.
        reads = backend.read_residual_multi_batch(
            [(b["conv"], b["reply"]) for b in batch], layers, position=None,
            enable_thinking=False, extra_left=1)
        for b, r in zip(batch, reads):
            n_sym = len(b["reply"].split())
            if r.shape[1] != n_sym + 1:
                raise RuntimeError(
                    f"span {r.shape[1]} != {n_sym}+1 for {b['reply']!r} -- either the tokenizer "
                    "is not giving one token per symbol, or extra_left did not take effect")
            # rows 0..k-1 are the generating states; the last row would generate the token AFTER
            # the reply and has no label.
            keep = min(r.shape[1] - 1, len(b["labels"]))
            if keep <= 0:
                continue
            for li in range(L):
                cols[li].append(r[li, :keep, :].numpy().astype(np.float16))
            meta.extend(dict(lb, depth=b["depth"], trial=b["trial"])
                        for lb in b["labels"][:keep])
        done = min(s + chunk, len(turns))
        el = time.time() - t0
        print(f"  reads {done}/{len(turns)}  {el:.0f}s  ({done/max(el,1e-9):.1f} turns/s)",
              flush=True)
    return np.stack([np.concatenate(c, 0) for c in cols]), meta


def read_letter_dists(backend, turns, chunk: int, top: int = 20):
    """Per read position, the model's own next-token distribution over the four move letters.

    THIS IS THE LABEL SOURCE as of 2026-08-11 (DESIGN 4.2b). The old label was the letter that
    came next on the forced path, which is nearly a function of the direction to the target, and
    the target is printed in the prompt. Measured consequence: the unfitted target direction
    predicted the old label at R2 0.562 against an oracle ceiling of 0.5715, so under 0.01 of
    label variance distinguished "reads the model" from "reads the prompt", and no decoder at any
    layer or any d cleared the target baseline. The model's own distribution has no rng in it and
    carries its private preference at ties, which the prompt does not determine.

    Returns ``(P, missing_frac)`` with ``P[i]`` summing to 1 over (r, l, u, d). A letter outside
    the top-`top` at a position is treated as probability 0, and how often that happens is
    reported rather than hidden. Same batching and indexing as ``read_logprobs``.
    """
    SP, TP = backend._SamplingParams, backend._TokensPrompt
    rows, missing, n, sym_ids = [], 0, 0, {}
    t0 = time.time()
    for s in range(0, len(turns), chunk):
        batch = turns[s: s + chunk]
        built = [backend._build_ids(b["conv"], b["reply"], None, False) for b in batch]
        outs = backend.llm.generate([TP(prompt_token_ids=ids) for ids, _, _ in built],
                                    SP(temperature=0.0, max_tokens=1, prompt_logprobs=top))
        for b, (ids, ss, sl), o in zip(batch, built, outs):
            moves = b["reply"].split()
            for j in range(1, min(sl, len(moves))):     # space-prefixed variants, learnt from ids
                sym_ids.setdefault(moves[j], ids[ss + j])
            plp = o.prompt_logprobs
            # plp[ss+j] is the distribution over the token AT ss+j given everything before it,
            # i.e. the distribution that generated letter j. Was ss+j+1 under the old convention.
            for j in range(min(sl, len(b["labels"]))):
                cell = plp[ss + j] if plp and plp[ss + j] else {}
                lp = np.full(len(SYMS), -np.inf)
                for k, sym in enumerate(SYMS):
                    tid = sym_ids.get(sym)
                    if tid is not None and tid in cell:
                        lp[k] = float(getattr(cell[tid], "logprob", -np.inf))
                n += 1
                fin = np.isfinite(lp)
                if not fin.any():
                    rows.append(np.full(len(SYMS), np.nan)); missing += 1; continue
                missing += int(not fin.all())
                p = np.where(fin, np.exp(lp - lp[fin].max()), 0.0)
                rows.append(p / max(p.sum(), 1e-12))
        done = min(s + chunk, len(turns))
        print(f"  dists {done}/{len(turns)}  {time.time()-t0:.0f}s", flush=True)
    return np.asarray(rows, dtype=np.float32), missing / max(n, 1)


def read_logprobs(backend, turns, chunk: int):
    """Model's own logprob of each forced move token -- the off-policy diagnostic (§4.5).
    Reported, never gated: three successive gates on this quantity were wrong."""
    SP, TP = backend._SamplingParams, backend._TokensPrompt
    out = []
    for s in range(0, len(turns), chunk):
        batch = turns[s: s + chunk]
        built = [backend._build_ids(b["conv"], b["reply"], None, False) for b in batch]
        outs = backend.llm.generate([TP(prompt_token_ids=ids) for ids, _, _ in built],
                                    SP(temperature=0.0, max_tokens=1, prompt_logprobs=1))
        for b, (ids, ss, sl), o in zip(batch, built, outs):
            plp = o.prompt_logprobs
            for j in range(min(sl, len(b["labels"]))):
                cell = plp[ss + j].get(ids[ss + j]) if plp[ss + j] else None
                out.append(float(getattr(cell, "logprob", np.nan)) if cell else np.nan)
    return np.asarray(out, dtype=float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--n-fit", type=int, default=1200)
    p.add_argument("--n-val", type=int, default=600)
    p.add_argument("--n-depths", type=int, default=N_DEPTHS)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--gpu-frac", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=24000)
    p.add_argument("--max-num-batched-tokens", type=int, default=None,
                   help="cap vLLM's prefill batch. The letter-distribution read uses "
                        "prompt_logprobs, which materializes a fp32 [batched_tokens x vocab] buffer "
                        "(x2 under step pipelining), so a large-vocab model OOMs mid-run at the "
                        "default. gemma-4 (262k vocab, multimodal) needs >=2560; vLLM rejects less.")
    p.add_argument("--out", default=str(ROOT / "artifacts"))
    a = p.parse_args()

    cfg = BoardConfig()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tag = a.model.split("/")[-1]

    print("[calibrate] simulating turns (CPU) ...", flush=True)
    fit_turns = simulate_turns(cfg, a.n_fit, random.Random(0), "fit")
    val_turns = simulate_turns(cfg, a.n_val, random.Random(1), "val")
    for nm, ts in (("fit", fit_turns), ("val", val_turns)):
        ks = [len(t["reply"].split()) for t in ts]
        print(f"  {nm}: {len(ts)} turns, reply len med {sorted(ks)[len(ks)//2]}, "
              f"depth {min(t['depth'] for t in ts)}-{max(t['depth'] for t in ts)}, "
              f"trials/session ~{max(t['trial'] for t in ts)}", flush=True)

    from nc.backends.vllm_lens_backend import VLLMLensBackend, VLLMLensConfig
    backend = VLLMLensBackend(VLLMLensConfig(
        model_name=a.model, gpu_memory_utilization=a.gpu_frac,
        max_model_len=a.max_model_len, enable_prefix_caching=False,
        max_num_batched_tokens=a.max_num_batched_tokens))
    nl = backend.num_layers
    layers = sorted({max(0, min(nl - 1, round((i + 1) * nl / (a.n_depths + 1))))
                     for i in range(a.n_depths)})
    print(f"[calibrate] {backend!r}\n  layers {layers} of {nl}", flush=True)

    print("[calibrate] reads (fit) ...", flush=True)
    U_fit, meta_fit = read_chunked(backend, fit_turns, layers, a.chunk)
    print("[calibrate] reads (val / emission sample) ...", flush=True)
    U_val, meta_val = read_chunked(backend, val_turns, layers, a.chunk)
    print("[calibrate] logprobs ...", flush=True)
    lp_fit = read_logprobs(backend, fit_turns[:400], a.chunk)
    lp_val = read_logprobs(backend, val_turns[:400], a.chunk)
    print("[calibrate] letter distributions (the label source, DESIGN 4.2b) ...", flush=True)
    P_fit, miss_fit = read_letter_dists(backend, fit_turns, a.chunk)
    P_val, miss_val = read_letter_dists(backend, val_turns, a.chunk)
    assert P_fit.shape[0] == U_fit.shape[1], (P_fit.shape, U_fit.shape)
    assert P_val.shape[0] == U_val.shape[1], (P_val.shape, U_val.shape)
    print(f"[calibrate] dists fit {P_fit.shape} (missing {miss_fit:.4f}) "
          f"val {P_val.shape} (missing {miss_val:.4f})", flush=True)

    def pack(meta):
        return {
            "next_move": np.array([m["next_move"] for m in meta]),
            "req": np.array([m["req"] for m in meta], dtype=np.float32),
            "req_bin": np.array([m["req_bin"] for m in meta], dtype=np.int16),
            "ambiguous": np.array([m["ambiguous"] for m in meta]),
            "depth": np.array([m["depth"] for m in meta], dtype=np.int32),
        }

    out = Path(a.out) / f"reads_{tag}.npz"
    np.savez(out, U_fit=U_fit, U_val=U_val, layers=np.array(layers), num_layers=nl,
             hidden=backend.hidden_size, lp_fit=lp_fit, lp_val=lp_val,
             fit_letter_p=P_fit, val_letter_p=P_val, letter_order=np.array(SYMS),
             **{f"fit_{k}": v for k, v in pack(meta_fit).items()},
             **{f"val_{k}": v for k, v in pack(meta_val).items()})
    report = {
        "model": a.model, "layers": layers, "num_layers": nl, "hidden": backend.hidden_size,
        "n_fit_turns": len(fit_turns), "n_val_turns": len(val_turns),
        "n_fit_samples": int(U_fit.shape[1]), "n_val_samples": int(U_val.shape[1]),
        "samples_per_turn_fit": round(U_fit.shape[1] / len(fit_turns), 2),
        "logprob_median_fit_target_reaching": float(np.nanmedian(lp_fit)),
        "logprob_median_val_random": float(np.nanmedian(lp_val)),
        "ambiguous_fraction_fit": float(np.mean([m["ambiguous"] for m in meta_fit])),
        "reads_bytes": int(U_fit.nbytes + U_val.nbytes),
        "letter_dist_missing_frac_fit": float(miss_fit),
        "letter_dist_missing_frac_val": float(miss_val),
    }
    (Path(a.out) / f"reads_{tag}.json").write_text(json.dumps(report, indent=2))
    print(f"\n[calibrate] fit {U_fit.shape}  val {U_val.shape}  "
          f"({report['reads_bytes']/1e9:.2f} GB)")
    print(f"[calibrate] logprob median: target-reaching {report['logprob_median_fit_target_reaching']:.2f}"
          f"  random {report['logprob_median_val_random']:.2f}")
    print(f"[calibrate] reads -> {out}", flush=True)


if __name__ == "__main__":
    main()
