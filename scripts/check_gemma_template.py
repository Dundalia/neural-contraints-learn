#!/usr/bin/env python
"""Does the board protocol survive Gemma 4's chat template? CPU-only, no GPU needed.

WHY THIS EXISTS. Every label in this experiment is positional: row j of a read span is the state
that generated letter j (DESIGN 7.2d). That mapping rests on two tokenizer facts, both of which are
Qwen facts until checked on Gemma:

  1. ONE TOKEN PER MOVE SYMBOL. The reply "r r u d" must be exactly 4 tokens. If a tokenizer merges
     a run into " r r", the span is shorter than the letter count and every label after the merge is
     attached to the wrong letter. calibrate.py asserts this, so it fails loudly -- but it fails
     after loading a 31B model, which is an expensive way to learn it.
  2. THE GENERATION PROMPT ENDS WHERE THE ANSWER STARTS, with no family prefix in between, so the
     forced-response tokens are exactly the span.

Gemma adds a third risk the other families do not have: its template REJECTS a system role. The
board protocol already folds the instructions into the first user message (task.render_user_turn),
so this should pass, but "should" is what this script replaces.

    python scripts/check_gemma_template.py --model google/gemma-4-31B-it
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nc.task import BoardConfig, Trial, render_user_turn, render_trial_end   # noqa: E402

SYMS = ("r", "l", "u", "d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    a = ap.parse_args()

    from transformers import AutoTokenizer                                   # noqa: PLC0415
    from nc.backends.vllm_lens_backend import (                   # noqa: PLC0415
        FAMILY_CONFIGS, build_teacher_forced_ids, detect_family)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    fam = detect_family(a.model)
    cfg_fam = FAMILY_CONFIGS[fam]
    print(f"model  {a.model}")
    print(f"family {fam}   assistant_prefix {cfg_fam.assistant_prefix!r}   "
          f"stop_tokens {cfg_fam.stop_tokens}   nothink_kwargs {cfg_fam.nothink_template_kwargs}")
    fails = []

    # ---- 1. one token per move symbol, in every position a reply can put them
    print("\n[1] one token per move symbol")
    for reply in ("r", "r r", "r r r r", "u d u d", "r u r u r u r",
                  " ".join(["r"] * 12), " ".join(SYMS * 3)):
        ids = tok(reply, add_special_tokens=False)["input_ids"]
        n = len(reply.split())
        ok = len(ids) == n
        if not ok:
            fails.append(f"reply {reply!r}: {len(ids)} tokens for {n} symbols "
                         f"-> {[tok.decode([i]) for i in ids]}")
        print(f"  {'ok ' if ok else 'FAIL'} {reply!r:34s} {n:2d} symbols -> {len(ids):2d} tokens")

    # ---- 2. does the template accept our message list, and does it reject a system role?
    print("\n[2] template behaviour on the board protocol's message shape")
    cfg = BoardConfig()
    trial = Trial(cfg=cfg, target=(5.0, 3.0))
    first = render_user_turn(trial, 0, first=True)
    convo = [{"role": "user", "content": first}]
    try:
        tok.apply_chat_template(convo, add_generation_prompt=True, tokenize=False,
                               **cfg_fam.nothink_template_kwargs)
        print("  ok   single user turn renders")
    except Exception as e:                                                   # noqa: BLE001
        fails.append(f"single user turn failed to render: {e}")
        print(f"  FAIL single user turn: {e}")
    try:
        tok.apply_chat_template([{"role": "system", "content": "x"},
                                 {"role": "user", "content": "y"}],
                                add_generation_prompt=True, tokenize=False)
        print("  note this template ACCEPTS a system role (we still do not use one)")
    except Exception as e:                                                   # noqa: BLE001
        print(f"  note this template REJECTS a system role ({type(e).__name__}), which is exactly")
        print("       why the board protocol folds instructions into the first user message")

    # ---- 3. a multi-turn conversation the way session.py grows it
    print("\n[3] multi-turn, grown the way nc/session.py grows it")
    conv = [{"role": "user", "content": first}]
    for t in range(1, 4):
        conv.append({"role": "assistant", "content": "r r u"})
        tr2 = Trial(cfg=cfg, target=(2.0, 6.0))
        conv.append({"role": "user", "content": render_user_turn(
            tr2, t, preamble=render_trial_end(trial, t - 1))})
    try:
        txt = tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=False,
                                      **cfg_fam.nothink_template_kwargs)
        print(f"  ok   {len(conv)} messages render, {len(tok(txt, add_special_tokens=False)['input_ids'])} tokens")
        print("  --- tail of the rendered prompt (what precedes the model's letters) ---")
        print("  " + repr(txt[-160:]))
    except Exception as e:                                                   # noqa: BLE001
        fails.append(f"multi-turn render failed: {e}")
        print(f"  FAIL multi-turn: {e}")

    # ---- 4. THE LOAD-BEARING ONE: span == n letters, and extra_left=1 reaches the prompt position
    print("\n[4] teacher-forced span, the read convention (DESIGN 7.2d)")
    for reply in ("r r u", "d", " ".join(["u"] * 9)):
        ids, ss, sl = build_teacher_forced_ids(tok, conv, reply, fam)
        n = len(reply.split())
        ok = sl == n and ss + sl == len(ids)
        if not ok:
            fails.append(f"span for {reply!r}: sl={sl} n={n} ss+sl={ss + sl} len={len(ids)}")
        # the assertion calibrate.py makes: span_len with extra_left=1 must be n+1
        print(f"  {'ok ' if ok else 'FAIL'} reply {reply!r:22s} span_len {sl:2d} == {n:2d} letters, "
              f"span is the tail ({ss}+{sl}=={len(ids)})")
        print(f"       span tokens {[tok.decode([i]) for i in ids[ss:]]}")
        print(f"       one to the left (extra_left=1, the state that generates letter 0): "
              f"{tok.decode([ids[ss - 1]])!r}")

    # ---- 5. BOS not doubled: the template prepends it, so the response must not add another
    print("\n[5] special-token hygiene")
    p_ids = tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=True,
                                    **cfg_fam.nothink_template_kwargs)
    p_ids = p_ids["input_ids"] if isinstance(p_ids, dict) or hasattr(p_ids, "input_ids") else p_ids
    if hasattr(p_ids, "tolist"):
        p_ids = p_ids.tolist()
    if p_ids and isinstance(p_ids[0], list):
        p_ids = p_ids[0]
    n_bos = sum(1 for i in p_ids if i == tok.bos_token_id)
    print(f"  bos id {tok.bos_token_id} ({tok.bos_token!r}) appears {n_bos}x in the prompt")
    print(f"  eos id {tok.eos_token_id} ({tok.eos_token!r})")
    resp_ids = tok("r r u", add_special_tokens=False)["input_ids"]
    if tok.bos_token_id in resp_ids:
        fails.append("bos leaked into the forced response")
        print("  FAIL bos leaked into the forced response")
    else:
        print("  ok   no bos in the forced response (add_special_tokens=False holds)")
    # letter ids for the label read, and their space-prefixed variants
    print("\n[6] letter token ids used for the label distribution")
    for s in SYMS:
        bare = tok(s, add_special_tokens=False)["input_ids"]
        pre = tok(" " + s, add_special_tokens=False)["input_ids"]
        print(f"  {s!r}: bare {bare} {[tok.decode([i]) for i in bare]}   "
              f"space-prefixed {pre} {[tok.decode([i]) for i in pre]}")
        if len(pre) != 1:
            fails.append(f"space-prefixed {s!r} is {len(pre)} tokens, label read assumes 1")

    print("\n" + "=" * 72)
    if fails:
        print(f"FAIL ({len(fails)})")
        for f in fails:
            print("  - " + f)
        raise SystemExit(1)
    print("PASS: the board protocol's positional label mapping holds under this template")


if __name__ == "__main__":
    main()
