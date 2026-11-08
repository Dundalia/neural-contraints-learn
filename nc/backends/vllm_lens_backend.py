"""vLLM-lens activation backend.

A drop-in `ActivationBackend` that reads residual streams via vLLM + the
`vllm-lens` plugin (`output_residual_stream`), instead of nnsight. It is ~9x
faster on the batched teacher-forced read workload (see `docs/VLLM_LENS.md`).

Per-family chat handling is the load-bearing part. Each model family templates
its assistant turn differently and disables "thinking" differently, and the
metric must be read over exactly the response *content* tokens (not the
template wrapper or any reasoning block). The recipes below were derived by
rendering each tokenizer's chat template (see `docs/VLLM_LENS.md`):

  * Qwen 3.x  : `enable_thinking=False` prefills an empty `<think></think>`;
                the answer follows directly. assistant_prefix = "".
  * Gemma 4   : no thinking by default (`enable_thinking=True` turns it on);
                the answer follows `<|turn>model\n`. assistant_prefix = "".
                BOS is auto-prepended by the template.
  * GPT-OSS   : Harmony format. The generation prompt ends at `<|start|>assistant`
                and the real answer is a *final channel*, so the response must be
                preceded by `<|channel|>final<|message|>` (assistant_prefix), with
                `reasoning_effort="low"` to minimise analysis.

Deployment note: vLLM's default flashinfer backend JIT-compiles at init and
needs nvcc, which Mila/RunPod compute nodes lack. We force the prebuilt
FlashAttention backend (`VLLM_ATTENTION_BACKEND=FLASH_ATTN`) and `enforce_eager`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Sequence

import torch

from .base import ActivationBackend

Message = dict


# ---- per-family chat recipes -------------------------------------------------
@dataclass
class FamilyConfig:
    name: str
    nothink_template_kwargs: dict = field(default_factory=dict)
    think_template_kwargs: dict = field(default_factory=dict)
    assistant_prefix: str = ""        # context between the gen-prompt and the answer (excluded from the read span)
    stop_tokens: list = field(default_factory=list)  # turn-delimiter stop strings beyond the default EOS (gpt-oss harmony, olmo <|im_end|>)


FAMILY_CONFIGS = {
    "qwen": FamilyConfig(
        "qwen",
        nothink_template_kwargs={"enable_thinking": False},
        think_template_kwargs={"enable_thinking": True},
        assistant_prefix="",
    ),
    "gemma": FamilyConfig(
        "gemma",
        nothink_template_kwargs={},
        think_template_kwargs={"enable_thinking": True},
        assistant_prefix="",
    ),
    "gpt_oss": FamilyConfig(
        "gpt_oss",
        nothink_template_kwargs={"reasoning_effort": "low"},
        think_template_kwargs={"reasoning_effort": "medium"},
        # Suppress reasoning the in-distribution way: prefill an empty (closed) analysis channel, then the
        # final-channel opener — so the model writes the answer directly instead of skipping analysis (OOD).
        assistant_prefix="<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>",
        # Without these it overruns past the final answer into a NEW reasoning turn (verified). Stop at the
        # harmony turn/message ends. (IDs: <|return|>=200002, <|end|>=200007, <|call|>=200012.)
        stop_tokens=["<|return|>", "<|end|>", "<|call|>"],
    ),
    # Mistral-Small-3.x: `[INST]...[/INST]` then the answer directly; no thinking mode.
    # Tokenized via the mistral_common backend (not a Jinja chat_template). EOS = </s>.
    "mistral": FamilyConfig(
        "mistral",
        nothink_template_kwargs={},
        think_template_kwargs={},
        assistant_prefix="",
    ),
    # Olmo 3.x: ChatML (`<|im_start|>assistant\n` then the answer); no thinking by default. EOS = <|im_end|>.
    "olmo": FamilyConfig(
        "olmo",
        nothink_template_kwargs={},
        think_template_kwargs={},
        assistant_prefix="",
        # Olmo closes its turn with <|im_end|> but its generation EOS is <|endoftext|> (only emitted in single-turn);
        # in multi-turn it runs past <|im_end|> into hallucinated turns unless we stop on the turn delimiter (verified).
        stop_tokens=["<|im_end|>"],
    ),
}


def detect_family(model_name: str) -> str:
    n = model_name.lower()
    if "gpt-oss" in n or "gpt_oss" in n:
        return "gpt_oss"
    if "gemma" in n:
        return "gemma"
    if "qwen" in n:
        return "qwen"
    if "olmo" in n:
        return "olmo"
    if "mistral" in n or "ministral" in n or "magistral" in n:
        return "mistral"
    return "qwen"  # conservative default; pass family= explicitly for unknown models


def _chat_prompt_ids(tokenizer, messages, template_kwargs):
    """Render the chat prompt to a flat list[int]. transformers >=5 returns a
    BatchEncoding dict from apply_chat_template(tokenize=True); older returns a
    bare list/tensor. Normalise both (NEVER list() a dict — that yields its keys)."""
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, **template_kwargs)
    ids = enc["input_ids"] if (isinstance(enc, dict) or hasattr(enc, "input_ids")) else enc
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if len(ids) and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def build_teacher_forced_ids(tokenizer, messages, forced_response, family,
                             response_prefix=None, enable_thinking=False):
    """Return ``(input_ids: list[int], span_start, span_len)``.

    The span is exactly the ``forced_response`` content tokens. Everything
    before it — the chat prompt, the family assistant_prefix (e.g. GPT-OSS's
    final-channel opener), and any ``response_prefix`` (e.g. a CoT block) — is
    in-context but excluded from the span, mirroring the nnsight backend.
    """
    cfg = FAMILY_CONFIGS[family]
    tkw = cfg.think_template_kwargs if enable_thinking else cfg.nothink_template_kwargs
    prompt_ids = _chat_prompt_ids(tokenizer, messages, tkw)
    prefix = cfg.assistant_prefix + (response_prefix or "")
    ctx_ids = prompt_ids + (
        tokenizer(prefix, add_special_tokens=False)["input_ids"] if prefix else [])
    resp_ids = tokenizer(forced_response, add_special_tokens=False)["input_ids"]
    if len(resp_ids) == 0:
        raise ValueError("forced_response tokenizes to 0 tokens")
    return ctx_ids + resp_ids, len(ctx_ids), len(resp_ids)


@dataclass
class VLLMLensConfig:
    model_name: str = "Qwen/Qwen3.6-27B"
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 60000     # huge headroom (horizon-50 convs ≈ 4.4k tok); see configs/backend/vllm_lens.yaml
    family: str | None = None          # auto-detected from model_name if None
    trust_remote_code: bool = True
    enable_prefix_caching: bool = True  # cache the append-only conversation prefix across turns (needed
                                        # for prefix_cache_reads; also speeds self-generation)
    limit_mm_per_prompt: dict | None = None  # cap multimodal items; auto-set to 0 for VLM families used
                                             # text-only (e.g. Mistral3/Pixtral) so vLLM skips vision profiling
    max_num_batched_tokens: int | None = None  # ADDITIVE 2026-08-14; None = vLLM's default (unchanged).
                                        # Needed for prompt_logprobs workloads: vLLM materializes a fp32
                                        # [batched_tokens x vocab] log_softmax buffer and pipelining can hold
                                        # TWO, so a large-vocab model OOMs mid-run at the default (~8k). At
                                        # gemma-4's 262k vocab that buffer is ~2.6 GiB per 2560 tokens. NOTE
                                        # vLLM rejects values below max_tokens_per_mm_item (2496) for
                                        # multimodal models, so gemma-4 needs >=2560 and a lower gmu for slack.


class VLLMLensBackend(ActivationBackend):
    def __init__(self, config: "VLLMLensConfig | str | None" = None):
        if isinstance(config, str):
            config = VLLMLensConfig(model_name=config)
        self.config = config or VLLMLensConfig()
        # Optimization C (RUN_ARCHITECTURE_TMP.md): capture only the response-span (tail) positions
        # instead of the whole prefill. WIRED but OFF — enabling it needs a vllm_lens fork that makes
        # the EOS extra-pass trim last-N-aware (see docs note); _read_one below is fallback-safe either way.
        self.span_capture = False
        # Prefix-cached span reads (vendor/vllm_lens fork): when on, a read REUSES the cached conversation
        # prefix and only recomputes+captures the response span -> O(span) instead of O(full context).
        # Caller passes allow_prefix_cache=True ONLY when the response is NOT already cached (observer arms;
        # the self arm's response is cached from generation, so it must stay full-recompute). Default off.
        self.prefix_cache_reads = False
        # Force the prebuilt FlashAttention path so flashinfer does not JIT (no nvcc on compute nodes).
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        # Pin vLLM's V1 model runner: vLLM 0.23 auto-selects a new V2 runner for some
        # architectures (Llama/Mistral/Qwen3ForCausalLM, dense+unquantized), and the V2
        # runner dropped `input_batch`, which vllm-lens 1.1.0's capture hook reads -> the
        # hook throws and no activations attach. V1 keeps `input_batch`, so force it for
        # every subject. (Our Qwen3.5/Gemma4 *ForConditionalGeneration subjects already
        # use V1; this also covers any future Llama/Qwen3 subject.) See runpod/VLLM_LENS_BUG.md.
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
        self._load()

    def _load(self):
        from vllm import LLM, SamplingParams, TokensPrompt
        from transformers import AutoConfig, AutoTokenizer

        cfg = self.config
        self.model_name = cfg.model_name
        self.family = cfg.family or detect_family(cfg.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=cfg.trust_remote_code)
        hf_cfg = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=cfg.trust_remote_code)
        tc = getattr(hf_cfg, "text_config", hf_cfg)
        self.num_layers = int(tc.num_hidden_layers)
        self.hidden_size = int(tc.hidden_size)
        self._TokensPrompt = TokensPrompt
        self._SamplingParams = SamplingParams
        _fam = FAMILY_CONFIGS[self.family]
        self.stop_ids = [i for t in _fam.stop_tokens
                         for i in (self.tokenizer.convert_tokens_to_ids(t),)
                         if isinstance(i, int) and i >= 0 and i != self.tokenizer.unk_token_id]
        # Keep special tokens in the generation output ONLY for families whose turn-delimiter stops need
        # post-hoc extraction (gpt-oss, olmo); clean families stay on the default skip_special_tokens=True
        # (else e.g. gemma's <end_of_turn> would leak into the answer).
        self.gen_skip_special = not bool(self.stop_ids)
        llm_kwargs = {}
        lim = cfg.limit_mm_per_prompt
        if lim is None and self.family == "mistral":   # Mistral3 is a VLM we use text-only: cap mm at 0 so
            lim = {"image": 0, "video": 0}              # vLLM skips vision profiling (the dummy-image pass that
                                                        # crashes the mistral_common Pixtral processor on empty text)
        if lim is not None:
            llm_kwargs["limit_mm_per_prompt"] = lim
        if cfg.max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = int(cfg.max_num_batched_tokens)
        self.llm = LLM(model=cfg.model_name, dtype=cfg.dtype,
                       gpu_memory_utilization=cfg.gpu_memory_utilization,
                       enforce_eager=True, max_model_len=cfg.max_model_len,
                       enable_prefix_caching=cfg.enable_prefix_caching,
                       trust_remote_code=cfg.trust_remote_code, **llm_kwargs)
        self._warmup()

    def _warmup(self):
        # The first activation-capturing generate() registers the plugin hooks and
        # does vLLM's lazy init (~5x slower than steady state on a 27B). Pay it once
        # here so the first real read is warm.
        try:
            self.read_residual([{"role": "user", "content": "warm up"}],
                               self.num_layers // 2, position=None, forced_response="ok")
        except Exception as e:  # pragma: no cover
            print(f"[VLLMLensBackend] warmup skipped: {e}")

    # ---- layer / depth ----
    def layer_for_depth(self, depth: float) -> int:
        return max(0, min(self.num_layers - 1, round(depth * (self.num_layers - 1))))

    # ---- ids ----
    def _build_ids(self, messages, forced_response, response_prefix=None, enable_thinking=False):
        return build_teacher_forced_ids(self.tokenizer, messages, forced_response,
                                        self.family, response_prefix, enable_thinking)

    def _read_one(self, out, ss, sl, position):
        a = out.activations["residual_stream"]
        a = a if isinstance(a, torch.Tensor) else torch.as_tensor(a)
        if a.dim() == 3:                      # (n_layers, total_pos, d) -> single requested layer
            a = a[0]
        # The span is ALWAYS the tail (build_teacher_forced_ids: ss+sl == len). The captured tensor is
        # either the full prompt prefill (force-skip read) or just the freshly-computed query suffix
        # (prefix-cached read) — both END at the response span, so the last sl positions are the span
        # regardless. (_trim_activations already dropped the generated/EOS-surplus tokens.)
        span = a[-sl:, :].float().cpu()
        if position is None:
            return span
        idx = position if position >= 0 else sl + position
        return span[idx]

    # ---- residual read (single) ----
    def read_residual(self, messages: Sequence[Message], layer: int, position: int | None = -1,
                      forced_response: str | None = None, response_prefix: str | None = None,
                      enable_thinking: bool | None = None, allow_prefix_cache: bool = False) -> torch.Tensor:
        if not (0 <= layer < self.num_layers):
            raise IndexError(f"layer {layer} out of range [0, {self.num_layers})")
        ids, ss, sl = self._build_ids(messages, forced_response, response_prefix, bool(enable_thinking))
        sp = self._SamplingParams(temperature=0.0, max_tokens=1,
                                  extra_args=self._read_extra(layer, sl, allow_prefix_cache))
        out = self.llm.generate([self._TokensPrompt(prompt_token_ids=ids)], sp)[0]
        return self._read_one(out, ss, sl, position)

    def _read_extra(self, layer: int, sl: int, allow_prefix_cache: bool,
                    extra_left: int = 0) -> dict:
        """SamplingParams.extra_args for a lens read. span_prefix_cache (fork) is added ONLY when the
        backend has prefix_cache_reads on AND the caller allows it (response not already cached)."""
        extra = {"output_residual_stream": [layer]}
        if self.span_capture:                              # C: worker-side last-N trim (off; needs fork)
            extra["output_residual_last_n"] = sl + extra_left
        if self.prefix_cache_reads and allow_prefix_cache:  # reuse cached context, recompute+capture span only
            extra["span_prefix_cache"] = True
        return extra

    # ---- residual read (batched — the speed win) ----
    def read_residual_batch(self, items, layer: int, position: int | None = None,
                            enable_thinking: bool | None = None, allow_prefix_cache: bool = False):
        """items: list of (messages, forced_response). Returns a list of reads,
        each [span_len, hidden] (position=None) or [hidden]."""
        built = [self._build_ids(m, r, None, bool(enable_thinking)) for (m, r) in items]
        if self.span_capture or (self.prefix_cache_reads and allow_prefix_cache):
            sp = [self._SamplingParams(temperature=0.0, max_tokens=1,
                  extra_args=self._read_extra(layer, sl, allow_prefix_cache)) for (_, _, sl) in built]
        else:
            sp = self._SamplingParams(temperature=0.0, max_tokens=1,
                                      extra_args={"output_residual_stream": [layer]})
        prompts = [self._TokensPrompt(prompt_token_ids=ids) for (ids, _, _) in built]
        outs = self.llm.generate(prompts, sp)
        return [self._read_one(o, ss, sl, position) for o, (_, ss, sl) in zip(outs, built)]

    # ---- residual read (multi-layer, batched — depth-axis pilots) ----
    def read_residual_multi_batch(self, items, layers, position: int | None = None,
                                  enable_thinking: bool | None = None,
                                  allow_prefix_cache: bool = False, extra_left: int = 0):
        """Batched span read at a LIST of layers. items: list of (messages, forced_response);
        layers: iterable of layer indices. Returns a list of reads, each
        [n_layers, span_len, hidden] (position=None) or [n_layers, hidden] (position=int).

        extra_left=n (ADDITIVE, 2026-08-12) returns n EXTRA positions to the left of the span,
        so the read covers [span_start - n, span_end). The tokens are already in the captured
        tensor (the span is a tail slice of a full-prefill capture); this only keeps more of it.
        Needed when the caller wants the state that GENERATED the first response token, which
        lives at the last PROMPT position and is otherwise sliced away. Default 0 leaves every
        existing caller byte-identical.

        Dim 0 is in ASCENDING layer index order (the vllm_lens worker stacks captured
        layers sorted — see _worker_ext.get_captured_states), regardless of input order;
        duplicates are dropped. DEFAULT (and science default): the PLAIN full-context read
        path (the plugin force-sets skip_reading_prefix_cache for capturing requests).

        allow_prefix_cache=True (ADDITIVE, 2026-07-10, Experiment B option-pricing — NOT an
        adopted convention): when the backend also has prefix_cache_reads=True, requests opt
        in to the vendor fork's span_prefix_cache path — the cached context KV is REUSED and
        only the freshly-computed query suffix is captured. Token id construction is
        UNCHANGED (byte-identical sequences); the fork's cache-aware trim
        (n_prompt - num_cached_tokens + n_gen - 1) makes the capture the computed suffix,
        which always ENDS at the response span, so the tail slice below stays valid. SPAN
        ALIASING (a capture shorter than the span => the span itself hit the cache — happens
        for byte-identical sequences, e.g. two sham streams of one sentence at turn 0, or a
        real t0 read after the same-context sham): detected per item and RE-READ via the
        plain full-recompute path (the frozen convention) in one extra batched call; count
        printed. The guard therefore never silently returns a short capture."""
        layers = sorted({int(l) for l in layers})
        for l in layers:
            if not (0 <= l < self.num_layers):
                raise IndexError(f"layer {l} out of range [0, {self.num_layers})")
        built = [self._build_ids(m, r, None, bool(enable_thinking)) for (m, r) in items]
        use_pc = bool(self.prefix_cache_reads and allow_prefix_cache)
        if use_pc:
            sp = [self._SamplingParams(temperature=0.0, max_tokens=1,
                                       extra_args={"output_residual_stream": list(layers),
                                                   "span_prefix_cache": True})
                  for _ in built]
        else:
            sp = self._SamplingParams(temperature=0.0, max_tokens=1,
                                      extra_args={"output_residual_stream": list(layers)})
        prompts = [self._TokensPrompt(prompt_token_ids=ids) for (ids, _, _) in built]
        outs = self.llm.generate(prompts, sp)
        reads = [None] * len(built)
        aliased = []                              # pc-path items whose span hit the cache
        for i, (o, (_, ss, sl)) in enumerate(zip(outs, built)):
            a = o.activations["residual_stream"]
            a = a if isinstance(a, torch.Tensor) else torch.as_tensor(a)
            if a.dim() == 2:                      # defensive: (positions, d) without layer dim
                a = a.unsqueeze(0)
            if a.shape[0] != len(layers):
                raise RuntimeError(f"expected {len(layers)} captured layers, got shape {tuple(a.shape)}")
            if use_pc and a.shape[1] < sl + extra_left:
                aliased.append(i)                 # span aliased into the cache -> re-read below
                o.activations = None
                continue
            # span = tail, same invariant as _read_one (build_teacher_forced_ids: ss+sl == len)
            span = a[:, -(sl + extra_left):, :].float().cpu()
            o.activations = None                  # free the full-context all-layer capture eagerly
            if position is None:
                reads[i] = span
            else:
                idx = position if position >= 0 else sl + position
                reads[i] = span[:, idx, :]
        if aliased:                               # full-recompute retry (frozen convention)
            print(f"[vllm-lens] span_prefix_cache: {len(aliased)}/{len(built)} spans hit the "
                  f"cache (duplicate contexts) — re-read via full recompute", flush=True)
            sp_full = self._SamplingParams(temperature=0.0, max_tokens=1,
                                           extra_args={"output_residual_stream": list(layers)})
            outs2 = self.llm.generate([self._TokensPrompt(prompt_token_ids=built[i][0])
                                       for i in aliased], sp_full)
            for i, o in zip(aliased, outs2):
                _, ss, sl = built[i]
                a = o.activations["residual_stream"]
                a = a if isinstance(a, torch.Tensor) else torch.as_tensor(a)
                if a.dim() == 2:
                    a = a.unsqueeze(0)
                if a.shape[0] != len(layers) or a.shape[1] < sl + extra_left:
                    raise RuntimeError(f"full-recompute retry still short: {tuple(a.shape)} "
                                       f"vs span {sl} — STOP")
                span = a[:, -(sl + extra_left):, :].float().cpu()
                o.activations = None
                if position is None:
                    reads[i] = span
                else:
                    idx = position if position >= 0 else sl + position
                    reads[i] = span[:, idx, :]
        return reads

    # ---- residual read (multi-layer, raw token ids, FULL sequence) ----
    def read_residual_multi_ids(self, ids_list, layers):
        """Batched multi-layer capture over CALLER-BUILT token id sequences, returning the
        FULL sequence activations [n_layers, len(ids), hidden] per item (float32 cpu).

        ADDITIVE (2026-07-09, LITTLEBOY Experiment B single-pass scripted shams): the caller
        pre-renders an entire teacher-forced conversation and slices per-turn response spans
        out of ONE capture, instead of one read per turn. Layer dim ASCENDING (worker stacks
        captured layers sorted); duplicates dropped. Plain full-context read path (capturing
        requests skip the prefix cache), exactly like read_residual_multi_batch — only the
        id construction and the (absent) span trim differ. Existing methods untouched."""
        layers = sorted({int(l) for l in layers})
        for l in layers:
            if not (0 <= l < self.num_layers):
                raise IndexError(f"layer {l} out of range [0, {self.num_layers})")
        sp = self._SamplingParams(temperature=0.0, max_tokens=1,
                                  extra_args={"output_residual_stream": list(layers)})
        prompts = [self._TokensPrompt(prompt_token_ids=list(ids)) for ids in ids_list]
        outs = self.llm.generate(prompts, sp)
        reads = []
        for o, ids in zip(outs, ids_list):
            a = o.activations["residual_stream"]
            a = a if isinstance(a, torch.Tensor) else torch.as_tensor(a)
            if a.dim() == 2:
                a = a.unsqueeze(0)
            if a.shape[0] != len(layers):
                raise RuntimeError(f"expected {len(layers)} captured layers, got "
                                   f"shape {tuple(a.shape)}")
            full = a[:, -len(ids):, :].float().cpu()   # defensive: drop any EOS surplus
            o.activations = None
            reads.append(full)
        return reads

    # ---- teacher-forced sequence log-prob (ProbeInterface.calibrate consumes this) ----
    @torch.no_grad()
    def sequence_logprob(self, messages: Sequence[Message], response: str,
                         enable_thinking: bool | None = None,
                         response_prefix: str | None = None) -> dict:
        """Mean/sum teacher-forced log-prob of the ``response`` content tokens given
        ``messages`` — same span construction as :meth:`read_residual`, computed via
        vLLM ``prompt_logprobs``. Returns ``{"mean", "sum", "n_tokens"}`` to match the
        nnsight backend (the random-probe interface uses the mean for its
        logprob-decorrelation filter). Without this, ProbeInterface.calibrate raised
        AttributeError on the vLLM-lens backend (see docs/BUGS_LOG.md BUG-03)."""
        ids, ss, sl = self._build_ids(messages, response, response_prefix, bool(enable_thinking))
        # prompt_logprobs=1: per prompt position vLLM returns the top-1 token AND the
        # actual prompt token's logprob (the actual token is always included). max_tokens=1
        # because only the prefill logprobs matter, not any continuation.
        sp = self._SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
        out = self.llm.generate([self._TokensPrompt(prompt_token_ids=ids)], sp)[0]
        plp = out.prompt_logprobs            # len == len(ids); plp[0] is None (no left context)
        if not plp:
            raise RuntimeError("vLLM returned no prompt_logprobs for sequence_logprob")
        lps = []
        for i in range(ss, ss + sl):
            entry = plp[i]
            cell = entry.get(ids[i]) if entry else None
            if cell is None:
                raise RuntimeError(f"sequence_logprob: missing logprob for token {ids[i]} at pos {i}")
            lps.append(cell.logprob if hasattr(cell, "logprob") else float(cell))
        return {"mean": sum(lps) / len(lps), "sum": sum(lps), "n_tokens": sl}

    # ---- generation ----
    def generate(self, messages: Sequence[Message], max_new_tokens: int = 64,
                 do_sample: bool = False, temperature: float = 1.0, top_p: float = 1.0,
                 seed: int | None = None, enable_thinking: bool | None = None,
                 assistant_prefix: str | None = None) -> str:
        cfg = FAMILY_CONFIGS[self.family]
        tkw = cfg.think_template_kwargs if enable_thinking else cfg.nothink_template_kwargs
        prompt_ids = _chat_prompt_ids(self.tokenizer, messages, tkw)
        pre = cfg.assistant_prefix + (assistant_prefix or "")
        if pre:
            prompt_ids = prompt_ids + self.tokenizer(pre, add_special_tokens=False)["input_ids"]
        sp = self._SamplingParams(
            temperature=(temperature if do_sample else 0.0),
            top_p=(top_p if do_sample else 1.0),
            max_tokens=max_new_tokens, seed=seed,
            stop_token_ids=(self.stop_ids or None), skip_special_tokens=self.gen_skip_special)
        out = self.llm.generate([self._TokensPrompt(prompt_token_ids=prompt_ids)], sp)[0]
        return self._extract_answer(out.outputs[0].text)

    # ---- generation (batched — vLLM continuous batching, the decode speed win) ----
    def generate_batch(self, batch_messages, max_new_tokens: int = 64, do_sample: bool = False,
                       temperature: float = 1.0, top_p: float = 1.0, seeds=None,
                       enable_thinking: bool | None = None) -> list[str]:
        """Generate for many conversations in ONE vLLM call so they decode as a batch
        (instead of one sequence at a time). Returns a list of answers aligned with input."""
        cfg = FAMILY_CONFIGS[self.family]
        tkw = cfg.think_template_kwargs if enable_thinking else cfg.nothink_template_kwargs
        pre_ids = self.tokenizer(cfg.assistant_prefix, add_special_tokens=False)["input_ids"] \
            if cfg.assistant_prefix else []
        prompts = []
        for messages in batch_messages:
            pids = _chat_prompt_ids(self.tokenizer, messages, tkw) + pre_ids
            prompts.append(self._TokensPrompt(prompt_token_ids=pids))
        if seeds is None:
            seeds = [None] * len(prompts)
        sps = [self._SamplingParams(temperature=(temperature if do_sample else 0.0),
                                    top_p=(top_p if do_sample else 1.0),
                                    max_tokens=max_new_tokens, seed=s,
                                    stop_token_ids=(self.stop_ids or None),
                                    skip_special_tokens=self.gen_skip_special) for s in seeds]
        outs = self.llm.generate(prompts, sps)
        return [self._extract_answer(o.outputs[0].text) for o in outs]

    def _extract_answer(self, text: str) -> str:
        # qwen: strip a leading think block; gpt-oss: take the final channel.
        if self.family == "gpt_oss" and "<|channel|>final<|message|>" in text:
            text = text.split("<|channel|>final<|message|>")[-1]
        if "</think>" in text:
            text = text.split("</think>")[-1]
        for t in ("<|im_end|>", "<|return|>", "<|end|>", "<turn|>", "<eos>", "</s>"):
            text = text.split(t)[0]
        return text.strip()

    def __repr__(self):
        return (f"VLLMLensBackend(model={self.model_name}, family={self.family}, "
                f"layers={getattr(self,'num_layers','?')}, hidden={getattr(self,'hidden_size','?')})")
