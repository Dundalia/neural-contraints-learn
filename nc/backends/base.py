"""Swappable backend interface for the neurofeedback experiment.

A backend provides exactly two primitives the experiment needs:

* ``generate`` — chat generation (greedy or sampled), and
* ``read_residual`` — read the residual-stream vector at a chosen decoder block
  and token position (the site a probe reads to compute the feedback score).

Keeping this behind an abstract interface lets us swap the activation engine
(nnsight now; a vLLM-generation path later) without touching experiment code.

Layer-index convention
----------------------
``layer`` is the index of the decoder block whose **output** (the residual
stream *after* that block) is read, with ``layer ∈ [0, num_layers - 1]``. This
equals HuggingFace ``output_hidden_states`` index ``layer + 1`` (since HF index
0 is the embedding output). The middle of the network is ``num_layers // 2``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch

# A chat message: {"role": "system" | "user" | "assistant", "content": str}
Message = dict


class ActivationBackend(ABC):
    """Generate text and read residual-stream activations from an LM."""

    #: number of decoder blocks
    num_layers: int
    #: residual-stream width
    hidden_size: int
    #: device the model's input embeddings live on
    device: torch.device

    @abstractmethod
    def generate(
        self,
        messages: Sequence[Message],
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
        enable_thinking: bool | None = None,
        assistant_prefix: str | None = None,
    ) -> str:
        """Generate the assistant reply to ``messages``; return decoded text only.

        ``enable_thinking`` (Qwen3.x): ``None`` uses the template default,
        ``False`` pre-fills an empty ``<think></think>`` for a direct answer.
        ``assistant_prefix``: text prepended to the assistant turn before
        generation continues (used to close a truncated ``<think>`` block and let
        the model produce its answer).
        """

    @abstractmethod
    def read_residual(
        self,
        messages: Sequence[Message],
        layer: int,
        position: int | None = -1,
        forced_response: str | None = None,
        response_prefix: str | None = None,
        enable_thinking: bool | None = None,
    ) -> torch.Tensor:
        """Residual stream at ``layer``.

        If ``forced_response`` is given, it is appended as the assistant turn
        (after the generation prompt) and ``position`` indexes *within* that
        response — ``position=-1`` is its last content token, i.e. the Paper-3
        fixed-sentence probe site (not the trailing ``<|im_end|>``).

        ``response_prefix`` is assistant text placed before ``forced_response``
        (e.g. a CoT ``<think>…</think>`` block): it is part of the attended
        context but **excluded from the read span**, so only the answer is read.

        ``position=int``  -> a single token vector, 1-D float32 CPU ``[hidden_size]``.
        ``position=None`` -> the whole span, 2-D float32 CPU ``[span_len, hidden_size]``
        (every response token, or the whole conversation if no forced response) —
        used for mean-over-tokens pooling.
        """

    # ---- reserved for the intervention arms (steering, latent slots) -------
    def add_intervention(self, *args, **kwargs):  # noqa: D401
        """Hook for activation interventions — not implemented in this backend yet."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement interventions yet"
        )
