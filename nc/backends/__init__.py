"""Activation-reading backend, vendored from Dundalia/llm-neurofeedback (neurofeedback/backends/, version of 2026-08-14,
the one every run in this repo used). Reads residual streams through vLLM plus the vllm-lens plugin."""
from .base import ActivationBackend, Message

__all__ = ["ActivationBackend", "Message"]
