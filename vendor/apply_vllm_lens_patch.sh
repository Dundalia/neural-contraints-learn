#!/usr/bin/env bash
# Patch vllm-lens 1.1.0 for the prefix-cached reads of Experiment 2 (backend.prefix_cache_reads).
# With the patch, a read that passes span_prefix_cache reuses the cached KV of the conversation
# and captures only the recomputed tokens of the sentence; without it, vllm-lens ignores
# span_prefix_cache and every read recomputes the whole conversation.
set -euo pipefail
PATCH="$(cd "$(dirname "$0")" && pwd)/vllm_lens_span_prefix.patch"
LENS=$(python -c "import os, vllm_lens; print(os.path.dirname(vllm_lens.__file__))")
cd "$LENS"
if grep -q span_prefix_cache _activations_plugin.py; then
  echo "vllm-lens already patched ($LENS)"
else
  patch -p1 --forward < "$PATCH"
  echo "patched $LENS/_activations_plugin.py"
fi
