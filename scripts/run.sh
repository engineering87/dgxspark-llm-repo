#!/usr/bin/env bash
###############################################################################
# run.sh
#
# Foreground launcher for llama-server. For production use, prefer the
# systemd unit in systemd/llama-server.service.
#
# Configuration: edit the variables below or override via environment.
#     MODEL_PATH=/models/foo.gguf ./scripts/run.sh
###############################################################################

set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-/opt/llama.cpp}"
BIN="$LLAMA_DIR/build/bin/llama-server"

MODEL_PATH="${MODEL_PATH:-/models/gemma-4-31B-it-NVFP4-turbo/gemma-4-31B-it-NVFP4-turbo-NVFP4.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-16384}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
THREADS="${THREADS:-16}"
PARALLEL="${PARALLEL:-4}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: $BIN not found. Run ./scripts/install.sh first." >&2
    exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: model not found at $MODEL_PATH" >&2
    echo "Hint:  ./scripts/download-model.sh" >&2
    exit 1
fi

exec "$BIN" \
    --model        "$MODEL_PATH" \
    --ctx-size     "$CTX_SIZE" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --host         "$HOST" \
    --port         "$PORT" \
    --threads      "$THREADS" \
    --parallel     "$PARALLEL" \
    --batch-size   "$BATCH_SIZE" \
    --ubatch-size  "$UBATCH_SIZE" \
    --flash-attn \
    --metrics
