#!/usr/bin/env bash
###############################################################################
# bench.sh
#
# Wrapper around llama-bench. Runs the benchmark, prints the raw output,
# and appends a Markdown summary row to bench-results.md so you can build
# a running comparison across models and quants.
#
# Usage:
#     ./scripts/bench.sh /models/gemma-4-31B-it-NVFP4-turbo/gemma-4-31B-it-NVFP4-turbo-NVFP4.gguf
#     ./scripts/bench.sh /models/foo.gguf -ngl 100 -p 512 -n 128
#
# Any additional arguments after the model path are forwarded verbatim
# to llama-bench. Defaults: -ngl 100 (full offload).
###############################################################################

set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-/opt/llama.cpp}"
BIN="$LLAMA_DIR/build/bin/llama-bench"
RESULTS_FILE="${RESULTS_FILE:-bench-results.md}"

if [[ $# -lt 1 ]]; then
    cat <<EOF
Usage: $0 <model.gguf> [extra llama-bench args...]

Examples:
    $0 /models/gemma-4-31B-it-NVFP4-turbo/gemma-4-31B-it-NVFP4-turbo-NVFP4.gguf
    $0 /models/qwen3-30b-a3b.gguf -ngl 100 -p 1024 -n 256
EOF
    exit 1
fi

MODEL="$1"
shift

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: $BIN not found. Run ./scripts/install.sh first." >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: model not found at $MODEL" >&2
    exit 1
fi

EXTRA_ARGS=("$@")
if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
    EXTRA_ARGS=("-ngl" "100")
fi

# ---- run the benchmark and capture output ----------------------------------
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

echo "Running llama-bench on $MODEL ..."
"$BIN" -m "$MODEL" "${EXTRA_ARGS[@]}" | tee "$TMP"

# ---- append a Markdown summary block ---------------------------------------
DATE_UTC=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
BUILD=$(grep -oE 'build: [a-f0-9]+ \([0-9]+\)' "$TMP" | head -1 || echo "build: unknown")

{
    echo
    echo "## $(basename "$MODEL")"
    echo
    echo "- **Date:** $DATE_UTC"
    echo "- **GPU:** $GPU (driver $DRIVER)"
    echo "- **llama.cpp $BUILD**"
    echo "- **Args:** \`${EXTRA_ARGS[*]}\`"
    echo
    echo "<details><summary>Raw output</summary>"
    echo
    echo '```'
    cat "$TMP"
    echo '```'
    echo
    echo "</details>"
} >> "$RESULTS_FILE"

echo
echo "Summary appended to $RESULTS_FILE"
