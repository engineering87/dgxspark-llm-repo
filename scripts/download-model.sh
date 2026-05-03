#!/usr/bin/env bash
###############################################################################
# download-model.sh
#
# Convenience wrapper around huggingface-cli for downloading GGUF models
# into a predictable location.
#
# Usage:
#     ./scripts/download-model.sh                                # default model
#     MODEL_REPO=meta-llama/Llama-3.3-70B-Instruct-GGUF \
#         MODEL_NAME=llama-3.3-70b ./scripts/download-model.sh
###############################################################################

set -euo pipefail

MODELS_ROOT="${MODELS_ROOT:-/models}"
MODEL_REPO="${MODEL_REPO:-google/gemma-4-31b-it-gguf}"
MODEL_NAME="${MODEL_NAME:-gemma-4-31b-it}"

if ! command -v huggingface-cli >/dev/null; then
    echo "Installing huggingface_hub CLI for the current user..."
    pip install --user huggingface_hub
    export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -d "$MODELS_ROOT" ]]; then
    sudo mkdir -p "$MODELS_ROOT"
    sudo chown "$USER:$USER" "$MODELS_ROOT"
fi

TARGET="$MODELS_ROOT/$MODEL_NAME"
mkdir -p "$TARGET"

echo "Downloading $MODEL_REPO into $TARGET"
huggingface-cli download "$MODEL_REPO" \
    --local-dir "$TARGET" \
    --local-dir-use-symlinks False

echo
echo "Done. Files:"
ls -lh "$TARGET" | head -20
