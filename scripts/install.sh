#!/usr/bin/env bash
###############################################################################
# install.sh
#
# Build llama.cpp with CUDA 13 support, targeting Blackwell sm_121 (GB10).
# Idempotent: safe to re-run.
#
# Usage:
#     ./scripts/install.sh                  # default: clones into /opt/llama.cpp
#     LLAMA_DIR=$HOME/llama.cpp ./scripts/install.sh
###############################################################################

set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-/opt/llama.cpp}"
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_REF="${LLAMA_REF:-master}"
JOBS="${JOBS:-$(nproc)}"

# ---- pretty output ---------------------------------------------------------
c_reset="\033[0m"; c_red="\033[1;31m"; c_grn="\033[1;32m"; c_ylw="\033[1;33m"
log()  { printf "${c_grn}==>${c_reset} %s\n" "$*"; }
warn() { printf "${c_ylw}!!${c_reset} %s\n"  "$*"; }
err()  { printf "${c_red}xx${c_reset} %s\n"  "$*" >&2; }

# ---- preflight checks ------------------------------------------------------
log "Verifying toolchain"

command -v git    >/dev/null || { err "git not found";    exit 1; }
command -v cmake  >/dev/null || { err "cmake not found";  exit 1; }
command -v gcc    >/dev/null || { err "gcc not found";    exit 1; }

if ! command -v nvcc >/dev/null; then
    err "nvcc not found. Install CUDA Toolkit 13.0 first."
    exit 1
fi

NVCC_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "unknown")
if [[ "$NVCC_VERSION" != 13.* ]]; then
    err "nvcc reports CUDA $NVCC_VERSION; CUDA 13.x is required for sm_121."
    err "Hint: sudo update-alternatives --config cuda  (or set PATH manually)"
    exit 1
fi
log "nvcc OK (CUDA $NVCC_VERSION)"

if ! command -v nvidia-smi >/dev/null; then
    warn "nvidia-smi not found; proceeding but GPU runtime check will be skipped."
else
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    log "GPU detected: $GPU_NAME"
fi

# ---- clone or update -------------------------------------------------------
if [[ ! -d "$LLAMA_DIR/.git" ]]; then
    log "Cloning llama.cpp into $LLAMA_DIR"
    sudo mkdir -p "$(dirname "$LLAMA_DIR")"
    sudo chown "$USER:$USER" "$(dirname "$LLAMA_DIR")"
    git clone "$LLAMA_REPO" "$LLAMA_DIR"
else
    log "Updating existing llama.cpp checkout"
    git -C "$LLAMA_DIR" fetch --all --tags --prune
fi

git -C "$LLAMA_DIR" checkout "$LLAMA_REF"
git -C "$LLAMA_DIR" pull --ff-only || true

# ---- configure -------------------------------------------------------------
log "Configuring CMake (CUDA backend, sm_121, FP16 accumulation)"

cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_F16=ON \
    -DCMAKE_CUDA_ARCHITECTURES=121 \
    -DLLAMA_CURL=ON

# ---- build -----------------------------------------------------------------
log "Building with $JOBS parallel jobs"
cmake --build "$LLAMA_DIR/build" -j "$JOBS" --config Release

# ---- verify ----------------------------------------------------------------
BIN="$LLAMA_DIR/build/bin/llama-server"
if [[ ! -x "$BIN" ]]; then
    err "Build succeeded but $BIN is missing. Investigate the build log."
    exit 1
fi

log "Build complete: $BIN"
"$BIN" --version || true

cat <<EOF

${c_grn}Done.${c_reset}

Next steps:
    1. Download a GGUF model:    ./scripts/download-model.sh
    2. Run the server:           ./scripts/run.sh
    3. Or install the systemd unit (see README "Run as a systemd service").
EOF
