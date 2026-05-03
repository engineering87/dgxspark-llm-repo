# Architecture

This document explains how the components of the stack fit together and why each choice was made.

## Components

### `llama.cpp`

A C++ inference engine for GGUF models. We use it because:

- **Zero Python at runtime.** No CUDA-Python ABI mismatches, no `pip` dependency hell at deploy time.
- **OpenAI-compatible HTTP server** built in (`llama-server`). No need to wrap it.
- **Mature CUDA backend.** The `ggml-cuda` kernels are tuned per architecture, including Blackwell.
- **Streaming, tool calling, embeddings, reranking** all available out of the box.

### CUDA 13 toolchain

CUDA 13 is required for Blackwell `sm_121` codegen. CUDA 12.x produces `ptxas` errors on the new MMA instructions used by Blackwell tensor cores. The `cuda-nvcc-13-0` package is the relevant one on DGX OS.

### `systemd`

The deployment supervisor. We picked `systemd` over Docker because:

- The host already runs it.
- Direct GPU access is trivial (no NVIDIA Container Toolkit shim).
- Logs land in the unified journal.
- Restart policy and resource limits are first-class.

A Docker recipe is on the roadmap as an alternative, not a replacement.

## Data path

For a single chat completion request:

1. **HTTP arrives** at `llama-server` on port 8080.
2. **Tokenization** runs on the Grace CPU. The tokenizer is small and benefits from CPU cache locality.
3. **KV cache lookup** in unified memory. If this is a continuation, prior keys/values are still resident.
4. **Prefill** runs the prompt through the model on the GPU. Throughput here is compute-bound for short prompts, bandwidth-bound for long ones.
5. **Decode loop** generates tokens one at a time. Each token requires a full forward pass over all parameters; this is the bandwidth-bound phase. On GB10, peak decode throughput is fundamentally limited to roughly `memory_bandwidth / model_size_in_bytes`. For a 31B Q4_K_M model (~16 GB) on 273 GB/s, the theoretical ceiling is about 17 t/s; NVFP4 with hardware FP4 acceleration roughly doubles that.
6. **Streaming** sends each token back over the open HTTP connection as a server-sent event.

## Why unified memory matters

On a discrete-GPU system, the model weights live in VRAM and prompt embeddings have to be copied across PCIe. That copy is small per request but adds latency and complexity (pinned host memory, async copy queues, double buffering).

GB10's unified memory eliminates the copy. The CPU and GPU see the same physical pages. `mmap()`-ing a GGUF file makes its contents directly addressable by CUDA kernels with no staging step. This is the hardware feature that makes a 30B+ class model feel responsive on a "personal" machine.

The trade-off: total memory bandwidth (~273 GB/s on GB10) is lower than discrete GPU HBM (~3 TB/s on H100). For dense models, decode is bandwidth-bound, so dense throughput is lower than what an H100 delivers. **MoE models** activate only a fraction of parameters per token and are correspondingly faster. This is why the benchmarks in the README show Qwen3 30B-A3B (MoE) at ~89 t/s while Qwen3 32B Dense sits at ~10 t/s on the same hardware.

## Concurrency model

`llama-server` uses **slots**: a fixed number of independent KV caches sized at startup. With `--parallel 4`, four requests can be in flight simultaneously, each with its own context. A fifth request queues until a slot frees.

KV cache memory per slot scales as `2 * num_layers * ctx_size * (head_dim) * (bytes_per_element)`. For a 31B model with 64 layers, 16k context, FP16 KV cache, this is on the order of 4 GB per slot. Four slots cost ~16 GB. This is why context size and parallelism multiply.

## Failure modes

The most common production failure is **OOM at request time** (not at startup), triggered by the KV cache hitting the unified memory ceiling once enough concurrent slots are occupied. Mitigations:

- Lower `--ctx-size` to a realistic upper bound for actual prompts.
- Lower `--parallel`.
- Move to a smaller quant (Q4_K_M, IQ3_XS).
- Use KV cache quantization (`--cache-type-k q8_0 --cache-type-v q8_0`) at a small quality cost.

Process supervision via `systemd` covers crashes. The `Restart=on-failure` directive brings the service back within `RestartSec=10s`, which is acceptable for a developer assistant but should be tuned for stricter SLOs.
