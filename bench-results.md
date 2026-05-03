# Benchmark results

This file collects raw `llama-bench` output captured on the reference DGX Spark machine. New runs are appended automatically by `scripts/bench.sh`.

Hardware baseline for all entries unless noted otherwise:

- **Platform:** NVIDIA DGX Spark
- **GPU:** NVIDIA GB10 (Blackwell, compute capability 12.1)
- **Memory:** 128 GB LPDDR5x unified, ~273 GB/s
- **OS:** DGX OS (Ubuntu 24.04 LTS, aarch64)
- **CUDA:** 13.0
- **Driver:** 580.x

---

## gemma-4-31B-it-NVFP4-turbo-NVFP4.gguf

- **Date:** 2026-05-03
- **llama.cpp build:** `550d684bd (8902)`
- **Args:** `-ngl 100`
- **Result:** prefill 542.92 ± 2.27 t/s, decode 11.56 ± 0.00 t/s
- **Decode efficiency:** ~76% of theoretical bandwidth ceiling (~15.2 t/s)

<details><summary>Raw output</summary>

```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 124610 MiB):
  Device 0: NVIDIA GB10, compute capability 12.1, VMM: yes, VRAM: 124610 MiB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 31B NVFP4               |  17.97 GiB |    30.70 B | CUDA       | 100 |           pp512 |        542.92 ± 2.27 |
| gemma4 31B NVFP4               |  17.97 GiB |    30.70 B | CUDA       | 100 |           tg128 |         11.56 ± 0.00 |
build: 550d684bd (8902)
```

</details>
