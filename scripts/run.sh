#!/bin/bash
./build/bin/llama-server \
  -m /models/gemma-4-31B-it-NVFP4-turbo/gemma-4-31B-it-NVFP4-turbo-NVFP4.gguf \
  -c 8192 \
  --n-gpu-layers 100 \
  --host 0.0.0.0 \
  --port 8080
