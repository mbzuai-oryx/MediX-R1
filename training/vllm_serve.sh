#!/bin/bash
# vllm_serve.sh - Start a vLLM OpenAI-compatible inference server
#
# Serves a model using vLLM with tensor parallelism across multiple GPUs,
# exposing an OpenAI-compatible API (chat completions, completions, etc.).
#
# Usage:
#   bash vllm_serve.sh
#
# Before running, configure the variables below:
#   MODEL        - HuggingFace model ID or local path to merged model
#   GPUS         - Comma-separated GPU IDs (e.g., "0,1")
#   PORT         - Port for the API server (default: 8080)
#   TP_SIZE      - Tensor parallel size, should match number of GPUs
#   MAX_LEN      - Maximum sequence length
#   GPU_MEM_UTIL - Fraction of GPU memory to use (0.0-1.0)
#
# Test the server with curl:
#   curl -X POST "http://localhost:${PORT}/v1/chat/completions" \
#     -H "Content-Type: application/json" \
#     --data '{
#       "model": "'"${MODEL}"'",
#       "messages": [{"role": "user", "content": "Hello!"}]
#     }'

MODEL="Qwen/Qwen3-4B-Instruct-2507"
GPUS="0,1"
PORT=8085
TP_SIZE=2
MAX_LEN=2048
GPU_MEM_UTIL=0.97

export CUDA_VISIBLE_DEVICES=$GPUS

vllm serve "$MODEL" \
    --port $PORT \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --tensor-parallel-size $TP_SIZE \
    --max-model-len $MAX_LEN
