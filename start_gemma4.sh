#!/bin/bash
# Gemma-4-31B-IT vLLM Serving Script
#
# Prerequisites:
#   pip install vllm>=0.8.5
#   Model checkpoint at /home/kyvhyvn.shim/to/public/checkpoints/gemma/gemma-4-31B-it
#
# Usage:
#   bash start_gemma4.sh                    # 기본값 (GPU 2장, port 8090)
#   TENSOR_PARALLEL=4 bash start_gemma4.sh  # GPU 4장
#   PORT=8091 bash start_gemma4.sh          # 포트 변경

set -euo pipefail

MODEL_ID="${MODEL_PATH:-/home/kyvhyvn.shim/to/public/checkpoints/gemma/gemma-4-31B-it}"
PORT="${PORT:-8090}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DTYPE="${DTYPE:-bfloat16}"

echo "============================================"
echo " Gemma-4-31B-IT vLLM Server"
echo "============================================"
echo " Model:         ${MODEL_ID}"
echo " Port:          ${PORT}"
echo " TP:            ${TENSOR_PARALLEL}"
echo " Max Model Len: ${MAX_MODEL_LEN}"
echo " GPU Mem Util:  ${GPU_MEMORY_UTILIZATION}"
echo " Dtype:         ${DTYPE}"
echo "============================================"

exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --tensor-parallel-size "${TENSOR_PARALLEL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --dtype "${DTYPE}" \
    --trust-remote-code \
    --served-model-name "gemma-4-31b-it" \
    --api-key EMPTY
