#!/bin/bash
# Gemma-4-31B-IT vLLM 서버 실행
#
# Usage:
#   bash run_gemma4.sh                              # 기본값 (GPU 2장, port 8090)
#   TENSOR_PARALLEL=4 bash run_gemma4.sh            # GPU 4장
#   PORT=8091 bash run_gemma4.sh                    # 포트 변경
#   GPU_IDS='"device=0,1"' bash run_gemma4.sh       # 특정 GPU 지정

set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-gemma4-vllm}"
IMAGE="${IMAGE:-vllm/vllm-openai:gemma4}"
MODEL_PATH="${MODEL_PATH:-/home/kyvhyvn.shim/to/public/checkpoints/gemma/gemma-4-31B-it}"
PORT="${PORT:-8090}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DTYPE="${DTYPE:-bfloat16}"
GPU_IDS="${GPU_IDS:-all}"

# 기존 컨테이너 정리
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removing existing container: ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1
fi

echo "============================================"
echo " Gemma-4-31B-IT vLLM Server"
echo "============================================"
echo " Container:     ${CONTAINER_NAME}"
echo " Model:         ${MODEL_PATH}"
echo " Port:          ${PORT}"
echo " TP:            ${TENSOR_PARALLEL}"
echo " Max Model Len: ${MAX_MODEL_LEN}"
echo " GPU Mem Util:  ${GPU_MEMORY_UTILIZATION}"
echo " Dtype:         ${DTYPE}"
echo " GPUs:          ${GPU_IDS}"
echo "============================================"

exec docker run \
    --name "${CONTAINER_NAME}" \
    --gpus "${GPU_IDS}" \
    --ipc=host \
    -e PYTHONWARNINGS="ignore::FutureWarning" \
    -v "${MODEL_PATH}:/model" \
    -p "${PORT}:8000" \
    --restart unless-stopped \
    "${IMAGE}" \
    --model /model \
    --tensor-parallel-size "${TENSOR_PARALLEL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --dtype "${DTYPE}" \
    --trust-remote-code \
    --served-model-name "gemma-4-31b-it" \
    --limit-mm-per-prompt '{"image": 0, "audio": 0}' \
    --api-key EMPTY
