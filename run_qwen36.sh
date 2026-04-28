#!/bin/bash
# Qwen3.6-27B vLLM 서버 실행
#
# Usage:
#   bash run_qwen36.sh                              # 기본값 (GPU 2,3,4,5, TP=4, port 8092)
#   PORT=8093 bash run_qwen36.sh                    # 포트 변경
#   TENSOR_PARALLEL=2 bash run_qwen36.sh            # TP 크기 변경
#   GPU_IDS='"device=0,1,2,3"' bash run_qwen36.sh   # GPU 변경

set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-qwen36-vllm}"
IMAGE="${IMAGE:-vllm/vllm-openai:latest}"
MODEL_PATH="${MODEL_PATH:-/shared/checkpoints/to_supercom/Qwen/Qwen3.6-27B}"
PORT="${PORT:-8092}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DTYPE="${DTYPE:-bfloat16}"
GPU_IDS="${GPU_IDS:-\"device=2,3,4,5\"}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"

# Docker bind mount은 symlink 경로에서 'mkdir ... file exists' 로 실패할 수 있어
# 실경로(canonical path)로 해석해서 넘긴다.
if [ ! -e "${MODEL_PATH}" ]; then
    echo "[ERROR] Model path not found: ${MODEL_PATH}" >&2
    exit 1
fi
MODEL_PATH_REAL="$(readlink -f "${MODEL_PATH}")"

# 기존 컨테이너 정리
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removing existing container: ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1
fi

echo "============================================"
echo " Qwen3.6-27B vLLM Server"
echo "============================================"
echo " Container:     ${CONTAINER_NAME}"
echo " Model:         ${MODEL_PATH}"
echo " Model (real):  ${MODEL_PATH_REAL}"
echo " Port:          ${PORT}"
echo " TP:            ${TENSOR_PARALLEL}"
echo " Max Model Len: ${MAX_MODEL_LEN}"
echo " GPU Mem Util:  ${GPU_MEMORY_UTILIZATION}"
echo " Dtype:         ${DTYPE}"
echo " GPUs:          ${GPU_IDS}"
echo " Tool Parser:   ${TOOL_CALL_PARSER}"
echo "============================================"

exec docker run \
    --name "${CONTAINER_NAME}" \
    --gpus "${GPU_IDS}" \
    --ipc=host \
    -v "${MODEL_PATH_REAL}:/model:ro" \
    -p "${PORT}:8000" \
    --restart unless-stopped \
    "${IMAGE}" \
    --model /model \
    --tensor-parallel-size "${TENSOR_PARALLEL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --dtype "${DTYPE}" \
    --trust-remote-code \
    --served-model-name "qwen3.6-27b" \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_CALL_PARSER}" \
    --api-key EMPTY
