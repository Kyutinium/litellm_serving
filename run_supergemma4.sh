#!/bin/bash
# SuperGemma4-26B-Uncensored GGUF v2 llama.cpp 서버 실행
#
# Usage:
#   bash run_supergemma4.sh                                # 기본값 (GPU all, port 8091)
#   PORT=8092 bash run_supergemma4.sh                      # 포트 변경
#   GPU_LAYERS=99 bash run_supergemma4.sh                  # GPU 레이어 수 변경
#   MODEL_FILE=other.gguf bash run_supergemma4.sh          # 다른 GGUF 파일 지정
#   CTX_SIZE=16384 bash run_supergemma4.sh                 # 컨텍스트 크기 변경
#   GPU_IDS='"device=0"' bash run_supergemma4.sh           # 특정 GPU 지정

set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-supergemma4-llamacpp}"
IMAGE="${IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}"
MODEL_DIR="${MODEL_DIR:-/shared/checkpoints/supergemma4-26b-uncensored-gguf-v2}"
MODEL_FILE="${MODEL_FILE:-}"
PORT="${PORT:-8091}"
GPU_LAYERS="${GPU_LAYERS:-99}"
CTX_SIZE="${CTX_SIZE:-8192}"
THREADS="${THREADS:-8}"
PARALLEL="${PARALLEL:-4}"
GPU_IDS="${GPU_IDS:-all}"

# ── GGUF 파일 결정 ──────────────────────────────────────
if [ -z "${MODEL_FILE}" ]; then
    MODEL_FILE=$(find "${MODEL_DIR}" -maxdepth 1 -name "*.gguf" -size +1M -print -quit 2>/dev/null)
    if [ -z "${MODEL_FILE}" ]; then
        echo "[ERROR] No .gguf file found in ${MODEL_DIR}"
        echo "        Run setup_supergemma4.sh first."
        exit 1
    fi
fi

# 상대 경로 → 절대 경로
if [[ "${MODEL_FILE}" != /* ]]; then
    MODEL_FILE="${MODEL_DIR}/${MODEL_FILE}"
fi

if [ ! -f "${MODEL_FILE}" ]; then
    echo "[ERROR] Model file not found: ${MODEL_FILE}"
    exit 1
fi

MODEL_FILENAME=$(basename "${MODEL_FILE}")
MODEL_PARENT=$(dirname "${MODEL_FILE}")

# ── 기존 컨테이너 정리 ──────────────────────────────────
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removing existing container: ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1
fi

echo "============================================"
echo " SuperGemma4-26B-Uncensored GGUF v2"
echo " (llama.cpp server)"
echo "============================================"
echo " Container:     ${CONTAINER_NAME}"
echo " Model:         ${MODEL_FILE}"
echo " Port:          ${PORT}"
echo " GPU Layers:    ${GPU_LAYERS}"
echo " Context Size:  ${CTX_SIZE}"
echo " Threads:       ${THREADS}"
echo " Parallel:      ${PARALLEL}"
echo " GPUs:          ${GPU_IDS}"
echo "============================================"

exec docker run \
    --name "${CONTAINER_NAME}" \
    --gpus "${GPU_IDS}" \
    -v "${MODEL_PARENT}:/models:ro" \
    -p "${PORT}:8080" \
    --restart unless-stopped \
    --health-cmd "curl -f http://localhost:8080/health || exit 1" \
    --health-interval 30s \
    --health-timeout 10s \
    --health-retries 5 \
    --health-start-period 120s \
    "${IMAGE}" \
    -m "/models/${MODEL_FILENAME}" \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl "${GPU_LAYERS}" \
    -c "${CTX_SIZE}" \
    -t "${THREADS}" \
    --parallel "${PARALLEL}" \
    --flash-attn \
    --metrics
