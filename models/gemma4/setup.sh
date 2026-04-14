#!/bin/bash
# Gemma-4-31B-IT 환경 셋업
# Docker 이미지 pull + 기존 컨테이너 정리

set -euo pipefail

IMAGE="${IMAGE:-vllm/vllm-openai:gemma4}"
CONTAINER_NAME="${CONTAINER_NAME:-gemma4-vllm}"
MODEL_PATH="${MODEL_PATH:-/home/kyvhyvn.shim/to/public/checkpoints/gemma/gemma-4-31B-it}"

echo "=== Setup: Gemma-4-31B-IT ==="

# 1. 모델 체크포인트 확인
if [ ! -d "${MODEL_PATH}" ]; then
    echo "[ERROR] Model checkpoint not found: ${MODEL_PATH}"
    exit 1
fi
echo "[OK] Model checkpoint: ${MODEL_PATH}"

# 2. Docker 이미지 pull
echo "[*] Pulling image: ${IMAGE}"
docker pull "${IMAGE}"
echo "[OK] Image ready"

# 3. 기존 컨테이너 정리
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[*] Removing existing container: ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1
fi

echo ""
echo "=== Setup complete. Run with: bash run_gemma4.sh ==="
