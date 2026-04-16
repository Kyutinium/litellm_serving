#!/bin/bash
# SuperGemma4-26B-Uncensored GGUF v2 환경 셋업
# Docker 이미지 pull + 모델 다운로드 + 검증

set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}"
CONTAINER_NAME="${CONTAINER_NAME:-supergemma4-llamacpp}"
MODEL_DIR="${MODEL_DIR:-/home/kyvhyvn.shim/to/public/checkpoints/supergemma4-26b-uncensored-gguf-v2}"
HF_REPO="https://huggingface.co/Jiunsong/supergemma4-26b-uncensored-gguf-v2"

echo "=== Setup: SuperGemma4-26B-Uncensored GGUF v2 ==="

# 1. git-lfs 확인
if ! command -v git-lfs &>/dev/null; then
    echo "[ERROR] git-lfs is not installed. Install with: apt install git-lfs"
    exit 1
fi
echo "[OK] git-lfs found"

# 2. 모델 다운로드
if [ ! -d "${MODEL_DIR}" ]; then
    echo "[*] Cloning model from: ${HF_REPO}"
    git clone "${HF_REPO}" "${MODEL_DIR}"
else
    echo "[OK] Model directory exists: ${MODEL_DIR}"
fi

# 3. GGUF 파일 확인 (lfs pull 자동 시도)
GGUF_FILES=$(find "${MODEL_DIR}" -maxdepth 1 -name "*.gguf" 2>/dev/null)
if [ -z "${GGUF_FILES}" ]; then
    echo "[WARN] No .gguf files found. Attempting git lfs pull..."
    cd "${MODEL_DIR}" && git lfs pull
    GGUF_FILES=$(find "${MODEL_DIR}" -maxdepth 1 -name "*.gguf" 2>/dev/null)
    if [ -z "${GGUF_FILES}" ]; then
        echo "[ERROR] Still no .gguf files after lfs pull."
        exit 1
    fi
fi
echo "[OK] GGUF files:"
find "${MODEL_DIR}" -maxdepth 1 -name "*.gguf" -exec ls -lh {} \;

# 4. Docker 이미지 pull
echo "[*] Pulling image: ${IMAGE}"
docker pull "${IMAGE}"
echo "[OK] Image ready"

# 5. 기존 컨테이너 정리
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[*] Removing existing container: ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1
fi

echo ""
echo "=== Setup complete. Run with: bash run_supergemma4.sh ==="
