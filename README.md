# litellm_serving

LiteLLM 프록시를 통해 SGLang / vLLM 등 비-Anthropic 백엔드 모델을 OpenAI 및 Anthropic 호환 API로 서빙합니다.

reasoning 모델(GLM, DeepSeek, Qwen 등)의 **thinking(추론) 출력**을 환경 변수 하나로 제어할 수 있습니다.

---

## 환경 변수

### `THINK_OUTPUT_MODE`

모델의 thinking/reasoning 콘텐츠 출력 방식을 제어합니다.

| 값 | 동작 | 출력 예시 |
|---|---|---|
| `default` | LiteLLM 기본 동작 (thinking_delta 그대로 전달) | *(litellm이 처리하는 대로)* |
| `think_tag` | `<think>...</think>` 태그로 감싸서 일반 텍스트로 출력 | `<think>\n추론 내용...\n</think>\n\n실제 응답` |
| `text` | 태그 없이 일반 텍스트로 출력 | `추론 내용...\n\n실제 응답` |
| `none` | thinking 콘텐츠를 아예 출력하지 않음 **(기본값)** | `실제 응답` |

> **참고**: 이 설정은 주로 Anthropic Messages API 스트리밍 엔드포인트(`/v1/messages`)에 적용됩니다.
> OpenAI 형식(`/v1/chat/completions`)에서는 `litellm_config.yaml`의 `merge_reasoning_content_in_choices: true` 설정이 reasoning 병합을 제어합니다.

### 기타 환경 변수

| 변수 | 설명 | 기본값 |
|---|---|---|
| `LITELLM_MASTER_KEY` | LiteLLM 프록시 인증 키 | `sk-1234` |
| `SSL_CERT_FILE` | SSL 인증서 경로 | *(시스템 기본)* |
| `NO_PROXY` / `no_proxy` | 프록시 우회 대상 | `localhost,127.0.0.1` |

---

## 사용법

### Docker 사용

```bash
# 1. 빌드 및 실행 (기본: thinking 출력 안 함)
docker compose up -d --build

# 2. thinking을 <think> 태그로 감싸서 출력
THINK_OUTPUT_MODE=think_tag docker compose up -d --build

# 3. thinking을 일반 텍스트로 출력
THINK_OUTPUT_MODE=text docker compose up -d --build

# 4. litellm 기본 동작
THINK_OUTPUT_MODE=default docker compose up -d --build
```

또는 `docker-compose.yml`에서 직접 수정:

```yaml
environment:
  - THINK_OUTPUT_MODE=think_tag   # default | think_tag | text | none
```

### Docker 미사용

```bash
# 1. 의존성 설치
pip install litellm

# 2. 환경 변수 설정
export LITELLM_WORKER_STARTUP_HOOKS=strip_thinking:apply_patch
export PYTHONPATH=.
export THINK_OUTPUT_MODE=none  # default | think_tag | text | none

# 3. 프록시 실행
litellm --config litellm_config.yaml --port 3999 --host 0.0.0.0
```

모드 변경 시 환경 변수만 바꾸면 됩니다:

```bash
# <think> 태그 모드
export THINK_OUTPUT_MODE=think_tag
litellm --config litellm_config.yaml --port 3999 --host 0.0.0.0

# 일반 텍스트 모드
export THINK_OUTPUT_MODE=text
litellm --config litellm_config.yaml --port 3999 --host 0.0.0.0
```

---

## 모델 설정

`litellm_config.yaml`에서 모델을 추가/수정합니다:

```yaml
model_list:
  - model_name: glm-5-fp8
    litellm_params:
      model: hosted_vllm/glm-5-fp8
      api_base: http://localhost:8088/v1
      api_key: EMPTY
      max_tokens: 131072
      merge_reasoning_content_in_choices: true
```

- `merge_reasoning_content_in_choices: true` — OpenAI 형식(`/v1/chat/completions`) 응답에서 reasoning_content를 content에 병합
- `api_base` — 백엔드 모델 서버 주소

---

## 프로젝트 구조

```
├── litellm_config.yaml    # 모델 및 LiteLLM 설정
├── strip_thinking.py      # thinking 출력 제어 (THINK_OUTPUT_MODE)
├── Dockerfile
├── docker-compose.yml
├── TROUBLESHOOTING.md     # 트러블슈팅 가이드
└── DEBUG_REPORT.md        # SDK 스트리밍 디버그 리포트
```
