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
| `reasoning_fallback` | GLM/vLLM처럼 `content` 없이 `reasoning_content`만 오는 응답을 빈 응답 방지를 위해 일반 텍스트로 승격 | `실제 응답` |
| `none` | thinking/reasoning 콘텐츠를 출력하지 않음 **(기본값, 정상 동작 영향 최소화)** | `실제 응답` |

> **참고**: 이 설정은 주로 Anthropic Messages API 스트리밍 엔드포인트(`/v1/messages`)에 적용됩니다.
> OpenAI 형식(`/v1/chat/completions`)에서는 `litellm_config.yaml`의 `merge_reasoning_content_in_choices: true` 설정이 reasoning 병합을 제어합니다.

> **GLM/vLLM 주의**: 일부 reasoning 모델은 `content: null` 상태로 실제 응답을 `reasoning_content`에만 담아 스트리밍할 수 있습니다.
> 이 경우 `reasoning_fallback` 모드를 사용하면 해당 텍스트를 `text_delta`로 승격하여 Claude Code subagent의 `response_text`가 빈 문자열이 되는 문제를 방지합니다.
> 정상 요청에서 thinking 노출/동작 변경을 피하려면 기본 `none` 모드를 유지하세요. `reasoning_fallback`은 해당 모델이 실제 답변을 `reasoning_content`에만 담는 것이 확인된 경우에만 사용합니다.
> 단, 백엔드가 실제로 텍스트/추론 델타를 전혀 보내지 않거나 `tool_calls`만 보내고 최종 텍스트를 생성하지 않는 경우에는 이 패치가 없는 답변을 새로 만들 수 없으므로 원본 vLLM/LiteLLM 스트림을 확인해야 합니다.

> **Docker 기본값**: 이 저장소의 Dockerfile/docker-compose는 GLM/vLLM의 `reasoning_content`-only 응답을 실제 답변으로 보존하기 위해 `THINK_OUTPUT_MODE=reasoning_fallback`을 기본 설정합니다.
> Python 코드 자체의 환경 변수 기본값은 여전히 `none`이므로, 다른 모델/프록시에서 thinking 노출을 피하려면 명시적으로 `THINK_OUTPUT_MODE=none`을 사용하세요.

> **완전 우회 옵션**: `THINK_OUTPUT_MODE=default`는 스트리밍 adapter patch만 끄고 입력 메시지의 thinking block 제거 callback은 계속 등록합니다.
> `strip_thinking.py`를 아예 거치지 않는 동작 확인이 필요하면 `STRIP_THINKING_ENABLED=false`를 설정하세요. 이 경우 callback 등록과 streaming patch가 모두 생략됩니다.

### 기타 환경 변수

| 변수 | 설명 | 기본값 |
|---|---|---|
| `LITELLM_MASTER_KEY` | LiteLLM 프록시 인증 키 | `sk-1234` |
| `SSL_CERT_FILE` | SSL 인증서 경로 | *(시스템 기본)* |
| `NO_PROXY` / `no_proxy` | 프록시 우회 대상 | `localhost,127.0.0.1` |
| `STRIP_THINKING_ENABLED` | `false`/`0`/`no`/`off`로 설정하면 `strip_thinking.py`의 입력 stripping과 스트리밍 adapter patch를 모두 우회 | `true` |

---

## 사용법

### Docker 사용

```bash
# 1. 빌드 및 실행 (Docker 기본: GLM/vLLM reasoning-only 응답 보호)
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
  - THINK_OUTPUT_MODE=reasoning_fallback   # default | think_tag | text | reasoning_fallback | none
  - STRIP_THINKING_ENABLED=true            # false면 strip_thinking 전체 우회
```

### Docker 미사용

```bash
# 1. 의존성 설치
pip install litellm

# 2. 환경 변수 설정
export LITELLM_WORKER_STARTUP_HOOKS=strip_thinking:apply_patch
export PYTHONPATH=.
export THINK_OUTPUT_MODE=none  # default | think_tag | text | reasoning_fallback | none
export STRIP_THINKING_ENABLED=true  # false로 설정하면 strip_thinking 전체 우회

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

# strip_thinking 전체 우회
export STRIP_THINKING_ENABLED=false
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

- `model_name` — 클라이언트가 요청 시 사용하는 모델 이름
- `model` — litellm이 사용하는 내부 모델 식별자 (prefix로 백엔드 타입 지정: `hosted_vllm/`, `openai/` 등)
- `api_base` — 모델이 서빙되고 있는 백엔드 서버 URL. **실제 모델 서버 주소에 맞게 반드시 수정해야 합니다**
- `max_tokens` — 최대 출력 토큰 수
- `merge_reasoning_content_in_choices: true` — OpenAI 형식(`/v1/chat/completions`) 응답에서 reasoning_content를 content에 병합

> **주의**: `api_base`는 실제로 모델이 서빙되고 있는 서버의 주소와 포트로 변경해야 합니다.
> 예를 들어 SGLang이 `http://192.168.1.100:8088/v1`에서 실행 중이라면 해당 URL로 설정하세요.

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
