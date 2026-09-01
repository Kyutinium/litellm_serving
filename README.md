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
| `none` | thinking 서명은 숨기되, GLM/vLLM처럼 `content` 없이 `reasoning_content`만 오는 응답은 빈 응답 방지를 위해 일반 텍스트로 승격 **(기본값)** | `실제 응답` |

> **참고**: 이 설정은 주로 Anthropic Messages API 스트리밍 엔드포인트(`/v1/messages`)에 적용됩니다.
> OpenAI 형식(`/v1/chat/completions`)에서는 `litellm_config.yaml`의 `merge_reasoning_content_in_choices: true` 설정이 reasoning 병합을 제어합니다.

> **GLM/vLLM 주의**: 일부 reasoning 모델은 `content: null` 상태로 실제 응답을 `reasoning_content`에만 담아 스트리밍할 수 있습니다.
> 이 경우 `none` 모드에서도 해당 텍스트를 `text_delta`로 승격하여 Claude Code subagent의 `response_text`가 빈 문자열이 되는 문제를 방지합니다.

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
├── sanitizer/             # Anthropic-facing 리버스 프록시 (LiteLLM 어댑터 버그 우회)
├── Dockerfile
├── docker-compose.yml
├── Dockerfile.dev         # sanitizer + LiteLLM 동시 기동
├── docker-compose-dev.yml
├── entrypoint.sh          # sanitizer/LiteLLM 이중 프로세스 엔트리포인트
├── TROUBLESHOOTING.md     # 트러블슈팅 가이드
└── DEBUG_REPORT.md        # SDK 스트리밍 디버그 리포트
```

---

## Sanitizer (Anthropic 리버스 프록시)

`sanitizer/`는 LiteLLM 프록시 앞단에 배치되는 얇은 FastAPI 리버스 프록시로,
Anthropic `/v1/messages` 엔드포인트를 노출하면서 LiteLLM 자체 Anthropic 어댑터의
확인된 버그(잘못된 SSE, 잘리거나 zero-payload인 `input_json_delta`, reasoning 콘텐츠
드롭)를 우회합니다. `SANITIZER_USE_OPENAI_BRIDGE=true`면 upstream의 정상적인
`/v1/chat/completions` 라우트를 직접 호출해 in-process로 변환하며,
`tool_result` 안의 이미지를 직후 user 메시지로 재배치해 vision 백엔드에서
이미지가 유실되던 문제(gateway issue #140)도 함께 해결합니다.

자세한 내용과 실행/테스트 방법은 [`sanitizer/README.md`](sanitizer/README.md) 참조.

## Claude Code 모델 디스커버리

Claude Code(2.1.129+)는 `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1` 이면 시작 시
`ANTHROPIC_BASE_URL` 의 `GET /v1/models?limit=1000` 을 읽어 `/model` 픽커를 채운다.
sanitizer 는 `POST /v1/messages` 외 전부를 상류로 릴레이하므로 이 요청은 이미
LiteLLM 까지 도달한다 — **막는 것은 모델 이름뿐이다.** Claude Code 는 id 에
`claude` 또는 `anthropic` 이 **포함된** 모델만 남긴다(접두사가 아니라 부분 문자열
— 앵커 없는 `/(claude|anthropic)/i`, 2.1.251 바이너리에서 확인).

기본 `litellm_config.yaml` 의 등록 이름에는 그 문자열이 없어서 0개가 남는다.
해결은 별칭 추가다: [`litellm_config.claude-code-discovery.example.yaml`](./litellm_config.claude-code-discovery.example.yaml)
이 기존 12개 이름을 그대로 두고 실모델 7종에 `claude-*` 별칭을 추가한 전체 예시다.

```bash
# LiteLLM 을 example config 로 실행한 뒤:
export ANTHROPIC_BASE_URL=http://<sanitizer-host>:3996
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
claude   # /model 픽커에 claude-* 7종이 "From gateway" 로 뜬다
```

이 example 은 실제 구성(LiteLLM proxy + sanitizer + claude-code 2.1.251)으로
E2E 검증됐다: CLI 디버그 로그에 `[gatewayDiscovery] cached 7 models` 가 기록되고
7종 전부 픽커 캐시에 실린다. **선택 후 실제 추론 라우팅은 백엔드가 살아 있는
배포에서 한 턴 돌려 확인할 것** (검증 환경에는 vLLM 백엔드/DB 가 없었다).

### 컨텍스트 윈도 주의

Claude Code 는 자기가 모르는 모델 id 의 컨텍스트 윈도를 **200k 로 가정**하고
auto-compact 를 그 기준으로 돌린다. 32k 모델이면 상류가 먼저 거부할 수 있다.

- **`CLAUDE_CODE_MAX_CONTEXT_TOKENS=<실제 윈도>`** — 미인식 id 에 적용되는 선언.
  단 전역 env 라 윈도가 다른 모델을 오가며 쓰기엔 불편하다.
- `modelOverrides` 설정은 윈도 지정이 아니라 **인식되는 Anthropic 모델 id → 게이트웨이
  별칭 문자열** 매핑이다. 인식되는 id 로 매핑하면 그 모델의 알려진 윈도를 따른다.
- `CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1` 은 API 가 too-long 을
  돌려줄 때까지 기다렸다 압축하는 사후 방식인데, **게이트웨이가 에러 문구를
  재작성하면 Claude Code 가 그 에러를 인식하지 못한다** — sanitizer 가 응답을
  정규화하는 이 구성에서는 신뢰하지 말 것.
