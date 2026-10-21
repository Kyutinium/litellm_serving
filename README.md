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

컨테이너 하나에 두 프로세스가 뜹니다: **LiteLLM(:3999)** 과 그 앞의 **sanitizer(:3996)**.
클라이언트(oh-my-gateway의 `ANTHROPIC_BASE_URL`)는 **:3996** 을 보게 하세요. 모델별로 받는
reasoning-effort 레벨을 sanitizer가 배워 `GET /v1/models`의 `effort_levels`로 올리고, 받지
않는 레벨은 vLLM에 닿기 전에 clamp하는 곳이 거기입니다 — :3999를 직접 보면 둘 다 없습니다.
자세한 동작은 [`sanitizer/README.md`](sanitizer/README.md)의 "Reasoning effort".

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

> **Sanitizer 브리지(`SANITIZER_USE_OPENAI_BRIDGE=true`)를 앞에 두는 모델에는 이 옵션을 켜지 마세요.**
> 브리지는 upstream의 `reasoning_content`를 Anthropic `thinking` 블록으로 옮기는데, 이 옵션이 켜져
> 있으면 LiteLLM이 그 전에 reasoning을 `content` 안의 리터럴 `<think>…</think>` 텍스트로 합쳐 버려서
> Anthropic 클라이언트(Claude Agent SDK / Claude Code)에는 thinking 블록 대신 `<think>` 태그가 섞인
> 본문 텍스트가 도착합니다 (실측: gateway → sanitizer → LiteLLM 경로에서 `text_delta: "<think>…</think>답변"`).
> ChatDRAGON은 리터럴 `<think>` 태그를 접어 주지만 다른 클라이언트는 태그를 그대로 보여 줍니다.
> 브리지 없이 LiteLLM의 `/v1/messages` 어댑터를 직접 쓰는 경우에만 의미가 있는 옵션입니다.

> **주의**: `api_base`는 실제로 모델이 서빙되고 있는 서버의 주소와 포트로 변경해야 합니다.
> 예를 들어 SGLang이 `http://192.168.1.100:8088/v1`에서 실행 중이라면 해당 URL로 설정하세요.

---

## 프로젝트 구조

```
├── litellm_config.yaml    # 모델 및 LiteLLM 설정
├── strip_thinking.py      # thinking 출력 제어 (THINK_OUTPUT_MODE)
├── prompt_capture.py      # 모델이 실제로 받는 요청 기록 (PROMPT_CAPTURE, 기본 꺼짐)
├── sanitizer/             # Anthropic-facing 리버스 프록시 (LiteLLM 어댑터 버그 우회)
├── balancer/              # 같은 모델의 vLLM 여러 대를 실제 부하 기준으로 분배 (BALANCER_POOLS)
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

프로덕션 `Dockerfile`과 `Dockerfile.dev` 모두 sanitizer를 LiteLLM 앞(:3996)에 **브리지를 켠 채**
(`SANITIZER_USE_OPENAI_BRIDGE=true`) 함께 띄웁니다(`entrypoint.sh`). 모델별 reasoning-effort 어휘
학습·게시(`effort_levels`)·clamp는 이 프로세스의 일이고, 게시되는 레벨은 `/v1/chat/completions`의
`reasoning_effort`로 배운 것이므로 `/v1/messages`의 `output_config.effort`를 같은 필드·같은 경로에
싣는 브리지가 켜져 있어야 그 레벨이 실제 요청 경로를 설명합니다. 브리지가 꺼지면 `/v1/messages`는
LiteLLM 자체 Anthropic 어댑터로 그대로 중계되어 변환도 clamp도 타지 않습니다. gateway가 :3999를
직접 보면 ChatDRAGON까지 레벨이 올라가지 않습니다. 이 짝은 `sanitizer/tests/test_production_topology.py`가
고정합니다.

자세한 내용과 실행/테스트 방법은 [`sanitizer/README.md`](sanitizer/README.md) 참조.

## vLLM 여러 대에 부하 기준으로 나누기 (balancer)

같은 모델을 vLLM 여러 대로 띄웠을 때 LiteLLM `routing_strategy: least-busy`는 동시에 들어온 요청을
**한 대에 몰아 줍니다**(issue #31). LiteLLM 1.106 코드 기준으로 네 가지 이유가 있습니다.
- 처리 중 요청 수가 같으면 늘 목록의 첫 번째가 뽑힙니다.
- 서버를 고른 **뒤에야** 처리 중 숫자를 올려서, 동시에 들어온 요청이 같은 숫자(0)를 봅니다.
- Redis가 없으면 그 숫자가 워커마다 따로 있습니다.
- 긴 컨텍스트 요청과 짧은 요청을 똑같이 1로 셉니다.

`balancer/`는 LiteLLM과 vLLM 사이에서 이렇게 고릅니다.
- 각 vLLM의 `/metrics`(`vllm:num_requests_waiting`, `vllm:num_requests_running`, KV 캐시 사용률
  `vllm:kv_cache_usage_perc` 또는 예전 이름 `vllm:gpu_cache_usage_perc`)를 주기적으로 읽어 점수를 매깁니다.
- 고르는 **순간** 그 서버의 숫자를 올리고, 응답이 끝나거나 끊기면 내립니다.
- 점수가 같으면 무작위로 고릅니다.
- 연결이 안 되는 서버(거부되거나 연결 제한 시간 안에 답이 없는 서버)는 잠시 빼고 다음 서버로 보냅니다.

LiteLLM은 그 모델의 주소 하나(balancer)만 봅니다.

**켜는 법**
1. compose에 `BALANCER_POOLS`를 넣습니다(풀 이름 → vLLM 주소 목록). 값이 있을 때만 balancer가
   `127.0.0.1:${BALANCER_PORT}`(기본 3997)에 뜨고, 잘못된 값이면 시작하지 않습니다.
   ```yaml
   - 'BALANCER_POOLS={"glm-5.1-fp8":["http://10.0.0.1:8089","http://10.0.0.2:8089"]}'
   ```
2. `litellm_config.yaml`에서 그 모델의 `api_base`를 balancer로 바꿉니다.
   ```yaml
   - model_name: glm-5.1-fp8
     litellm_params:
       model: hosted_vllm/glm-5.1-fp8
       api_base: http://localhost:3997/glm-5.1-fp8/v1
       api_key: EMPTY
   ```
   이 모델에 LiteLLM `routing_strategy`는 필요 없습니다(LiteLLM이 보는 deployment가 하나).

| 변수 | 기본값 | 설명 |
|---|---|---|
| `BALANCER_POOLS` | *(비어 있음 = 끔)* | `{"풀 이름": ["http://host:port", …]}`. 주소 끝의 `/v1`은 떼고 씁니다 |
| `BALANCER_PORT` | `3997` | balancer 포트(컨테이너 안 루프백) |
| `BALANCER_SCRAPE_INTERVAL` | `1.0` | `/metrics`를 읽는 주기(초) |
| `BALANCER_SCRAPE_TIMEOUT` | `0.8` | `/metrics` 한 번의 제한 시간(초). 넘으면 그 서버는 우리 쪽 처리 중 수로만 셉니다 |
| `BALANCER_WEIGHT_WAITING` / `_RUNNING` / `_KV` | `4` / `1` / `8` | 점수 = 대기×4 + (처리 중 + 마지막 확인 뒤 보낸 수)×1 + KV 사용률(0~1)×8 |
| `BALANCER_DOWN_SECONDS` | `10` | 연결이 안 된 서버를 빼 두는 시간(초). 모두 빠지면 가장 최근에 정상이던 서버로 보냅니다 |
| `BALANCER_CONNECT_TIMEOUT` | `3` | 연결 단계 제한 시간(초). 방화벽 DROP이나 죽은 IP처럼 답이 없는 서버를 이 시간 안에 포기하고 다음 서버로 보냅니다 |
| `BALANCER_REQUEST_TIMEOUT` | `600` | 연결된 뒤 응답을 기다리는 제한 시간(초). 추론은 몇 분 걸릴 수 있습니다 |
| `BALANCER_MAX_BODY_MB` | `64` | 요청 본문 상한(MiB). 넘으면 서버를 고르기 전에 413입니다 |
| `BALANCER_MAX_CONNECTIONS` / `BALANCER_MAX_KEEPALIVE` | `1024` / `128` | 공유 업스트림 클라이언트의 연결 상한과 재사용할 유휴 연결 수 |

`GET http://localhost:3997/balancer/status`로 서버별 점수 재료(대기·처리 중·KV), 우리가 보낸 처리 중 수,
뺀 상태, 고른 횟수를 볼 수 있습니다.

연결 단계에서 실패한 요청(거부 `ConnectError`, 제한 시간 초과 `ConnectTimeout`)만 다음 서버로 다시 보냅니다.
아직 아무것도 보내지 않았으므로 안전합니다. 연결이 된 뒤의 5xx나 응답 대기 타임아웃(`ReadTimeout`)은 그 서버에서
이미 실행 중일 수 있으므로 다시 보내지 않고 그대로 돌려줍니다. 재시도·fallback은 LiteLLM이 합니다. 테스트는
`python -m pytest balancer/tests -q`입니다.

**메모리·연결.** 연결이 거부되면 같은 본문을 다음 서버로 다시 보내야 하므로, 본문은 메모리에 통째로 들고
있습니다(`BALANCER_MAX_BODY_MB`까지). 처리 중인 큰 요청 하나는 대략 **본문 크기의 2배**를 차지하고(우리가 든
사본 + httpx 요청 객체) 응답이 끝나면 놓습니다. 업스트림 연결은 프로세스 하나가 공유하는 클라이언트에서 재사용됩니다.

로컬 측정(가짜 vLLM 2대, 2026-10-21):
- 15 MiB 본문 20건을 동시에 보내고 각각 10초씩 스트리밍: 전부 성공했고 RSS는 54 MB → 최대 597 MB → 끝나고 200 MB, FD는 최대 56개였습니다.
- 작은 요청 200건을 동시에 2초씩 스트리밍: 전부 성공했고 p50 2.45 s, FD는 최대 416개였다가 끝나고 18개로 돌아왔습니다.
- 요청 420건에 업스트림 연결은 202개만 열렸습니다. 동시 200건만큼은 새로 열고, 그다음 순차 요청 200건은 그 연결을 재사용했습니다.
- 끝난 뒤 서버별 처리 중 수는 0이었습니다.

큰 이미지 요청을 많이 동시에 받는 배포라면 `BALANCER_MAX_BODY_MB`와 컨테이너 메모리를 함께 정해 두세요.

---

## 모델이 실제로 받는 요청 기록

ChatDRAGON·oh-my-gateway 화면에는 SDK 이벤트만 보이고, 그 아래에서 모델이 실제로 읽는 것은 보이지 않습니다.
예를 들어 Claude Code 시스템 프롬프트, MCP를 포함한 모든 도구 스키마, 매 턴 다시 보내는 전체 대화 기록,
그리고 LiteLLM이 덧붙이거나 바꾼 것(`max_tokens`, `stream_options`, 서빙 모델 이름, 지워진 thinking 블록)이
그렇습니다. 이 모두가 한 번에 있는 곳은 LiteLLM이 백엔드(vLLM/SGLang)로 보내기 직전의 요청 본문
(`complete_input_dict`)뿐입니다. 기록되는 것은 백엔드가 받는 것과 **같은 JSON 값**입니다. 보기 명령이 다시
직렬화하므로 공백이나 이스케이프까지 전송 바이트와 똑같지는 않습니다.
`prompt_capture.py`가 그 지점에서 요청을 기록합니다.

**기본 꺼짐.** 켜면 사용자 대화, 도구 결과, Agent가 읽은 모든 내용이 디스크에 남습니다.
사용자 데이터베이스처럼 다루고, 볼륨·접근 권한·보존 기간을 정한 뒤 켜세요.
인증 헤더와 API 키는 기록하지 않습니다(헤더는 허용 목록만 남깁니다).

| 변수 | 기본값 | 설명 |
|---|---|---|
| `PROMPT_CAPTURE` | `false` | `true`면 기록 |
| `PROMPT_CAPTURE_DIR` | `/app/prompt_captures` | 저장 위치(compose는 `./prompt_captures`에 마운트) |
| `PROMPT_CAPTURE_RETENTION_DAYS` | `7` | 보존 일수(UTC 날짜 기준, 지난 날 파일은 삭제) |
| `PROMPT_CAPTURE_MAX_MB` | `2048` (compose: `102400` = 100GB) | 전체 용량 상한(`0` = 상한 없음). 아래 "용량 상한" 참고 |

- **저장 형식:** 날짜별 `YYYY-MM-DD.jsonl`에 백엔드 호출 한 건이 한 줄입니다.
  Claude Code는 같은 시스템 프롬프트·도구 목록·대화 기록을 매 턴 다시 보내므로, `messages[i]`와 `tools`는
  `blobs/`에 내용 해시로 **한 번만** 저장하고 줄에는 참조만 둡니다.
- **요청 경로에 남는 비용:** 훅은 본문을 직렬화해 큐에 넣기만 하고, 디스크 쓰기는 별도 스레드가 합니다.
  직렬화는 요청 경로에서 일어나며 0.7MB(약 128K 토큰) 본문에 2ms 정도 걸립니다. 큐가 차면 그 기록을 버리고
  로그만 남깁니다. 모델 호출을 기다리게 하거나 실패시키지는 않습니다. 정상 종료 때는 큐에 남은 기록을 다 쓰고 끝납니다.
- **여러 worker:** LiteLLM worker 여러 개가 한 디렉터리를 같이 써도 됩니다. 모든 쓰기·정리는 `.lock`
  파일 잠금(flock) 안에서 일어나고, 조각 파일은 worker마다 고유한 임시 파일에 다 쓴 뒤 원자적으로 교체됩니다.
- **용량 상한:** 기록 하나를 쓰기 **전에** 디스크 사용량과 비교하므로, 순간적으로도 상한을 넘지 않습니다.
  넘을 것 같으면 상한의 80%까지 줄입니다. 먼저 오래된 날짜를 통째로 지우고, 하루치만 남았는데도 크면 그날의
  오래된 기록부터 지웁니다(가장 최근 기록이 남습니다). 기록 하나가 상한의 80%보다 크면 저장하지 않고 건너뜁니다.
  상한은 이 디렉터리만 지키므로, 볼륨에 그만큼의 여유가 실제로 있어야 합니다.
- **세션과 연결:** Claude CLI가 보내는 `x-claude-code-session-id`·`agent-type`·`agent-id`를 함께 기록합니다.
  그래서 한 ChatDRAGON 대화, 또는 그 안의 subagent 호출만 골라 볼 수 있습니다.

보기(컨테이너 안에서):

```bash
python /app/prompt_capture.py list                       # 최근 50건
python /app/prompt_capture.py list --session <세션 id>    # 한 Claude 세션의 호출만
python /app/prompt_capture.py show <call_id>             # 시스템 프롬프트·도구·메시지를 읽기 좋게
python /app/prompt_capture.py show <call_id> --tools-full  # 도구 스키마 전체
python /app/prompt_capture.py show <call_id> --raw       # 백엔드가 받은 JSON 그대로
```

여기에 남는 것은 chat template을 적용하기 **전** JSON입니다. 토큰으로 렌더된 최종 프롬프트(도구가 시스템
프롬프트 안에 펼쳐진 모습)까지 보려면 vLLM 쪽 요청 로그나 `/tokenize`를 써야 합니다.

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
