# litellm_serving

SGLang / vLLM 같은 비-Anthropic 백엔드 모델을 **Anthropic Messages API(`/v1/messages`)** 와
**OpenAI API(`/v1/chat/completions`)** 로 서빙합니다. Claude Code, oh-my-gateway(ChatDRAGON)가
그대로 붙어 쓰는 것을 목표로 합니다.

LiteLLM 프록시 앞에 sanitizer(얇은 리버스 프록시)를 두어 LiteLLM Anthropic 어댑터의 버그를
우회하고, reasoning 모델(GLM, Qwen, Gemma 등)의 thinking 출력과 reasoning effort를 다룹니다.

```
Claude Code / oh-my-gateway        ANTHROPIC_BASE_URL=http://<host>:3996
        │  Anthropic /v1/messages (+ /v1/models 등은 그대로 중계)
        ▼
  :3996  sanitizer   ── SSE 정리 · Anthropic→OpenAI 브리지 · effort 학습/clamp · tool_use id 정리
        │  /v1/chat/completions
        ▼
  :3999  LiteLLM     ── model_name → 백엔드 라우팅, 인증(master key), strip_thinking · prompt_capture 훅
        │
        ├──────────────► vLLM / SGLang  (litellm_config.yaml 의 api_base)
        └─► :3997 balancer (선택, BALANCER_POOLS) ─► 같은 모델의 vLLM 여러 대
```

> **클라이언트는 항상 :3996(sanitizer)을 보게 하세요.** :3999(LiteLLM)를 직접 보면 SSE 정리,
> 이미지 재배치, reasoning effort 게시(`effort_levels`)와 clamp가 모두 빠집니다.

---

## 빠른 시작

### Docker (권장)

컨테이너 하나에 sanitizer(:3996)와 LiteLLM(:3999)이 함께 뜹니다(`entrypoint.sh`). `BALANCER_POOLS`가
있으면 balancer(:3997, 루프백)도 뜹니다. 셋 중 하나라도 죽으면 컨테이너가 종료되어 재시작됩니다.
`docker-compose.yml`은 `network_mode: host`라서 컨테이너 안의 `localhost`가 호스트입니다.

```bash
# 1. litellm_config.yaml 의 api_base 를 실제 모델 서버 주소로 고친다 (아래 "모델 설정")
# 2. 빌드 및 실행
docker compose up -d --build

# 3. 확인
curl -s http://localhost:3996/health                                   # {"status":"ok"}
curl -s http://localhost:3996/v1/models -H "Authorization: Bearer sk-1234"
curl -s http://localhost:3996/v1/messages \
  -H "Authorization: Bearer sk-1234" -H "content-type: application/json" \
  -d '{"model":"glm-5.1-fp8","max_tokens":64,"messages":[{"role":"user","content":"ping"}]}'
```

`litellm_config.yaml`은 이미지에 복사되므로 설정을 바꾸면 `--build`로 다시 띄웁니다. 개발용
`docker-compose-dev.yml`(`Dockerfile.dev`)은 sanitizer·balancer·설정 파일을 bind-mount 하므로
재빌드 없이 재시작만 하면 되고, sanitizer 포트가 **:5501** 입니다.

thinking 출력 방식은 환경 변수로 바꿉니다(아래 "환경 변수").

```bash
THINK_OUTPUT_MODE=think_tag docker compose up -d --build
```

### Docker 미사용

컨테이너와 같은 구성(sanitizer + LiteLLM)을 직접 띄웁니다. 레포 루트에서 실행합니다.

```bash
# 1. 의존성 (sanitizer는 litellm을 import하지 않고 HTTP만 쓴다)
pip install 'litellm[proxy]' "fastapi==0.115.*" "uvicorn[standard]==0.32.*" "httpx==0.27.*"

# 2. 공통 환경 변수
export PYTHONPATH=.
export LITELLM_MASTER_KEY=sk-1234
export THINK_OUTPUT_MODE=none                     # default | think_tag | text | none
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1

# 3. LiteLLM (:3999) — thinking 패치 + (선택) 요청 기록 훅
LITELLM_WORKER_STARTUP_HOOKS=strip_thinking:apply_patch,prompt_capture:install \
  litellm --config litellm_config.yaml --port 3999 --host 0.0.0.0 &

# 4. sanitizer (:3996) — 클라이언트가 볼 주소
UPSTREAM_BASE_URL=http://localhost:3999 SANITIZER_USE_OPENAI_BRIDGE=true \
  python -m uvicorn sanitizer.main:app --host 0.0.0.0 --port 3996
```

`SANITIZER_USE_OPENAI_BRIDGE`의 코드 기본값은 `false`지만, 운영 구성(Dockerfile·compose)은 `true`입니다.
위처럼 직접 띄울 때도 켜 두세요(이유는 아래 "Sanitizer" 절).

---

## 환경 변수

### `THINK_OUTPUT_MODE`

모델의 thinking/reasoning 콘텐츠 출력 방식을 제어합니다. LiteLLM 쪽 패치(`strip_thinking.py`)와
sanitizer가 같은 변수를 읽습니다.

| 값 | 동작 | 출력 예시 |
|---|---|---|
| `default` | LiteLLM 기본 동작 (thinking_delta 그대로 전달) | *(litellm이 처리하는 대로)* |
| `think_tag` | `<think>...</think>` 태그로 감싸서 일반 텍스트로 출력 | `<think>\n추론 내용...\n</think>\n\n실제 응답` |
| `text` | 태그 없이 일반 텍스트로 출력 | `추론 내용...\n\n실제 응답` |
| `none` | thinking 서명은 숨기되, GLM/vLLM처럼 `content` 없이 `reasoning_content`만 오는 응답은 빈 응답 방지를 위해 일반 텍스트로 승격 **(이미지 기본값)** | `실제 응답` |

sanitizer는 여기에 `bridge`(브리지가 만든 thinking 블록을 그대로 두기)를 더 받습니다. 자세한 것은
[`sanitizer/README.md`](sanitizer/README.md).

> **참고**: 이 설정은 주로 Anthropic Messages API 스트리밍 엔드포인트(`/v1/messages`)에 적용됩니다.
> OpenAI 형식(`/v1/chat/completions`)에서는 `litellm_config.yaml`의 `merge_reasoning_content_in_choices: true` 설정이 reasoning 병합을 제어합니다.

> **GLM/vLLM 주의**: 일부 reasoning 모델은 `content: null` 상태로 실제 응답을 `reasoning_content`에만 담아 스트리밍할 수 있습니다.
> 이 경우 `none` 모드에서도 해당 텍스트를 `text_delta`로 승격하여 Claude Code subagent의 `response_text`가 빈 문자열이 되는 문제를 방지합니다.

### 포트와 기본 구성

| 변수 | 설명 | 기본값 (Dockerfile) |
|---|---|---|
| `LITELLM_MASTER_KEY` | LiteLLM 프록시 인증 키. 클라이언트의 `ANTHROPIC_AUTH_TOKEN`이 이 값이어야 하고, sanitizer의 effort 프로브도 이 키를 씁니다 | `sk-1234` (compose) |
| `SANITIZER_PORT` | sanitizer 포트 — 클라이언트가 보는 주소 | `3996` (dev compose: `5501`) |
| `LITELLM_PORT` | LiteLLM 포트 | `3999` |
| `UPSTREAM_BASE_URL` | sanitizer가 중계할 LiteLLM 주소 | `http://localhost:3999` |
| `SANITIZER_USE_OPENAI_BRIDGE` | `/v1/messages`를 LiteLLM `/v1/chat/completions`로 변환해 호출 | `true` |
| `BALANCER_POOLS` / `BALANCER_PORT` | 부하 기준 분배(아래 balancer 절). 비우면 balancer는 뜨지 않음 | *(끔)* / `3997` |
| `PROMPT_CAPTURE` | 모델이 실제로 받는 요청 기록(아래 절) | `false` |
| `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` | 사내 CA 인증서 경로 | *(시스템 기본)* |
| `NO_PROXY` / `no_proxy` | 프록시 우회 대상. 사내 프록시가 있으면 `localhost,127.0.0.1`은 꼭 넣습니다(TROUBLESHOOTING #1) | `localhost,127.0.0.1` (compose) |

sanitizer 전용 변수(`SANITIZER_TLS_VERIFY`, `SANITIZER_REQUEST_TIMEOUT`, `SANITIZER_EFFORT_PROBE`,
`SANITIZER_EFFORT_VOCABULARY` 등)는 [`sanitizer/README.md`](sanitizer/README.md)의 Configuration 표에 있습니다.

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

- `model_name` — 클라이언트가 요청 시 사용하는 모델 이름. **대소문자를 구분합니다**(그래서 `glm-5-fp8`/`GLM-5-FP8`처럼 두 벌을 둡니다)
- `model` — litellm이 사용하는 내부 모델 식별자 (prefix로 백엔드 타입 지정: `hosted_vllm/`, `openai/` 등).
  vLLM/SGLang에는 `hosted_vllm/`을 쓰세요. `openai/`면 LiteLLM이 백엔드에 없는 `/v1/responses`를 부를 수 있습니다(TROUBLESHOOTING #3)
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

모델 서버 자체를 띄우는 스크립트 예시는 `models/`(gemma4, qwen35)와 `run_qwen36.sh`·`setup_qwen36.sh`,
`run_supergemma4.sh`·`setup_supergemma4.sh`에 있습니다.

---

## Claude Code 연결

Claude Code는 `ANTHROPIC_BASE_URL`이 가리키는 곳에 Anthropic Messages API로 요청하므로, 이 스택의
sanitizer(:3996)를 그 주소로 주면 됩니다. 이 절의 설정은 Claude Code **2.1.283** 번들 CLI로 실제
경로(CLI → sanitizer(브리지 켬) → LiteLLM 자리의 가짜 서버)를 띄워 확인했습니다(2026-11-06).
아래에 "실측"이라고 적은 값은 그때 상류가 실제로 받은 요청에서 읽은 것입니다.

### 1. 연결 전 확인

Claude Code를 켜기 전에, Claude Code가 돌 머신에서 아래 두 요청이 되는지 먼저 봅니다. 여기서 막히면
Claude Code에서도 똑같이 막힙니다.

```bash
export GW=http://<서버 주소>:3996
export KEY=sk-1234          # LITELLM_MASTER_KEY

curl -s $GW/v1/models -H "Authorization: Bearer $KEY"       # 등록된 model_name 목록 (+ effort_levels)
curl -s $GW/v1/messages -H "Authorization: Bearer $KEY" -H "content-type: application/json" \
  -d '{"model":"glm-5.1-fp8","max_tokens":64,"messages":[{"role":"user","content":"ping"}]}'
```

- `Connection refused` → 포트가 :3996인지(개발 compose면 :5501), 방화벽이 열려 있는지 확인합니다.
- 사내 프록시 환경에서 프록시 응답(403/HTML)이 오면 서버 주소를 `NO_PROXY`에 넣습니다. Claude Code도
  `HTTPS_PROXY`/`HTTP_PROXY`/`NO_PROXY`를 따릅니다.

### 2. 가장 짧은 연결

```bash
export ANTHROPIC_BASE_URL=http://<서버 주소>:3996
export ANTHROPIC_AUTH_TOKEN=sk-1234                  # LITELLM_MASTER_KEY
export ANTHROPIC_MODEL=glm-5.1-fp8                   # litellm_config.yaml 의 model_name 그대로
export ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.6-27b     # 백그라운드 작업용 경량 모델 (아래 3절)
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1    # 폐쇄망: 텔레메트리·오류 보고·자동 업데이트 끔

claude                      # 대화형
claude -p "say ping"        # 한 번만 실행해 연결 확인
```

- **인증은 `ANTHROPIC_AUTH_TOKEN`을 쓰세요.** `Authorization: Bearer <키>`로 나가고(실측), sanitizer는
  인증 헤더를 그대로 LiteLLM에 넘기며, LiteLLM이 master key와 비교합니다. Anthropic 계정 로그인
  (`/login`)은 필요 없습니다. `ANTHROPIC_API_KEY`는 `x-api-key` 헤더로 나가고, 대화형 실행에서는 처음에
  "이 API 키를 쓸지" 묻습니다.
- `ANTHROPIC_BASE_URL`에는 `/v1`을 붙이지 않습니다. Claude Code가 `/v1/messages`를 붙입니다.

### 3. `~/.claude/settings.json`으로 고정하기

매번 export하지 않으려면 settings 파일의 `env`에 넣습니다. 사용자 전체에 적용하려면
`~/.claude/settings.json`, 한 프로젝트에만 적용하려면 그 프로젝트의 `.claude/settings.json`
(개인용이면 `.claude/settings.local.json`)에 둡니다.

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://<서버 주소>:3996",
    "ANTHROPIC_AUTH_TOKEN": "sk-1234",

    "ANTHROPIC_MODEL": "glm-5.1-fp8",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-5.1-fp8",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-5.1-fp8",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen3.6-27b",
    "CLAUDE_CODE_SUBAGENT_MODEL": "glm-5.1-fp8",

    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "131072",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "16384",

    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "API_TIMEOUT_MS": "600000"
  }
}
```

**우선순위 주의(실측):** settings 파일의 `env`가 셸에서 export한 같은 이름의 변수보다 이깁니다.
settings에 `ANTHROPIC_MODEL`을 적어 두면 셸의 `ANTHROPIC_MODEL=...`은 무시됩니다. 한 번만 다른 모델을
쓰려면 `claude --model <이름>`을 쓰세요. `--model`은 둘 다 이깁니다.

각 변수의 역할:

| 변수 | 역할 |
|---|---|
| `ANTHROPIC_MODEL` | 기본 메인 모델. 값이 그대로 `/v1/messages`의 `model`이 되어 LiteLLM `model_name`과 맞춰집니다 |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` / `_SONNET_MODEL` / `_HAIKU_MODEL` | `/model`의 `opus`·`sonnet`·`haiku` 별칭이 가리킬 모델. **haiku는 꼭 지정하세요**: Claude Code는 대화 제목, 요약 같은 백그라운드 작업에 haiku 계열을 씁니다. 지정하지 않으면 LiteLLM이 모르는 Claude 모델 id가 나가서 `400 Invalid model name`이 납니다 |
| `CLAUDE_CODE_SUBAGENT_MODEL` | subagent(Agent 도구)가 쓸 모델 |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | Claude Code가 모르는 모델의 컨텍스트 윈도 선언. 5절의 이름 규칙 주의 |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | 요청의 `max_tokens`. 지정하지 않으면 모르는 모델에 **32000**이 나갑니다(실측) |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | Anthropic 쪽으로 나가는 부가 트래픽(텔레메트리, 오류 보고, 업데이트 확인)을 끕니다 |
| `API_TIMEOUT_MS` | 요청 하나의 제한 시간(ms). 긴 추론, 큰 컨텍스트 prefill이 길면 올립니다 |

### 4. 모델 고르기

LiteLLM의 `model_name`이면 **어떤 이름이든** 쓸 수 있습니다. 이름에 `claude`가 들어갈 필요는 없습니다.

- 시작할 때: `claude --model glm-5.1-fp8`
- 기본값: `ANTHROPIC_MODEL` (위 settings)
- 대화 중: `/model glm-5.1-fp8`처럼 이름을 직접 적거나, `/model`에서 `opus`/`sonnet`/`haiku`를
  고르면 `ANTHROPIC_DEFAULT_*_MODEL`이 가리키는 모델로 갑니다.

`model_name`은 대소문자를 구분하므로 철자를 `GET /v1/models` 결과와 똑같이 적습니다.

#### `/model` 픽커에 목록으로 띄우기 (gateway model discovery)

Claude Code(2.1.129+)는 `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`이면 시작할 때
`ANTHROPIC_BASE_URL`의 `GET /v1/models?limit=1000`을 읽어 `/model` 픽커를 채웁니다("From gateway").
sanitizer는 `POST /v1/messages` 외 전부를 상류로 중계하므로 이 요청은 이미 LiteLLM까지 갑니다.
**막는 것은 모델 이름뿐입니다.** Claude Code는 id에 `claude` 또는 `anthropic`이 **포함된** 모델만
남깁니다. 접두사 매치가 아니라 부분 문자열 매치입니다(앵커 없는 `/(claude|anthropic)/i`, 2.1.251
바이너리에서 확인).

기본 `litellm_config.yaml`의 이름에는 그 문자열이 없어서 0개가 남습니다. 별칭을 추가하면 됩니다.
[`litellm_config.claude-code-discovery.example.yaml`](./litellm_config.claude-code-discovery.example.yaml)은
기존 12개 이름을 그대로 두고 실모델 7종에 `claude-*` 별칭을 추가한 전체 예시입니다.

```bash
# LiteLLM 을 별칭이 들어간 config 로 띄운 뒤:
export ANTHROPIC_BASE_URL=http://<서버 주소>:3996
export ANTHROPIC_AUTH_TOKEN=sk-1234
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
claude        # /model 픽커에 별칭들이 "From gateway" 로 뜬다
```

- 목록은 `~/.claude/cache/gateway-models.json`에 캐시됩니다. 별칭을 바꿨는데 픽커가 그대로면 이 파일을
  지우고 다시 시작하세요. `claude --debug`의 `[gatewayDiscovery] cached N models`로 몇 개가 실렸는지
  볼 수 있습니다.
- 이 example은 실제 구성(LiteLLM proxy + sanitizer + claude-code 2.1.251)으로 E2E 검증됐습니다. CLI
  디버그 로그에 `[gatewayDiscovery] cached 7 models`가 기록되고 7종 전부 픽커 캐시에 실렸습니다.
- **별칭 이름을 새로 정한다면 `claude`를 앞이 아닌 뒤에 붙이는 쪽을 권합니다**(예: `glm-5.1-fp8-claude`).
  디스커버리 필터는 똑같이 통과하고, 5절의 `CLAUDE_CODE_MAX_CONTEXT_TOKENS`도 적용됩니다(2.1.283 실측).

### 5. 컨텍스트 윈도와 출력 토큰

Claude Code는 자기가 모르는 모델 id의 컨텍스트 윈도를 **200k로 가정**하고 auto-compact를 그 기준으로
돌립니다. 실제 윈도가 128k나 32k인 모델이면 Claude Code가 압축하기 전에 vLLM이 먼저
`prompt is too long` 류로 거부합니다. 실제 윈도는 vLLM의 `--max-model-len`입니다.

- **`CLAUDE_CODE_MAX_CONTEXT_TOKENS=<실제 윈도>`** — 모르는 모델 id에 적용되는 선언입니다. 전역 env라
  윈도가 다른 모델을 오가며 쓰기에는 불편합니다(한 settings에는 하나의 값).
- **`claude-`로 시작하는 이름에는 적용되지 않습니다.** 2.1.283에서 `claude-`로 시작하는 소문자 id는
  Claude 모델 이름으로 취급되어 이 변수를 건너뜁니다. 실측(같은 값 131072를 줌):

  | 모델 id | Claude Code가 쓴 윈도 |
  |---|---|
  | `glm-5.1-fp8` | 131072 |
  | `glm-5.1-fp8-claude` | 131072 |
  | `claude-glm-5.1-fp8` | 200000 (변수 무시) |

  `claude-*` 별칭을 계속 쓴다면 압축 시점을 `CLAUDE_CODE_AUTO_COMPACT_WINDOW=<토큰 수>`(또는 대화 중
  `/autocompact <토큰 수>`)로 앞당깁니다. 이 값은 모델과 상관없이 "이만큼 차면 압축"이고, 실제 기준은
  이 값과 모델 윈도 중 작은 쪽입니다(CLI 설명 문구 기준, 직접 측정은 하지 않음).
- **출력 토큰:** `CLAUDE_CODE_MAX_OUTPUT_TOKENS`가 요청의 `max_tokens`로 그대로 나갑니다(실측: 8192를
  주면 상류가 `max_tokens: 8192`를 받음). 입력 + `max_tokens`가 `--max-model-len`을 넘으면 vLLM이
  거부하므로, 윈도가 작은 모델일수록 낮게 잡습니다.
- `modelOverrides` 설정은 윈도 지정이 아니라 **인식되는 Anthropic 모델 id → 게이트웨이 별칭 문자열**
  매핑입니다. 인식되는 id로 매핑하면 그 모델의 알려진 윈도를 따릅니다.
- `CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1`은 API가 too-long을 돌려줄 때까지 기다렸다
  압축하는 사후 방식입니다. 그런데 **게이트웨이가 에러 문구를 재작성하면 Claude Code가 그 에러를
  인식하지 못합니다.** sanitizer가 응답을 정규화하는 이 구성에서는 믿지 마세요.

### 6. Reasoning effort

Claude Code는 기본으로 `effort: high`를 보냅니다(실측: 상류가 `reasoning_effort: "high"`를 받음).
모델 템플릿이 `high`를 받지 않아도(예: `low|medium|xhigh`만 받는 Qwen) sanitizer가 모델별로 받는 레벨을
배워 두고 clamp하므로 400이 나지 않습니다. 배운 표는 `GET http://<서버>:3996/sanitizer/effort-levels`에서
봅니다. Claude Code 쪽에서 바꾸려면 대화 중 `/effort`나 `CLAUDE_CODE_EFFORT_LEVEL`을 씁니다.
동작 원리는 [`sanitizer/README.md`](sanitizer/README.md)의 "Reasoning effort".

### 7. 문제 해결

| 증상 | 원인과 조치 |
|---|---|
| `401` / `Authentication Error` | `ANTHROPIC_AUTH_TOKEN` ≠ `LITELLM_MASTER_KEY`. 1절의 curl로 같은 키가 통하는지 확인합니다 |
| `400 Invalid model name passed in model=...` | `model_name` 철자·대소문자가 틀렸거나, `ANTHROPIC_DEFAULT_HAIKU_MODEL`을 지정하지 않아 백그라운드 작업이 Claude 모델 id를 보냈습니다. 오류 메시지의 `model=` 값을 보세요 |
| 응답이 비어 있음 / subagent 결과가 빈 문자열 | `reasoning_content`만 오는 모델입니다. `THINK_OUTPUT_MODE=none`(기본)이면 텍스트로 승격됩니다 |
| 답에 `<think>…</think>` 태그가 섞임 | 그 모델에 `merge_reasoning_content_in_choices: true`가 켜져 있습니다. 브리지 앞에서는 끄세요("모델 설정") |
| 대화가 길어지면 `prompt is too long` / context length 오류 | 5절: `CLAUDE_CODE_MAX_CONTEXT_TOKENS`(또는 `claude-*` 별칭이면 `CLAUDE_CODE_AUTO_COMPACT_WINDOW`)와 `CLAUDE_CODE_MAX_OUTPUT_TOKENS`를 실제 윈도에 맞춥니다 |
| `/model` 픽커에 게이트웨이 모델이 안 뜸 | `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`인지, id에 `claude`/`anthropic`이 들어갔는지 확인하고 `~/.claude/cache/gateway-models.json`을 지웁니다 |
| 긴 작업에서 타임아웃 | `API_TIMEOUT_MS`를 올립니다. sanitizer 쪽 상한은 `SANITIZER_REQUEST_TIMEOUT` |
| 셸에서 바꾼 `ANTHROPIC_MODEL`이 안 먹음 | settings 파일 `env`가 이깁니다(3절). `--model`을 쓰거나 settings에서 지웁니다 |
| 같은 도구 호출을 반복하며 멈추지 않음 | 상류가 같은 `tool_use` id를 반복하는 경우입니다. sanitizer가 고치므로 :3999를 직접 보고 있지 않은지 확인합니다 |

모델이 실제로 무엇을 받았는지 봐야 할 때는 아래 "모델이 실제로 받는 요청 기록"을 켜고
`python /app/prompt_capture.py list --session <세션 id>`로 봅니다.

### oh-my-gateway에서 쓰기

oh-my-gateway(ChatDRAGON 백엔드)도 Claude Code CLI를 띄우므로 연결 방법이 같습니다. gateway의
`ANTHROPIC_BASE_URL`을 :3996으로, `ANTHROPIC_AUTH_TOKEN`을 master key로 주고, 모델은 요청의 `model`로
`model_name`을 보냅니다. ChatDRAGON의 effort 선택지는 sanitizer가 `GET /v1/models`에 싣는
`effort_levels`에서 옵니다. gateway가 :3999를 직접 보면 이 값이 없습니다.

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

## 선택형 SMG Router (HTTP PoC)

3~4개의 동일 모델 인스턴스에 대한 `cache_aware` 분산을 시험하려면
[`smg/README.md`](smg/README.md)를 참고하세요. 별도
[`docker-compose.smg.yml`](docker-compose.smg.yml) 오버레이를 사용하므로 기본 배포는
변경되지 않습니다. 기존 `balancer/`는 보존하고, SMG는 대체 라우트로만 사용합니다.
`scripts/benchmark_smg.py`로 shuffle / 자체 balancer / SMG의 TTFT와 처리량을
같은 조건에서 비교할 수 있습니다. SMG HTTP 모드는 실제 KV-cache 이벤트가 아닌
추정 기반 prefix routing이므로 성능 검증 후 선택적으로 전환하세요.

---

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

---

## 프로젝트 구조

```
├── litellm_config.yaml          # 모델 및 LiteLLM 설정 (이미지에 /app/config.yaml 로 복사)
├── litellm_config.claude-code-discovery.example.yaml
│                                # Claude Code /model 픽커용 claude-* 별칭을 더한 전체 예시
├── strip_thinking.py            # thinking 출력 제어 (THINK_OUTPUT_MODE, LiteLLM 워커 훅)
├── prompt_capture.py            # 모델이 실제로 받는 요청 기록 (PROMPT_CAPTURE, 기본 꺼짐)
├── sanitizer/                   # Anthropic-facing 리버스 프록시 (:3996) — 브리지, effort, tool_use id
├── balancer/                    # 같은 모델의 vLLM 여러 대를 실제 부하 기준으로 분배 (:3997, BALANCER_POOLS)
├── smg/                         # 선택형 SMG cache-aware 라우터 런처 (docker-compose.smg.yml)
├── scripts/
│   ├── effort_path_smoke.py     # 가짜 LiteLLM + 실제 sanitizer로 effort 경로 스모크
│   └── benchmark_smg.py         # shuffle / balancer / SMG TTFT·처리량 비교
├── models/                      # 모델 서버(vLLM) 실행 예시: gemma4/, qwen35/
├── run_qwen36.sh, setup_qwen36.sh, run_supergemma4.sh, setup_supergemma4.sh
│                                # 모델 서버 실행/준비 스크립트
├── Dockerfile                   # 운영 이미지: sanitizer + LiteLLM (+ balancer)
├── docker-compose.yml           # 운영 compose (host network, :3996/:3999)
├── Dockerfile.dev, docker-compose-dev.yml
│                                # 개발용: 소스 bind-mount, sanitizer :5501
├── docker-compose.smg.yml       # SMG 오버레이 (선택)
├── entrypoint.sh                # sanitizer / balancer / LiteLLM 동시 기동, 하나라도 죽으면 종료
├── tests/                       # strip_thinking · prompt_capture 테스트
├── TROUBLESHOOTING.md           # 트러블슈팅 기록 (프록시, 모델 이름 대소문자, hosted_vllm/ prefix)
├── DEBUG_REPORT.md              # SDK 스트리밍 디버그 리포트
└── weekly_idd.md                # 주간 공유 메모
```

---

## 테스트

```bash
pip install 'litellm[proxy]' "fastapi==0.115.*" "uvicorn[standard]==0.32.*" "httpx==0.27.*" pytest pytest-asyncio

python -m pytest sanitizer/tests -q      # sanitizer (상류는 httpx.MockTransport)
python -m pytest balancer/tests -q       # balancer
python -m pytest smg/tests -q            # SMG 런처·벤치마크
python -m pytest tests -q                # strip_thinking · prompt_capture (litellm 필요)

python scripts/effort_path_smoke.py      # 실제 sanitizer 프로세스로 effort 경로 끝까지 확인
```
