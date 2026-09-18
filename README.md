# litellm_serving

vLLM / SGLang / llama.cpp 위의 오픈 모델을 **LiteLLM 프록시 하나**로 OpenAI 호환 API로 내고,
Anthropic API를 직접 말하는 클라이언트(Claude Code CLI)를 위해 그 앞에 얇은 **sanitizer**를 둔다.

## 누가 어디로 들어오나

이 저장소를 쓰는 소비자는 둘이고, 경로가 다르다. 설정을 볼 때 항상 이 축으로 보면 된다.

```
oh-my-gateway ──(gateway 자체 브리지)──▶ LiteLLM :3999 ──▶ 모델 서버들
ChatDRAGON 직접 연결 ───────────────────▶ LiteLLM :3999      (:8088 GLM, :8092 Qwen, …)
Claude Code CLI ──▶ sanitizer :3996 ───▶ LiteLLM :3999
```

| 소비자 | 말하는 API | 거치는 것 | 필요한 profile |
|---|---|---|---|
| oh-my-gateway | OpenAI `/v1/chat/completions` | LiteLLM만 (Anthropic↔OpenAI 변환은 gateway 안의 브리지가 함) | `litellm` |
| ChatDRAGON 직접 연결 | OpenAI `/v1/chat/completions` | LiteLLM만 | `litellm` |
| Claude Code CLI | Anthropic `/v1/messages` | **sanitizer** → LiteLLM | `litellm` + `sanitizer` |

sanitizer는 Claude Code 경로에만 필요하다. gateway 경로에 sanitizer를 끼우면 같은 변환을 두 번 한다.

## 설정은 어디에 사나

파일이 여러 개지만 **손으로 고치는 것은 둘**이다. 나머지는 생성물이다.

| 파일 | 역할 | 편집 |
|---|---|---|
| `models.yaml` | 모델의 모든 사실: 이름·별칭·역할·서빙 방법·LiteLLM 파라미터·effort 어휘 | **손으로** |
| `.env` | 비밀값과 배포별 값 (`LITELLM_MASTER_KEY` 등). gitignore. `.env.example`이 전체 키 목록 | **손으로** |
| `litellm_config.yaml` | LiteLLM이 읽는 모델 목록. 별칭마다 한 항목 | 생성 |
| `compose/<id>.yml` | 이 저장소가 직접 띄우는 모델(vllm/llamacpp)의 compose profile | 생성 |
| `.env.rendered` | `models.yaml`에서 나오는 env 한 줄 (`SANITIZER_EFFORT_VOCABULARY`). `.env`에 복사 | 생성 |
| `docker-compose.yml` | LiteLLM·sanitizer 서비스 + `compose/*.yml` include. 모델 추가 시 include 한 줄만 | 손으로(드물게) |

```bash
python3 scripts/render.py          # models.yaml → 위 생성물 세 종류
python3 scripts/render.py --check  # 생성물이 models.yaml과 어긋나면 exit 1 (CI/커밋 전)
```

생성물은 커밋한다. 그래야 `--check`가 "누군가 `litellm_config.yaml`을 직접 고쳐서 `models.yaml`이
거짓말을 하는" 상태를 잡는다. 실제 배포의 모델 목록은 저장소에 넣지 않는다 — 아래 [배포별 registry](#배포별-registry).

## 빠른 시작

```bash
cp .env.example .env            # LITELLM_MASTER_KEY 등을 채운다
python3 scripts/render.py       # 생성물 확인 (커밋된 상태면 no-op)

docker compose --profile litellm up -d --build                      # LiteLLM만
docker compose --profile litellm --profile sanitizer up -d --build  # + Claude Code용 sanitizer
docker compose --profile qwen3.6-27b up -d                          # 이 저장소가 띄우는 모델
```

profile 없이 `docker compose up`을 하면 아무것도 뜨지 않는다 — 의도된 동작이다.
이미지는 하나(`Dockerfile`)고 compose가 `command`로 LiteLLM/sanitizer 프로세스를 고른다.

Docker 없이:

```bash
pip install litellm
export LITELLM_WORKER_STARTUP_HOOKS=strip_thinking:apply_patch PYTHONPATH=.
export LITELLM_MASTER_KEY=…   # litellm_config.yaml이 os.environ/LITELLM_MASTER_KEY 로 읽는다
litellm --config litellm_config.yaml --port 3999 --host 0.0.0.0
```

## 모델 추가·변경

`models.yaml`에 한 블록을 쓰고 render한다. 다른 파일은 만지지 않는다.

```yaml
  - id: qwen3.6-27b                 # 클라이언트가 보내는 이름 (대소문자 구분)
    aliases: [Qwen3.6-27B]          # 같은 백엔드의 다른 이름. 각각 LiteLLM 항목이 된다
    roles: [gateway, direct]        # 문서용: 누구를 위한 모델인가
    serve:
      engine: vllm                  # vllm | llamacpp | external
      image: vllm/vllm-openai:latest
      checkpoint: /shared/checkpoints/…/Qwen3.6-27B
      gpus: ["2", "3", "4", "5"]
      port: 8092
      args: { tensor-parallel-size: "4", max-model-len: "32768", … }
    litellm:
      provider: hosted_vllm         # hosted_vllm | openai (LiteLLM model prefix)
      max_tokens: 32768
      merge_reasoning_content_in_choices: true
    sanitizer:
      effort_supported: [low, medium, xhigh]   # 이 모델의 upstream이 받는 effort만
```

- `serve.engine: external` — 모델 서버를 이 저장소 밖에서 띄운다. 포트(와 host)만 적으면
  LiteLLM 항목만 생성되고 compose profile은 만들지 않는다.
- `serve.engine: vllm | llamacpp` — `compose/<id>.yml`이 생성된다. **`docker-compose.yml`의
  `include:`에 그 파일을 한 줄 추가**한다 (`--check`가 빠진 것·없는 것을 잡는다).
  llama.cpp는 `gguf`를 비우면 checkpoint 디렉터리의 첫 `*.gguf`를 쓴다.
- `aliases` — 대소문자 변형, Claude Code 디스커버리용 `claude-*`, 제품명 등 **같은 백엔드**를
  가리키는 다른 이름. 이름은 registry 전체에서 유일해야 한다(로드 시 검증).
- `sanitizer.effort_supported` — 생략하면 그 모델은 effort를 그대로 통과시킨다. 아래 [effort](#reasoning-effort).

### 배포별 registry

실제 배포의 모델 목록(내부 모델명, 호스트 경로)은 저장소에 두지 않는다. 같은 형태의
`models.local.yaml`(gitignore)을 만들고 그것으로 render한다:

```bash
MODELS_FILE=models.local.yaml python3 scripts/render.py
```

생성물의 위치는 같으므로, 배포 머신에서는 render 후 그대로 `docker compose … up`이다.
저장소의 `models.yaml`은 개발용 registry이자 형태의 예시다.

## 환경 변수 — 누가 읽나

같은 이름을 두 프로세스가 읽던 것이 설정을 읽기 어렵게 만들던 원인이라, 표에 **읽는 쪽**을 적는다.

| 변수 | 읽는 쪽 | 기본값 | 의미 |
|---|---|---|---|
| `LITELLM_MASTER_KEY` | LiteLLM (`litellm_config.yaml`의 `os.environ/`) | **필수** | 프록시 bearer 키. 예전의 하드코딩 `sk-1234`는 없다 |
| `LITELLM_PORT` | compose | `3999` | LiteLLM 포트 |
| `THINK_OUTPUT_MODE` | LiteLLM worker hook `strip_thinking.py` | `none` | LiteLLM 자체 `/v1/messages` 어댑터의 thinking 출력. 아래 표 |
| `UPSTREAM_BASE_URL` | sanitizer | `http://localhost:3999` | sanitizer가 부르는 LiteLLM |
| `SANITIZER_PORT` | sanitizer / compose | `3996` | |
| `SANITIZER_USE_OPENAI_BRIDGE` | sanitizer | `true` (compose) | LiteLLM의 Anthropic 어댑터 대신 in-process 변환 |
| `SANITIZER_THINK_OUTPUT_MODE` | sanitizer `output_transform.py` | `default` | sanitizer **자신의** thinking 출력. 없으면 `THINK_OUTPUT_MODE`로 폴백 |
| `SANITIZER_EFFORT_VOCABULARY` | sanitizer (효력은 PR #27 병합 후) | *(없음 = 그대로 전달)* | 모델별 허용 effort. `.env.rendered`에서 복사 |
| `SANITIZER_REQUEST_TIMEOUT` | sanitizer | `0` | 초. 0 = 없음 |
| `SANITIZER_TLS_VERIFY` | sanitizer | `true` | `true` / `false` / CA 번들 경로 |
| `SANITIZER_FORWARD_REASONING_EFFORT` | sanitizer | `true` | `output_config.effort` → `reasoning_effort` 전달 |
| `MODELS_FILE` | `scripts/render.py` | `models.yaml` | 배포별 registry |
| `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `NO_PROXY` | LiteLLM | compose가 채움 | 사내 CA / 프록시 우회 |

sanitizer 쪽 변수의 상세는 [`sanitizer/README.md`](sanitizer/README.md).

### `THINK_OUTPUT_MODE`

reasoning 모델(GLM, DeepSeek, Qwen 등)의 thinking 출력 방식. **읽는 쪽이 둘**이라 키가 둘이다:
`THINK_OUTPUT_MODE`는 LiteLLM 안의 `strip_thinking.py`(LiteLLM 자체 `/v1/messages` 어댑터를
쓰는 경로)가, `SANITIZER_THINK_OUTPUT_MODE`는 sanitizer가 읽는다. 한쪽만 바꿀 수 있다.

| 값 | 동작 |
|---|---|
| `default` | thinking_delta 그대로 전달 |
| `think_tag` | `<think>…</think>` 태그로 감싸 일반 텍스트로 |
| `text` | 태그 없이 일반 텍스트로 |
| `none` | thinking 서명은 숨기되, GLM/vLLM처럼 `content` 없이 `reasoning_content`만 오는 응답은 빈 응답 방지를 위해 텍스트로 승격 (LiteLLM 쪽 기본값) |

OpenAI 형식(`/v1/chat/completions`)에서는 이 변수가 아니라 모델의 `merge_reasoning_content_in_choices`가
reasoning 병합을 결정한다.

> **`merge_reasoning_content_in_choices: true`는 sanitizer 브리지 뒤의 모델에는 켜지 마라.**
> 브리지는 upstream의 `reasoning_content`를 Anthropic `thinking` 블록으로 옮기는데, 이 옵션이 켜져 있으면
> LiteLLM이 먼저 reasoning을 `content` 안의 리터럴 `<think>…</think>` 텍스트로 합쳐 버려 Anthropic
> 클라이언트에는 thinking 블록 대신 태그가 섞인 본문이 도착한다(실측: `text_delta: "<think>…</think>답변"`).
> 브리지 없이 LiteLLM의 `/v1/messages` 어댑터를 직접 쓰는 경우에만 의미 있는 옵션이다.
> 한 모델을 두 경로가 함께 쓴다면 이 옵션을 끄고 gateway/브리지 쪽에서 reasoning을 처리하는 편이 맞다.

### Reasoning effort

Claude Code / oh-my-gateway는 `low|medium|high|xhigh|max`를 보내지만, 어떤 레벨을 받는지는 **모델의
chat template이 정한다** (현재 Qwen 3.6 템플릿은 `low|medium|xhigh`만 받고 `high`에 400을 낸다).
그래서 어휘는 모델별로 `models.yaml`의 `sanitizer.effort_supported`에 적고, render가
`SANITIZER_EFFORT_VOCABULARY`(별칭까지 전개한 JSON 맵)로 만들어 준다. 적지 않은 모델은 그대로 통과.
클램프는 위쪽 우선(`high` → `xhigh`)이다. sanitizer 쪽 구현과 규칙은 PR #27 —
그 전까지 main의 sanitizer는 이 변수를 읽지 않고 레벨을 그대로 전달한다.

## Claude Code 모델 디스커버리

Claude Code(2.1.129+)는 `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`이면 시작 시
`ANTHROPIC_BASE_URL`의 `GET /v1/models?limit=1000`을 읽어 `/model` 픽커를 채운다. sanitizer는
`POST /v1/messages` 외 전부를 LiteLLM으로 릴레이하므로 이 요청은 이미 도달한다 — **막는 것은 이름뿐**이다.
Claude Code는 id에 `claude` 또는 `anthropic`이 **포함된**(접두사가 아니라 부분 문자열, 앵커 없는
`/(claude|anthropic)/i`, 2.1.251 바이너리에서 확인) 모델만 남긴다.

그러므로 픽커에 띄우고 싶은 모델에 `claude-*` 별칭을 준다. 별칭 문자열이 곧 픽커에 뜨는 이름이자
`/v1/messages`의 `model`로 그대로 돌아오는 라우팅 키다:

```yaml
  - id: qwen3.6-27b
    aliases: [Qwen3.6-27B, claude-qwen3.6-27b]
```

실모델 7종에 `claude-glm-5-fp8`, `claude-glm-5.1-fp8`, `claude-qwen3.5-122b-a10b`, `claude-samuel-v2`,
`claude-gemma-4-31b-it`, `claude-supergemma4-26b`, `claude-qwen3.6-27b`를 준 구성은 실제
LiteLLM + sanitizer + claude-code 2.1.251로 E2E 검증됐다(`[gatewayDiscovery] cached 7 models`).
선택 후의 실제 추론은 백엔드가 살아 있는 배포에서 한 턴 돌려 확인할 것.

```bash
export ANTHROPIC_BASE_URL=http://<sanitizer-host>:3996
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
claude   # /model 픽커에 claude-* 가 "From gateway"로 뜬다
```

**컨텍스트 윈도 주의.** Claude Code는 모르는 모델 id의 윈도를 200k로 가정하고 auto-compact를 그 기준으로
돌린다. 32k 모델이면 upstream이 먼저 거부할 수 있다.

- `CLAUDE_CODE_MAX_CONTEXT_TOKENS=<실제 윈도>` — 미인식 id에 적용되는 선언. 전역 env라 윈도가 다른
  모델을 오가며 쓰기엔 불편하다.
- `modelOverrides`는 윈도 지정이 아니라 **인식되는 Anthropic 모델 id → 게이트웨이 별칭** 매핑이다.
  인식되는 id로 매핑하면 그 모델의 알려진 윈도를 따른다.
- `CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1`은 API가 too-long을 돌려줄 때까지 기다렸다
  압축하는 사후 방식인데, sanitizer가 에러 문구를 정규화하는 이 구성에서는 Claude Code가 그 에러를
  인식하지 못할 수 있다 — 신뢰하지 말 것.

## 프로젝트 구조

```
├── models.yaml            # ★ 모델 registry (손으로 편집하는 유일한 모델 설정)
├── scripts/render.py      # models.yaml → 아래 생성물. --check 로 드리프트 검사
├── litellm_config.yaml    # (생성) LiteLLM 모델 목록
├── compose/<id>.yml       # (생성) 모델별 compose profile
├── .env.rendered          # (생성) SANITIZER_EFFORT_VOCABULARY
├── .env.example           # 모든 env 키와 읽는 쪽. .env 로 복사
├── docker-compose.yml     # litellm / sanitizer profile + compose/*.yml include
├── Dockerfile             # LiteLLM + sanitizer 단일 이미지
├── strip_thinking.py      # LiteLLM worker hook (THINK_OUTPUT_MODE)
├── sanitizer/             # Anthropic-facing 리버스 프록시 (Claude Code 경로)
├── tests/test_render.py   # render 검증
├── TROUBLESHOOTING.md
└── DEBUG_REPORT.md
```

## 테스트

```bash
python3 scripts/render.py --check
python3 -m unittest tests.test_render -v
python3 -m pytest sanitizer/tests -q
```
