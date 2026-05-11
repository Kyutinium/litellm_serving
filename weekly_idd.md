# Weekly IDD 주간 공유

## 날짜: 2026-03-27 (금)

---

## 1. 프로젝트 개요: LiteLLM Serving

- **목적**: LiteLLM 프록시를 활용한 LLM 모델 서빙 환경 구축
- **현재 구성**: OpenAI 호환 API 형태로 로컬 모델 서빙

## 2. 현재 상태

### 모델 구성
| 모델명 | 백엔드 | 엔드포인트 | 비고 |
|--------|--------|------------|------|
| glm-5-fp8 | SGLang (OpenAI 호환) | `http://localhost:10042/v1` | FP8 양자화 |

### 인프라
- LiteLLM Proxy를 통해 단일 엔드포인트로 여러 모델 라우팅 가능
- SGLang 백엔드 사용 (별도 API 키 불필요)

## 3. 이번 주 진행 사항

- [x] LiteLLM 서빙 레포지토리 초기 구성
- [x] GLM-5 FP8 모델 설정 (`litellm_config.yaml`)
- [x] SGLang 백엔드 연동 확인

## 4. 다음 주 계획

- [ ] 추가 모델 등록 (필요 시)
- [ ] 로드밸런싱 / fallback 설정 검토
- [ ] 모니터링 및 로깅 설정
- [ ] 사용량 트래킹 (budget/rate limit) 설정 검토

## 5. 이슈 / 논의 사항

- GLM-5-FP8 외 추가 모델 서빙 요청 여부 확인 필요
- 프로덕션 배포 시 인증(API Key) 설정 방안 논의 필요
- `general_settings` 세부 설정 (캐싱, 타임아웃 등) 정책 결정 필요

---

> 참고: LiteLLM 설정 파일 경로 - `litellm_config.yaml`
