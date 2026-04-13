# AI Debate Hub (ai-debate)

Claude 3.5 Sonnet과 Gemini 2.0 Pro가 라운드별로 격렬하게 토론하고, 제3의 심판(Claude 3 Opus/GPT-4o)이 최종 권고를 내리는 실시간 AI 토론 도구입니다.

## 주요 특징
- **실시간 스트리밍**: SSE(Server-Sent Events)를 통해 토론자가 답변을 작성하는 과정을 실시간으로 관전.
- **체크포인트 개입**: 라운드 중간에 사용자가 추가 맥락을 주입하여 토론 방향 수정 가능.
- **토큰 절약 설계**: 
  - **빠른 토론 모드**: Round 1 후 바로 심판 단계로 이동 (API 호출 3회).
  - **엄격한 출력 제한**: 각 라운드별 `max_tokens` 강제 및 프롬프트 최적화.
- **비용 모니터링**: 세션별 예상 비용 및 사용 토큰 수 UI 표시.

## 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정
`.env` 파일을 생성하고 API 키를 입력하세요.
```env
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
OPENAI_API_KEY=optional_openai_key_for_judge
```

### 3. 서버 실행
```bash
uvicorn app.main:app --reload
```
- 접속 URL: `http://127.0.0.1:8000`

### 4. IntelliJ Run Configuration (선택)
- **Module**: `uvicorn`
- **Parameters**: `app.main:app --reload`
- **Environment variables**: `.env` 파일 경로 지정 또는 수동 입력

## 설정 가이드

### profile.md 작성 팁
`profile.md`는 토론의 맥락을 결정하는 중요한 파일입니다. 1000자 이내로 본인의 전문성, 상황, 목표를 솔직하게 작성하세요. (서버 실행 시 자동으로 로드되지는 않으며, 필요시 UI에서 입력하거나 config를 통해 확장 가능)

### 모델 업데이트 안내
`app/config.py`에서 최신 모델명을 관리합니다.
- 토론자 A: `claude-3-5-sonnet-20241022` (기본값)
- 토론자 B: `gemini-2.0-pro-exp` (최신 모델명 확인 권장)
- 심판: `claude-3-opus-20240229` (OpenAI 키 없을 시)

## 비용 안내 (대략치)
- **빠른 토론**: 약 $0.05 ~ $0.15 (입력 컨텍스트에 따라 상이)
- **풀 토론**: 약 $0.10 ~ $0.25
- 출력 길이 제한(Max Tokens)이 걸려 있어 예상치 못한 대량 과금을 방지합니다.

## 주의 사항
- `profile.md`의 내용은 외부 API(Anthropic, Google)로 전송됩니다. 민감한 개인정보는 제외해 주세요.
- 모델명은 공급자 문서에서 최신 버전을 수시로 확인하시기 바랍니다.
