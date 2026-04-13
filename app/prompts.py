"""
Prompt templates — all prompts enforce strict output length and structure
to minimise token consumption and eliminate filler language.
"""

from __future__ import annotations

COMMON_PROHIBITIONS = """\
금지: 서문·결어 반복, 과도한 공감·아첨, "상황에 따라 다릅니다"류 회피성 문구,
같은 말 재강조, 지정 분량 초과. 핵심만, 마크다운 사용.\
"""

# ── Round 1 ───────────────────────────────────────────────────────────────────

ROUND1_SYSTEM = """\
당신은 날카롭고 실용적인 조언자입니다.
사용자의 프로필과 질문을 받아 아래 구조로만 답합니다.
출력은 500~600자 마크다운, 지정 분량 초과 절대 금지.
{prohibitions}
""".format(prohibitions=COMMON_PROHIBITIONS)

ROUND1_USER = """\
## 사용자 프로필
{profile}

## 질문
{question}

## 출력 구조 (이 형식만 사용, 각 섹션 헤더 유지)
### 핵심 권고
(1줄)

### 근거
- (최대 3개, 각 1줄)

### 리스크
- (최대 2개, 각 1줄)

### 사용자가 확인해야 할 변수
- (최대 2개, 각 1줄)
"""

# ── Round 2 (반박) ────────────────────────────────────────────────────────────

ROUND2_SYSTEM = """\
당신은 날카롭고 실용적인 조언자입니다.
상대방의 Round 1 답변을 비판적으로 검토하고 아래 구조로만 답합니다.
출력은 400~500자 마크다운, 지정 분량 초과 절대 금지.
{prohibitions}
""".format(prohibitions=COMMON_PROHIBITIONS)

ROUND2_USER = """\
## 사용자 프로필
{profile}

## 질문
{question}

## 나의 Round 1 핵심 권고 (1줄 요약)
{my_round1_headline}

## 상대방 Round 1 전문
{opponent_round1_full}
{injection_block}

## 출력 구조 (이 형식만 사용)
### 동의점
(1개, 1줄)

### 사실오류 지적
(없으면 "없음")

### 놓친 관점
- (1~2개)

### 내 입장 수정
(수정 있으면 기술, 없으면 "변경 없음")

### 조정된 권고
(1줄)
"""

INJECTION_BLOCK_TEMPLATE = """\

## [사용자 추가 맥락 - 우선순위 높음]
{comment}
※ 이 맥락을 반영해 이전 입장을 재조정할 것.
"""

# ── Judge ─────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """\
당신은 중립적이고 냉철한 심판입니다.
양측 토론과 사용자 개입을 분석하고 아래 구조로만 최종 권고를 내립니다.
출력은 700~800자 마크다운, 지정 분량 초과 절대 금지.
{prohibitions}
""".format(prohibitions=COMMON_PROHIBITIONS)

JUDGE_USER = """\
## 사용자 프로필
{profile}

## 질문
{question}

## 토론 전문
{debate_transcript}

## 출력 구조 (이 형식만 사용)
### 합의된 것
(2~3줄)

### 이견과 설득력 평가
(누가 왜 더 근거 있는가, 2~3줄)

### 최종 단일 권고
(사용자 상황 맞춤, 2~3줄)

### 사용자만 알 수 있는 변수
- (2~3개 — 판단 외주화 방지용 질문)

### 즉시 실행 가능한 다음 액션
- (1~3개)

### 다음 회고 권장 시점
(1줄)
"""


def build_round1_prompt(profile: str, question: str) -> tuple[str, str]:
    """Returns (system, user) prompts for Round 1."""
    return ROUND1_SYSTEM, ROUND1_USER.format(profile=profile, question=question)


def build_round2_prompt(
    profile: str,
    question: str,
    my_round1_headline: str,
    opponent_round1_full: str,
    user_injection: str | None = None,
) -> tuple[str, str]:
    """Returns (system, user) prompts for Round 2."""
    if user_injection:
        injection_block = INJECTION_BLOCK_TEMPLATE.format(comment=user_injection)
    else:
        injection_block = ""
    user = ROUND2_USER.format(
        profile=profile,
        question=question,
        my_round1_headline=my_round1_headline,
        opponent_round1_full=opponent_round1_full,
        injection_block=injection_block,
    )
    return ROUND2_SYSTEM, user


def build_judge_prompt(
    profile: str,
    question: str,
    debate_transcript: str,
) -> tuple[str, str]:
    """Returns (system, user) prompts for the Judge."""
    user = JUDGE_USER.format(
        profile=profile,
        question=question,
        debate_transcript=debate_transcript,
    )
    return JUDGE_SYSTEM, user


def extract_headline(round1_text: str) -> str:
    """
    Extract the '핵심 권고' line from a Round 1 response.
    Falls back to first non-empty line if section not found.
    """
    lines = round1_text.splitlines()
    capture_next = False
    for line in lines:
        stripped = line.strip()
        if "핵심 권고" in stripped:
            capture_next = True
            continue
        if capture_next and stripped and not stripped.startswith("#"):
            return stripped
    # Fallback
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:120]
    return round1_text[:120]
