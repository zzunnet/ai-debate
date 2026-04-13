"""
Central configuration — swap model names here when providers release updates.

⚠️  WARNING: Model names change frequently. Always verify against:
    - Anthropic: https://docs.anthropic.com/en/docs/about-claude/models
    - Google:    https://ai.google.dev/gemini-api/docs/models
    - OpenAI:    https://platform.openai.com/docs/models
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Model identifiers ──────────────────────────────────────────────────────────
CLAUDE_DEBATER_MODEL: str = os.getenv("CLAUDE_DEBATER_MODEL", "claude-sonnet-4-5")
GEMINI_DEBATER_MODEL: str = os.getenv("GEMINI_DEBATER_MODEL", "gemini-2.5-pro")
JUDGE_MODEL_CLAUDE: str = os.getenv("JUDGE_MODEL", "claude-opus-4-5")
JUDGE_MODEL_OPENAI: str = "gpt-4o"  # fallback label if gpt-5 is unavailable

# ── API keys ───────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

# Judge selection: prefer OpenAI if key is present
USE_OPENAI_JUDGE: bool = bool(OPENAI_API_KEY)
JUDGE_MODEL: str = JUDGE_MODEL_OPENAI if USE_OPENAI_JUDGE else JUDGE_MODEL_CLAUDE

# ── Token limits (output) ──────────────────────────────────────────────────────
ROUND1_MAX_TOKENS: int = 700
ROUND2_MAX_TOKENS: int = 600
JUDGE_MAX_TOKENS: int = 900

# ── Cost per 1 M tokens (USD) — approximate, verify with provider ──────────────
# Claude Sonnet 4.5
COST_CLAUDE_SONNET_IN: float = 3.0
COST_CLAUDE_SONNET_OUT: float = 15.0
# Gemini 2.5 Pro
COST_GEMINI_PRO_IN: float = 3.5
COST_GEMINI_PRO_OUT: float = 10.5
# Claude Opus 4.5
COST_CLAUDE_OPUS_IN: float = 15.0
COST_CLAUDE_OPUS_OUT: float = 75.0
# OpenAI (placeholder — update when gpt-5 pricing is published)
COST_OPENAI_IN: float = 10.0
COST_OPENAI_OUT: float = 30.0

# ── Misc ───────────────────────────────────────────────────────────────────────
LLM_TIMEOUT_SECONDS: int = 120
PROFILE_MAX_CHARS: int = 1000
DATABASE_URL: str = "sqlite+aiosqlite:///./debate.db"
EXPORTS_DIR: str = "exports"

# ── Startup validation ─────────────────────────────────────────────────────────
MISSING_KEYS: list[str] = []
if not ANTHROPIC_API_KEY:
    MISSING_KEYS.append("ANTHROPIC_API_KEY")
if not GOOGLE_API_KEY:
    MISSING_KEYS.append("GOOGLE_API_KEY")
