from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000)
    quick_mode: bool = True  # default: Round1 + Judge only (3 LLM calls)
    profile: Optional[str] = Field(None, max_length=2000)


class ActionRequest(BaseModel):
    action: Literal["continue", "inject", "skip_to_judge", "stop"]
    comment: Optional[str] = Field(None, max_length=1000)


class SessionSummary(BaseModel):
    id: str
    question: str
    quick_mode: bool
    state: str
    created_at: str
    total_tokens_input: int
    total_tokens_output: int
    estimated_cost_usd: float


class SessionDetail(SessionSummary):
    messages: list[dict]
