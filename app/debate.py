"""
Debate state machine.

State flow:
  IDLE → ROUND1 → CHECKPOINT1 → (ROUND2 → CHECKPOINT2)? → JUDGING → DONE

Quick mode skips ROUND2 and CHECKPOINT2.
Each CHECKPOINT waits for a user action via POST /sessions/{id}/action.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app import config as cfg
from app.llm import stream_claude, stream_gemini, stream_judge, StreamResult
from app.models import Message, Session
from app.prompts import (
    build_round1_prompt,
    build_round2_prompt,
    build_judge_prompt,
    extract_headline,
)

logger = logging.getLogger(__name__)

# ── Cost calculation ───────────────────────────────────────────────────────────

def _cost(model_key: str, in_tok: int, out_tok: int) -> float:
    table = {
        "claude": (cfg.COST_CLAUDE_SONNET_IN, cfg.COST_CLAUDE_SONNET_OUT),
        "gemini": (cfg.COST_GEMINI_PRO_IN, cfg.COST_GEMINI_PRO_OUT),
        "judge_claude": (cfg.COST_CLAUDE_OPUS_IN, cfg.COST_CLAUDE_OPUS_OUT),
        "judge_openai": (cfg.COST_OPENAI_IN, cfg.COST_OPENAI_OUT),
    }
    in_rate, out_rate = table.get(model_key, (0.0, 0.0))
    return (in_tok * in_rate + out_tok * out_rate) / 1_000_000


# ── In-memory session state ────────────────────────────────────────────────────

@dataclass
class DebateState:
    session_id: str
    question: str
    profile: str
    quick_mode: bool

    # SSE event log — append-only, clients replay from index 0
    events: list[dict] = field(default_factory=list)
    new_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Checkpoint synchronisation
    action_event: asyncio.Event = field(default_factory=asyncio.Event)
    pending_action: dict | None = None

    # Round outputs
    claude_r1: str = ""
    gemini_r1: str = ""
    claude_r2: str = ""
    gemini_r2: str = ""
    judge_text: str = ""

    # Token accumulators
    total_in: int = 0
    total_out: int = 0
    cost_usd: float = 0.0

    # Control flags
    stop_requested: bool = False
    done: bool = False
    task: asyncio.Task | None = None


# Global registry: session_id → DebateState
_states: dict[str, DebateState] = {}


def get_state(session_id: str) -> DebateState | None:
    return _states.get(session_id)


def create_state(
    session_id: str, question: str, profile: str, quick_mode: bool
) -> DebateState:
    state = DebateState(
        session_id=session_id,
        question=question,
        profile=profile,
        quick_mode=quick_mode,
    )
    _states[session_id] = state
    return state


def cleanup_state(session_id: str) -> None:
    _states.pop(session_id, None)


# ── Event helpers ──────────────────────────────────────────────────────────────

def _push(state: DebateState, event: dict) -> None:
    state.events.append(event)
    state.new_event.set()
    state.new_event.clear()
    # Re-set so waiting coroutines wake up
    state.new_event.set()


async def _push_async(state: DebateState, event: dict) -> None:
    _push(state, event)
    await asyncio.sleep(0)  # yield to let SSE readers flush


# ── Core round runners ─────────────────────────────────────────────────────────

async def _run_round1_model(
    state: DebateState,
    model_key: str,  # "claude" | "gemini"
    db: AsyncSession,
) -> str:
    system, user = build_round1_prompt(state.profile, state.question)
    display_model = cfg.CLAUDE_DEBATER_MODEL if model_key == "claude" else cfg.GEMINI_DEBATER_MODEL

    await _push_async(state, {"type": "round_start", "round": 1, "model": model_key})

    async def on_token(text: str) -> None:
        _push(state, {"type": "token", "model": model_key, "text": text})

    if model_key == "claude":
        result: StreamResult = await stream_claude(system, user, cfg.ROUND1_MAX_TOKENS, on_token)
    else:
        prompt = f"{system}\n\n{user}"
        result = await stream_gemini(prompt, cfg.ROUND1_MAX_TOKENS, on_token)

    if result.error:
        await _push_async(state, {"type": "error", "model": model_key, "message": result.error})

    state.total_in += result.tokens_input
    state.total_out += result.tokens_output
    state.cost_usd += _cost(model_key, result.tokens_input, result.tokens_output)

    await _push_async(state, {
        "type": "round_complete",
        "round": 1,
        "model": model_key,
        "tokens_used": result.tokens_output,
    })

    # Persist to DB
    msg = Message(
        session_id=state.session_id,
        round=1,
        model=model_key,
        content=result.text or f"[Error: {result.error}]",
        tokens_input=result.tokens_input,
        tokens_output=result.tokens_output,
    )
    db.add(msg)
    await db.commit()

    return result.text


async def _run_round2_model(
    state: DebateState,
    model_key: str,
    opponent_r1: str,
    user_injection: str | None,
    db: AsyncSession,
) -> str:
    my_r1 = state.claude_r1 if model_key == "claude" else state.gemini_r1
    my_headline = extract_headline(my_r1)
    system, user = build_round2_prompt(
        state.profile,
        state.question,
        my_headline,
        opponent_r1,
        user_injection,
    )

    await _push_async(state, {"type": "round_start", "round": 2, "model": model_key})

    async def on_token(text: str) -> None:
        _push(state, {"type": "token", "model": model_key, "text": text})

    if model_key == "claude":
        result = await stream_claude(system, user, cfg.ROUND2_MAX_TOKENS, on_token)
    else:
        prompt = f"{system}\n\n{user}"
        result = await stream_gemini(prompt, cfg.ROUND2_MAX_TOKENS, on_token)

    if result.error:
        await _push_async(state, {"type": "error", "model": model_key, "message": result.error})

    state.total_in += result.tokens_input
    state.total_out += result.tokens_output
    state.cost_usd += _cost(model_key, result.tokens_input, result.tokens_output)

    await _push_async(state, {
        "type": "round_complete",
        "round": 2,
        "model": model_key,
        "tokens_used": result.tokens_output,
    })

    msg = Message(
        session_id=state.session_id,
        round=2,
        model=model_key,
        content=result.text or f"[Error: {result.error}]",
        tokens_input=result.tokens_input,
        tokens_output=result.tokens_output,
    )
    db.add(msg)
    await db.commit()

    return result.text


async def _run_judge(state: DebateState, db: AsyncSession) -> None:
    # Build transcript
    parts = []
    if state.claude_r1:
        parts.append(f"### Claude Round 1\n{state.claude_r1}")
    if state.gemini_r1:
        parts.append(f"### Gemini Round 1\n{state.gemini_r1}")
    if state.claude_r2:
        parts.append(f"### Claude Round 2\n{state.claude_r2}")
    if state.gemini_r2:
        parts.append(f"### Gemini Round 2\n{state.gemini_r2}")

    # Include user injections from events
    injections = [
        e for e in state.events if e.get("type") == "user_injection"
    ]
    for inj in injections:
        parts.append(f"### 사용자 추가 맥락\n{inj.get('comment', '')}")

    transcript = "\n\n".join(parts)
    system, user = build_judge_prompt(state.profile, state.question, transcript)

    await _push_async(state, {"type": "judge_start"})

    async def on_token(text: str) -> None:
        _push(state, {"type": "judge_token", "text": text})

    judge_key = "judge_openai" if cfg.USE_OPENAI_JUDGE else "judge_claude"
    result = await stream_judge(system, user, cfg.JUDGE_MAX_TOKENS, on_token)

    if result.error:
        await _push_async(state, {"type": "error", "model": "judge", "message": result.error})

    state.judge_text = result.text
    state.total_in += result.tokens_input
    state.total_out += result.tokens_output
    state.cost_usd += _cost(judge_key, result.tokens_input, result.tokens_output)

    await _push_async(state, {
        "type": "judge_complete",
        "tokens_used": result.tokens_output,
    })

    msg = Message(
        session_id=state.session_id,
        round=0,
        model="judge",
        content=result.text or f"[Error: {result.error}]",
        tokens_input=result.tokens_input,
        tokens_output=result.tokens_output,
    )
    db.add(msg)
    await db.commit()


# ── Checkpoint ─────────────────────────────────────────────────────────────────

async def _checkpoint(state: DebateState, after_round: int) -> dict:
    """Push checkpoint event and wait for user action. Returns the action dict."""
    options = ["continue", "inject", "skip_to_judge", "stop"]
    await _push_async(state, {
        "type": "checkpoint",
        "after_round": after_round,
        "options": options,
    })

    # Wait for action (with timeout guard — shouldn't be needed in practice)
    state.action_event.clear()
    state.pending_action = None
    await state.action_event.wait()

    action = state.pending_action or {"action": "continue"}
    return action


def submit_action(state: DebateState, action: dict) -> None:
    """Called by the action endpoint to unblock a waiting checkpoint."""
    state.pending_action = action
    state.action_event.set()

    # Record user injection in event log if present
    if action.get("action") == "inject" and action.get("comment"):
        _push(state, {
            "type": "user_injection",
            "comment": action["comment"],
        })


# ── Main debate orchestrator ───────────────────────────────────────────────────

async def run_debate(state: DebateState, db_session_factory) -> None:
    """
    Top-level coroutine that drives the full debate.
    Runs as a background asyncio Task.
    """
    async with db_session_factory() as db:
        try:
            # ── Round 1 ────────────────────────────────────────────────────────
            state.claude_r1 = await _run_round1_model(state, "claude", db)
            if state.stop_requested:
                return await _finish(state, db, "STOPPED")

            state.gemini_r1 = await _run_round1_model(state, "gemini", db)
            if state.stop_requested:
                return await _finish(state, db, "STOPPED")

            # ── Checkpoint 1 ───────────────────────────────────────────────────
            action1 = await _checkpoint(state, after_round=1)

            if action1["action"] == "stop":
                return await _finish(state, db, "STOPPED")

            user_injection1: str | None = None
            if action1["action"] == "inject":
                user_injection1 = action1.get("comment")
                # Save user injection to DB
                db.add(Message(
                    session_id=state.session_id,
                    round=-1,
                    model="user",
                    content=user_injection1 or "",
                    tokens_input=0,
                    tokens_output=0,
                ))
                await db.commit()

            skip_to_judge = (
                action1["action"] == "skip_to_judge"
                or state.quick_mode
            )

            if not skip_to_judge:
                # ── Round 2 ────────────────────────────────────────────────────
                state.claude_r2 = await _run_round2_model(
                    state, "claude", state.gemini_r1, user_injection1, db
                )
                if state.stop_requested:
                    return await _finish(state, db, "STOPPED")

                state.gemini_r2 = await _run_round2_model(
                    state, "gemini", state.claude_r1, user_injection1, db
                )
                if state.stop_requested:
                    return await _finish(state, db, "STOPPED")

                # ── Checkpoint 2 ───────────────────────────────────────────────
                action2 = await _checkpoint(state, after_round=2)

                if action2["action"] == "stop":
                    return await _finish(state, db, "STOPPED")

                if action2["action"] == "inject" and action2.get("comment"):
                    db.add(Message(
                        session_id=state.session_id,
                        round=-1,
                        model="user",
                        content=action2["comment"],
                        tokens_input=0,
                        tokens_output=0,
                    ))
                    await db.commit()

            # ── Judging ────────────────────────────────────────────────────────
            await _run_judge(state, db)

            await _finish(state, db, "DONE")

        except asyncio.CancelledError:
            await _finish(state, db, "STOPPED")
        except Exception as e:
            logger.exception("Debate error for session %s", state.session_id)
            await _push_async(state, {"type": "error", "message": str(e)})
            await _finish(state, db, "ERROR")


async def _finish(state: DebateState, db: AsyncSession, final_state: str) -> None:
    # Update session in DB
    from sqlalchemy import update
    await db.execute(
        update(Session)
        .where(Session.id == state.session_id)
        .values(
            state=final_state,
            finished_at=datetime.now(timezone.utc),
            total_tokens_input=state.total_in,
            total_tokens_output=state.total_out,
            estimated_cost_usd=state.cost_usd,
        )
    )
    await db.commit()

    # Push completion event
    await _push_async(state, {
        "type": "session_complete",
        "total_tokens_input": state.total_in,
        "total_tokens_output": state.total_out,
        "estimated_cost_usd": round(state.cost_usd, 6),
        "final_state": final_state,
    })
    state.done = True


def start_debate(
    state: DebateState,
    db_session_factory,
) -> asyncio.Task:
    """Schedule the debate coroutine and store the task reference."""
    task = asyncio.create_task(run_debate(state, db_session_factory))
    state.task = task
    return task
