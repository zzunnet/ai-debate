from __future__ import annotations

import asyncio
import json
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db, AsyncSessionLocal
from app.models import Session, Message
from app.schemas import CreateSessionRequest, ActionRequest, SessionSummary, SessionDetail
from app.debate import create_state, get_state, start_debate, submit_action, _push_async

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("/", response_model=dict)
async def create_session(
    req: CreateSessionRequest,
    db: AsyncSession = Depends(get_db)
):
    profile_content = req.profile
    if not profile_content:
        import os
        if os.path.exists("profile.md"):
            with open("profile.md", "r", encoding="utf-8") as f:
                profile_content = f.read()
        else:
            profile_content = "일반 사용자"

    session = Session(
        question=req.question,
        profile=profile_content,
        quick_mode=req.quick_mode,
        state="ROUND1",
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    # Start the background debate task
    state = create_state(
        session_id=session.id,
        question=session.question,
        profile=session.profile,
        quick_mode=session.quick_mode,
    )
    start_debate(state, lambda: AsyncSessionLocal())

    return {"id": session.id}


@router.get("/{session_id}/stream")
async def stream_session(session_id: str, request: Request):
    state = get_state(session_id)
    if not state:
        # If not in memory (e.g. server restarted), we can't stream live
        # but we could potentially replay from DB if needed.
        # For now, return 404 if not active.
        raise HTTPException(status_code=404, detail="Active session not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        # Send existing events first
        for event in state.events:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        # Wait for new events
        last_index = len(state.events)
        while not state.done or last_index < len(state.events):
            if await request.is_disconnected():
                break

            await state.new_event.wait()
            state.new_event.clear()

            while last_index < len(state.events):
                event = state.events[last_index]
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                last_index += 1

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{session_id}/action")
async def handle_action(session_id: str, req: ActionRequest):
    state = get_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Active session not found")

    if req.action == "stop":
        state.stop_requested = True
        if state.task:
            state.task.cancel()
        submit_action(state, {"action": "stop"})
        return {"status": "stopping"}

    submit_action(state, req.model_dump())
    return {"status": "ok"}


@router.get("/", response_model=list[SessionSummary])
async def list_sessions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Session).order_by(Session.created_at.desc()).limit(20)
    )
    sessions = result.scalars().all()
    return [
        SessionSummary(
            id=s.id,
            question=s.question,
            quick_mode=s.quick_mode,
            state=s.state,
            created_at=s.created_at.isoformat(),
            total_tokens_input=s.total_tokens_input,
            total_tokens_output=s.total_tokens_output,
            estimated_cost_usd=s.estimated_cost_usd,
        )
        for s in sessions
    ]


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session_detail(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Session).where(Session.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Fetch messages
    msg_result = await db.execute(
        select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
    )
    messages = msg_result.scalars().all()

    return SessionDetail(
        id=session.id,
        question=session.question,
        quick_mode=session.quick_mode,
        state=session.state,
        created_at=session.created_at.isoformat(),
        total_tokens_input=session.total_tokens_input,
        total_tokens_output=session.total_tokens_output,
        estimated_cost_usd=session.estimated_cost_usd,
        messages=[m.to_dict() for m in messages]
    )


@router.post("/{session_id}/export")
async def export_session(session_id: str, db: AsyncSession = Depends(get_db)):
    import os
    from app.config import EXPORTS_DIR

    result = await db.execute(
        select(Session).where(Session.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    msg_result = await db.execute(
        select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
    )
    messages = msg_result.scalars().all()

    os.makedirs(EXPORTS_DIR, exist_ok=True)
    filename = f"debate_{session_id}.md"
    filepath = os.path.join(EXPORTS_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# 토론 결과: {session.question}\n\n")
        f.write(f"- 일시: {session.created_at}\n")
        f.write(f"- 모드: {'빠름' if session.quick_mode else '전체'}\n")
        f.write(f"- 총 토큰: {session.total_tokens_input + session.total_tokens_output}\n")
        f.write(f"- 예상 비용: ${session.estimated_cost_usd:.4f}\n\n")
        f.write("---\n\n")

        for msg in messages:
            role = {
                "claude": "Claude (토론자 A)",
                "gemini": "Gemini (토론자 B)",
                "judge": "심판",
                "user": "사용자 개입"
            }.get(msg.model, msg.model)

            f.write(f"## {role}")
            if msg.round > 0:
                f.write(f" (Round {msg.round})")
            f.write("\n\n")
            f.write(msg.content)
            f.write("\n\n")

    return {"filename": filename, "path": filepath}
