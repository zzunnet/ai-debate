"""
Async streaming wrappers for Claude (Anthropic) and Gemini (Google).
Each function accepts an async callback `on_token(text: str)` and returns a
StreamResult with the full text and token counts.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable

import anthropic
import google.generativeai as genai

from app.config import (
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    CLAUDE_DEBATER_MODEL,
    GEMINI_DEBATER_MODEL,
    JUDGE_MODEL_CLAUDE,
    JUDGE_MODEL_OPENAI,
    USE_OPENAI_JUDGE,
    LLM_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

# ── Client helpers (per-request, no global state) ─────────────────────────────

def _get_anthropic_client(api_key: str | None) -> anthropic.AsyncAnthropic | None:
    key = api_key or ANTHROPIC_API_KEY
    if not key:
        return None
    return anthropic.AsyncAnthropic(api_key=key)


def _configure_gemini(api_key: str | None) -> bool:
    key = api_key or GOOGLE_API_KEY
    if not key:
        return False
    genai.configure(api_key=key)
    return True


@dataclass
class StreamResult:
    text: str = ""
    tokens_input: int = 0
    tokens_output: int = 0
    error: str | None = None


TokenCallback = Callable[[str], Awaitable[None]]


# ── Claude ────────────────────────────────────────────────────────────────────

async def stream_claude(
    system: str,
    user: str,
    max_tokens: int,
    on_token: TokenCallback,
    model: str = CLAUDE_DEBATER_MODEL,
    api_key: str | None = None,
) -> StreamResult:
    client = _get_anthropic_client(api_key)
    if client is None:
        return StreamResult(error="ANTHROPIC_API_KEY not configured")

    result = StreamResult()

    async def _stream() -> None:
        async with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            async for text in stream.text_stream:
                result.text += text
                await on_token(text)
            final = await stream.get_final_message()
            result.tokens_input = final.usage.input_tokens
            result.tokens_output = final.usage.output_tokens

    try:
        await asyncio.wait_for(_stream(), timeout=LLM_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        result.error = f"Claude timeout after {LLM_TIMEOUT_SECONDS}s"
    except anthropic.APIStatusError as e:
        if e.status_code == 400 and "credit" in str(e).lower():
            result.error = "Claude API error: 크레딧이 부족합니다. Anthropic 콘솔에서 크레딧을 충전해주세요. (Credit balance too low)"
        elif e.status_code == 429:
            result.error = "Claude API error: 요청 한도 초과 (rate limit). 잠시 후 다시 시도해주세요."
        else:
            result.error = f"Claude API error: {e}"
    except anthropic.APIError as e:
        result.error = f"Claude API error: {e}"
    except Exception as e:
        result.error = f"Claude unexpected error: {e}"
        logger.exception("stream_claude failed")
    return result


# ── Gemini ────────────────────────────────────────────────────────────────────

async def stream_gemini(
    prompt: str,
    max_tokens: int,
    on_token: TokenCallback,
    model: str = GEMINI_DEBATER_MODEL,
    api_key: str | None = None,
) -> StreamResult:
    if not _configure_gemini(api_key):
        return StreamResult(error="GOOGLE_API_KEY not configured")

    loop = asyncio.get_event_loop()
    token_queue: asyncio.Queue[str | None] = asyncio.Queue()

    def on_token_sync(text: str) -> None:
        loop.call_soon_threadsafe(token_queue.put_nowait, text)

    # Run blocking Gemini call in thread
    future = loop.run_in_executor(
        None, _gemini_stream_in_thread_sync, prompt, max_tokens, model, on_token_sync, token_queue, loop
    )

    result = StreamResult()

    async def _drain() -> None:
        while True:
            token = await token_queue.get()
            if token is None:
                break
            result.text += token
            await on_token(token)
        thread_result = await future
        result.tokens_input = thread_result.tokens_input
        result.tokens_output = thread_result.tokens_output
        result.error = thread_result.error

    try:
        await asyncio.wait_for(_drain(), timeout=LLM_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        result.error = f"Gemini timeout after {LLM_TIMEOUT_SECONDS}s"
    except Exception as e:
        result.error = f"Gemini stream error: {e}"
    return result


_GEMINI_RETRY_DELAYS = [5, 15, 30]  # seconds between retries on 429


def _is_gemini_rate_limit(e: Exception) -> bool:
    msg = str(e).lower()
    return "429" in msg or "resource exhausted" in msg or "quota" in msg or "rate" in msg


def _gemini_stream_in_thread_sync(
    prompt: str,
    max_tokens: int,
    model_name: str,
    on_token_sync: Callable[[str], None],
    token_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
) -> StreamResult:
    """Sync version that signals completion via None sentinel in queue."""
    result = StreamResult()
    last_error: Exception | None = None

    for attempt, delay in enumerate([0] + _GEMINI_RETRY_DELAYS):
        if delay > 0:
            logger.warning("Gemini 429 rate limit, retrying in %ds (attempt %d)...", delay, attempt + 1)
            time.sleep(delay)
        try:
            model = genai.GenerativeModel(model_name)
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.7,
            )
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
            ]
            response = model.generate_content(
                prompt,
                stream=True,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )
            for chunk in response:
                text = ""
                try:
                    text = chunk.text
                except Exception:
                    pass
                if text:
                    result.text += text
                    on_token_sync(text)

            try:
                meta = response.usage_metadata
                if meta:
                    result.tokens_input = getattr(meta, "prompt_token_count", 0) or 0
                    result.tokens_output = getattr(meta, "candidates_token_count", 0) or 0
            except Exception:
                result.tokens_input = len(prompt) // 4
                result.tokens_output = len(result.text) // 4

            last_error = None
            break  # success
        except Exception as e:
            last_error = e
            if _is_gemini_rate_limit(e):
                continue  # retry
            logger.exception("Gemini sync stream failed")
            result.error = f"Gemini error: {e}"
            loop.call_soon_threadsafe(token_queue.put_nowait, None)
            return result

    if last_error is not None:
        result.error = "Gemini error: 요청 한도 초과 (429 rate limit). 잠시 후 다시 시도해주세요."
        logger.error("Gemini rate limit exceeded after retries: %s", last_error)

    loop.call_soon_threadsafe(token_queue.put_nowait, None)
    return result


# ── OpenAI judge (optional) ───────────────────────────────────────────────────

async def stream_openai(
    system: str,
    user: str,
    max_tokens: int,
    on_token: TokenCallback,
    model: str = JUDGE_MODEL_OPENAI,
    api_key: str | None = None,
) -> StreamResult:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return StreamResult(error="openai package not installed")

    key = api_key or OPENAI_API_KEY
    if not key:
        return StreamResult(error="OPENAI_API_KEY not configured")

    client = AsyncOpenAI(api_key=key)
    result = StreamResult()

    async def _stream() -> None:
        async with client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=True,
            stream_options={"include_usage": True},
        ) as stream:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    result.text += text
                    await on_token(text)
                if chunk.usage:
                    result.tokens_input = chunk.usage.prompt_tokens
                    result.tokens_output = chunk.usage.completion_tokens

    try:
        await asyncio.wait_for(_stream(), timeout=LLM_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        result.error = f"OpenAI timeout after {LLM_TIMEOUT_SECONDS}s"
    except Exception as e:
        result.error = f"OpenAI error: {e}"
    return result


# ── Unified judge call ────────────────────────────────────────────────────────

async def stream_judge(
    system: str,
    user: str,
    max_tokens: int,
    on_token: TokenCallback,
    anthropic_key: str | None = None,
    openai_key: str | None = None,
) -> StreamResult:
    use_openai = bool(openai_key or OPENAI_API_KEY)
    if use_openai:
        result = await stream_openai(system, user, max_tokens, on_token, model=JUDGE_MODEL_OPENAI, api_key=openai_key)
        if result.error:
            logger.warning("OpenAI judge failed, falling back to Claude: %s", result.error)
            result = await stream_claude(system, user, max_tokens, on_token, model=JUDGE_MODEL_CLAUDE, api_key=anthropic_key)
        return result
    return await stream_claude(system, user, max_tokens, on_token, model=JUDGE_MODEL_CLAUDE, api_key=anthropic_key)
