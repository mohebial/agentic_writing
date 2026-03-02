#!/usr/bin/env python3
"""
Helper functions for the Journal Peer Review system — Claude Edition.

Provides Claude-specific API layer (stream_agent, build_content,
validate_startup) and re-exports all domain logic (prompts, sentinels,
parsing) from journal_review_gemini.helpers so no prompts are duplicated.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ── Optional Anthropic import ─────────────────────────────────────────────────

try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None  # type: ignore[assignment]

# ── Re-export generic terminal / text utilities from NIH_review_gemini ────────

from NIH_review_gemini.helpers import (        # noqa: F401
    banner,
    extract_section,
    format_critiques,
    preprocess_markdown,
    convert_to_pdf,
)

# ── Re-export ALL domain logic from journal_review_gemini ─────────────────────
# Prompts, sentinel constants, and parsing functions are model-agnostic.

from journal_review_gemini.helpers import (    # noqa: F401
    _REVIEWER_SENTINELS,
    _CHALLENGE_SENTINELS,
    RESPONSE_LETTER_START,
    RESPONSE_LETTER_END,
    REVISED_MANUSCRIPT_START,
    REVISED_MANUSCRIPT_END,
    load_prompt,
    build_combined_reviewer_system,
    build_challenge_system,
    parse_combined_critiques,
    merge_challenge_addenda,
    parse_decision,
    extract_author_revision,
    build_revised_text,
)

# ── Configuration ─────────────────────────────────────────────────────────────

PDF_MIME = "application/pdf"
DEFAULT_MODEL = "claude-sonnet-4-20250514"

MODEL_FALLBACK_CHAIN = [
    "claude-sonnet-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
]

# Max tokens per response — reviews + manuscripts can be very long
MAX_TOKENS = 16384

# Retry delays (seconds) for 529 overloaded errors
_529_RETRY_DELAYS = [15, 30, 60]

# Retry delays (seconds) for 429 rate-limit errors
# (Claude's 429 is per-minute, not daily — short backoff usually suffices)
_429_RETRY_DELAYS = [60, 120]


# ── Startup Validation ────────────────────────────────────────────────────────

def validate_startup() -> None:
    """Check that anthropic is installed and ANTHROPIC_API_KEY is set."""
    if _anthropic is None:
        sys.exit(
            "Error: missing dependency 'anthropic'.\n"
            "Install it with: pip install anthropic"
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "Error: ANTHROPIC_API_KEY environment variable is not set.\n"
            "Export it with: export ANTHROPIC_API_KEY=sk-ant-..."
        )


# ── Content Building ──────────────────────────────────────────────────────────

def build_content(
    prompt: str,
    pdf_base64: str | None = None,
    manuscript_text: str | None = None,
) -> list[dict]:
    """
    Build the Claude content block list for a messages request.

    Round 1 (pdf_base64 set):  PDF document block + prompt text block
    Round 2+ (text set):       combined text block (manuscript + prompt)

    Returns a list of Claude content block dicts.
    """
    if pdf_base64:
        return [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": PDF_MIME,
                    "data": pdf_base64,
                },
            },
            {
                "type": "text",
                "text": prompt,
            },
        ]
    else:
        combined = (
            f"## Manuscript (Revised)\n\n{manuscript_text}\n\n{prompt}"
            if manuscript_text
            else prompt
        )
        return [{"type": "text", "text": combined}]


# ── Streaming Agent ───────────────────────────────────────────────────────────

def stream_agent(
    client: "_anthropic.Anthropic",
    model: str,
    system: str,
    content: list[dict],
    label: str,
) -> tuple[str, str]:
    """
    Stream a Claude response to stdout and return (full_text, model_used).

    Error handling:
    - 529 overloaded_error: retry same model up to 3 times with backoff
    - 429 rate_limit_error: retry with 60/120s backoff (per-minute limit)
    - Other errors: re-raise immediately
    """
    models_to_try: list[str] = [model] + [
        m for m in MODEL_FALLBACK_CHAIN if m != model
    ]

    last_exc: Exception | None = None

    for attempt_model in models_to_try:
        if attempt_model != model:
            print(f"\n[Fallback] Switching → '{attempt_model}'\n")

        # ── 529 retry loop ────────────────────────────────────────────────────
        for retry_num in range(1 + len(_529_RETRY_DELAYS)):
            print(banner(f"{label}  [model: {attempt_model}]"))
            collected: list[str] = []
            try:
                with client.messages.stream(
                    model=attempt_model,
                    system=system,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=MAX_TOKENS,
                    temperature=1.0,
                ) as stream:
                    for text in stream.text_stream:
                        print(text, end="", flush=True)
                        collected.append(text)
                print("\n")
                return "".join(collected), attempt_model

            except Exception as exc:  # noqa: BLE001
                exc_str = str(exc)
                status = getattr(exc, "status_code", None)

                is_overloaded = (
                    (_anthropic and isinstance(exc, _anthropic.OverloadedError))
                    or status == 529
                    or "529" in exc_str
                    or "overloaded" in exc_str.lower()
                )
                is_rate_limit = (
                    (_anthropic and isinstance(exc, _anthropic.RateLimitError))
                    or status == 429
                    or "429" in exc_str
                    or "rate_limit" in exc_str.lower()
                )

                # ── 529: retry same model with increasing delays ───────────
                if is_overloaded and retry_num < len(_529_RETRY_DELAYS):
                    wait = _529_RETRY_DELAYS[retry_num]
                    print(
                        f"\n[529] '{attempt_model}' is overloaded. "
                        f"Retrying in {wait} s… "
                        f"(attempt {retry_num + 1}/{len(_529_RETRY_DELAYS)})"
                    )
                    time.sleep(wait)
                    continue

                # ── 429: per-minute rate limit — retry with longer backoff ─
                if is_rate_limit:
                    for ri, wait in enumerate(_429_RETRY_DELAYS):
                        print(
                            f"\n[429] Rate limit hit on '{attempt_model}'. "
                            f"Retrying in {wait} s… "
                            f"(attempt {ri + 1}/{len(_429_RETRY_DELAYS)})"
                        )
                        time.sleep(wait)
                        try:
                            collected = []
                            with client.messages.stream(
                                model=attempt_model,
                                system=system,
                                messages=[{"role": "user", "content": content}],
                                max_tokens=MAX_TOKENS,
                                temperature=1.0,
                            ) as stream:
                                for text in stream.text_stream:
                                    print(text, end="", flush=True)
                                    collected.append(text)
                            print("\n")
                            return "".join(collected), attempt_model
                        except Exception as retry_exc:  # noqa: BLE001
                            if ri == len(_429_RETRY_DELAYS) - 1:
                                raise RuntimeError(
                                    f"Rate limit not resolved after retries. "
                                    f"Model: {attempt_model}. "
                                    f"Last error: {retry_exc}"
                                ) from retry_exc
                            exc = retry_exc  # noqa: PLW2901

                # ── 529 after all per-model retries exhausted ─────────────
                if is_overloaded:
                    more = attempt_model != models_to_try[-1]
                    print(
                        f"\n[529] '{attempt_model}' still overloaded after retries."
                        + (" Trying next model…" if more else " No more fallbacks.")
                    )
                    last_exc = exc
                    break  # exit retry loop → try next model

                raise  # non-retryable error

    raise RuntimeError(
        f"All models exhausted. Last error: {last_exc}"
    ) from last_exc
