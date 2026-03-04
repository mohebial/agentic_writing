"""
Unified Claude backend for all review types.

Replaces the three separate claude.py files with a single implementation.
All domain-specific behaviour comes from ReviewConfig.
"""

from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path
from typing import Callable

from review_engine.helpers import banner

# ── Anthropic SDK ────────────────────────────────────────────────────────────

try:
    import anthropic as _anthropic
    _ANTHROPIC_OK = True
except ImportError:
    _anthropic = None  # type: ignore[assignment]
    _ANTHROPIC_OK = False

# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
PDF_MIME = "application/pdf"

MODEL_FALLBACK_CHAIN = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-sonnet-3-7-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
]

MAX_TOKENS = 16384
_529_RETRY_DELAYS = [15, 30, 60]
_429_RETRY_DELAYS = [60, 120]


# ── Public API ───────────────────────────────────────────────────────────────

def validate_startup() -> None:
    """Check that anthropic is installed and ANTHROPIC_API_KEY is set."""
    if not _ANTHROPIC_OK:
        sys.exit(
            "Error: missing dependency 'anthropic'.\n"
            "Install it with: pip install anthropic"
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "Error: ANTHROPIC_API_KEY environment variable is not set.\n"
            "Export it with: export ANTHROPIC_API_KEY=sk-ant-..."
        )


def make_client() -> "_anthropic.Anthropic":
    """Create an Anthropic client from environment."""
    return _anthropic.Anthropic()


def encode_pdf(pdf_path: str) -> str:
    """Read and base64-encode a PDF file."""
    return base64.standard_b64encode(Path(pdf_path).read_bytes()).decode("ascii")


def build_content(
    prompt: str,
    pdf_base64: str | None = None,
    manuscript_text: str | None = None,
    revised_heading: str = "Document (Revised)",
) -> list[dict]:
    """
    Build the Claude content block list.

    Round 1 (pdf_base64 set):  PDF document block + prompt text block
    Round 2+ (text set):       combined text block (revised doc + prompt)
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
            {"type": "text", "text": prompt},
        ]
    combined = (
        f"## {revised_heading}\n\n{manuscript_text}\n\n{prompt}"
        if manuscript_text else prompt
    )
    return [{"type": "text", "text": combined}]


def stream_agent(
    client: "_anthropic.Anthropic",
    model: str,
    system: str,
    content: list[dict],
    label: str,
    on_chunk: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    """
    Stream a Claude response and return (full_text, model_used).

    If on_chunk is None, chunks are printed to stdout (CLI mode).
    If on_chunk is provided, each text chunk is passed to the callback
    (Streamlit mode).

    Retry logic:
      - 529 overloaded: retry same model up to 3 times with backoff
      - 429 rate limit:  retry with 60/120 s backoff
      - Other errors:    re-raise, then try next model in fallback chain
    """
    if on_chunk is None:
        on_chunk = lambda text: print(text, end="", flush=True)

    models_to_try = [model] + [m for m in MODEL_FALLBACK_CHAIN if m != model]
    last_exc: Exception | None = None

    for attempt_model in models_to_try:
        if attempt_model != model:
            msg = f"\n[Fallback] Switching -> '{attempt_model}'\n"
            on_chunk(msg)

        for retry_num in range(1 + len(_529_RETRY_DELAYS)):
            header = banner(f"{label}  [model: {attempt_model}]")
            on_chunk(header)
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
                        on_chunk(text)
                        collected.append(text)
                on_chunk("\n")
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

                if is_overloaded and retry_num < len(_529_RETRY_DELAYS):
                    wait = _529_RETRY_DELAYS[retry_num]
                    on_chunk(
                        f"\n[529] '{attempt_model}' is overloaded. "
                        f"Retrying in {wait} s... "
                        f"(attempt {retry_num + 1}/{len(_529_RETRY_DELAYS)})"
                    )
                    time.sleep(wait)
                    continue

                if is_rate_limit:
                    for i, wait in enumerate(_429_RETRY_DELAYS):
                        on_chunk(
                            f"\n[429] Rate limit hit for '{attempt_model}'. "
                            f"Waiting {wait} s... "
                            f"(attempt {i + 1}/{len(_429_RETRY_DELAYS)})"
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
                                    on_chunk(text)
                                    collected.append(text)
                            on_chunk("\n")
                            return "".join(collected), attempt_model
                        except Exception:
                            if i == len(_429_RETRY_DELAYS) - 1:
                                raise
                    break

                if is_overloaded:
                    more = attempt_model != models_to_try[-1]
                    on_chunk(
                        f"\n[Fallback] '{attempt_model}' still overloaded after retries."
                        + (" Trying next model..." if more else " No more fallbacks.")
                    )
                    last_exc = exc
                    break

                raise

            break

    raise RuntimeError(
        f"All models exhausted ({models_to_try}). Last: {last_exc}"
    ) from last_exc
