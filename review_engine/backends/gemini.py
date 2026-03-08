"""
Unified Gemini backend for all review types.

Replaces the three separate gemini.py files with a single implementation.
All domain-specific behaviour comes from ReviewConfig.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from review_engine.helpers import banner

# ── Gemini SDK ───────────────────────────────────────────────────────────────

try:
    from google import genai
    from google.genai import types
    _GENAI_OK = True
except ImportError:
    genai = types = None
    _GENAI_OK = False

# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gemini-2.0-flash"
PDF_MIME = "application/pdf"

DEFAULT_FALLBACK_CHAIN = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview-06-17",
]

_503_RETRY_DELAYS = [15, 30, 60]


# ── Public API ───────────────────────────────────────────────────────────────

class GeminiSetupError(RuntimeError):
    """Raised when the Gemini backend cannot be initialised."""


def validate_startup() -> None:
    """Check that google-genai is installed and GEMINI_API_KEY is set."""
    if not _GENAI_OK:
        raise GeminiSetupError(
            "Missing dependency 'google-genai'.\n"
            "Install it with: pip install google-genai"
        )
    if not os.environ.get("GEMINI_API_KEY"):
        raise GeminiSetupError(
            "GEMINI_API_KEY environment variable is not set.\n"
            "Export it with: export GEMINI_API_KEY=AIza..."
        )


def make_client() -> Any:
    """Create a Gemini client from environment."""
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def upload_pdf(client: Any, pdf_path: str) -> tuple[str, str]:
    """Upload a PDF and return (file_uri, file_name) for cleanup."""
    pdf = Path(pdf_path)
    uploaded = client.files.upload(
        file=str(pdf),
        config=types.UploadFileConfig(mime_type=PDF_MIME, display_name=pdf.name),
    )
    return uploaded.uri, uploaded.name


def cleanup_file(client: Any, file_name: str) -> None:
    """Delete an uploaded file from Gemini servers."""
    try:
        client.files.delete(name=file_name)
    except Exception:
        pass


def build_parts(
    prompt: str,
    file_uri: str | None = None,
    manuscript_text: str | None = None,
    revised_heading: str = "Document (Revised)",
) -> list[Any]:
    """
    Build the Part list for a Gemini request.

    Round 1 (file_uri set):  PDF part + prompt text part
    Round 2+ (text set):     combined text part
    """
    if file_uri:
        return [
            types.Part.from_uri(file_uri=file_uri, mime_type=PDF_MIME),
            types.Part.from_text(text=prompt),
        ]
    combined = f"## {revised_heading}\n\n{manuscript_text}\n\n{prompt}"
    return [types.Part.from_text(text=combined)]


def stream_agent(
    client: Any,
    model: str,
    system: str,
    parts: list,
    label: str,
    fallback_chain: list[str] | None = None,
    on_chunk: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    """
    Stream a Gemini response and return (full_text, model_used).

    Retry logic:
      - 503 UNAVAILABLE:        retry same model up to 3 times
      - 429 RESOURCE_EXHAUSTED:  raise immediately (daily quota)
      - Other errors:            re-raise, try next model
    """
    if on_chunk is None:
        on_chunk = lambda text: print(text, end="", flush=True)

    chain = fallback_chain or DEFAULT_FALLBACK_CHAIN
    models_to_try = [model] + [m for m in chain if m != model]
    last_exc: Exception | None = None

    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=1.0,
    )

    for attempt_model in models_to_try:
        if attempt_model != model:
            on_chunk(f"\n[Fallback] Switching -> '{attempt_model}'\n")

        for retry_num in range(1 + len(_503_RETRY_DELAYS)):
            header = banner(f"{label}  [model: {attempt_model}]")
            on_chunk(header)
            collected: list[str] = []
            try:
                for chunk in client.models.generate_content_stream(
                    model=attempt_model,
                    contents=[types.Content(parts=parts, role="user")],
                    config=config,
                ):
                    if chunk.text:
                        on_chunk(chunk.text)
                        collected.append(chunk.text)
                on_chunk("\n")
                return "".join(collected), attempt_model

            except Exception as exc:  # noqa: BLE001
                exc_str = str(exc)
                status = getattr(exc, "status_code", None)
                is_503 = status == 503 or "503" in exc_str or "UNAVAILABLE" in exc_str
                is_429 = status == 429 or "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str

                if is_503 and retry_num < len(_503_RETRY_DELAYS):
                    wait = _503_RETRY_DELAYS[retry_num]
                    on_chunk(
                        f"\n[503] '{attempt_model}' is under high demand. "
                        f"Retrying in {wait} s... "
                        f"(attempt {retry_num + 1}/{len(_503_RETRY_DELAYS)})"
                    )
                    time.sleep(wait)
                    continue

                if is_429:
                    on_chunk(
                        f"\n[429] Daily request quota exhausted for '{attempt_model}'.\n"
                        "All free-tier models share the same RPD quota -- "
                        "no point trying fallbacks.\n"
                        "Wait until tomorrow or upgrade your API plan."
                    )
                    raise RuntimeError(
                        f"Daily quota exhausted (429). Model: {attempt_model}."
                    ) from exc

                if is_503:
                    more = attempt_model != models_to_try[-1]
                    on_chunk(
                        f"\n[Fallback] '{attempt_model}' still unavailable after retries."
                        + (" Trying next model..." if more else " No more fallbacks.")
                    )
                    last_exc = exc
                    break

                raise

            break

    raise RuntimeError(
        f"All models exhausted ({models_to_try}). Last: {last_exc}"
    ) from last_exc
