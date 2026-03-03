#!/usr/bin/env python3
"""
Multi-Agent Private Foundation Grant Review — Claude Backend

Accepts a PDF grant proposal and runs a single-pass peer review through:
  - Scientific Reviewer   (deep domain expert, all four criteria)
  - Innovation Reviewer   (originality and transformative potential specialist)
  - Program Advisor       (foundation mission fit, budget, organizational capacity)
  - Independence Auditor  (detects echo bias and score anchoring)
  - Panel Chair           (synthesizes reviews into a recommendation letter)
  - Project Director      (applicant revision: cover letter + revised narrative)

Scoring follows a 5-point scale (5 = Exceptional, 1 = Poor).
High weight on Novelty & Innovation — foundations fund what conventional
agencies won't.
One pass only: reviews → panel chair recommendation → applicant revision.

The PDF is encoded as base64 and sent inline — no server upload required.

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
"""

from __future__ import annotations

import base64
import os
import sys
import argparse
import time
from pathlib import Path

# ── Allow direct execution: `python foundation_review/claude.py` ──────────────
if __name__ == "__main__" and __package__ in (None, ""):
    _pkg_root = Path(__file__).resolve().parent.parent
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))
    __package__ = "foundation_review"

# ── Helpers (model-agnostic domain logic) ────────────────────────────────────

try:
    from .helpers import (
        banner,
        format_critiques,
        convert_to_pdf,
        load_prompt,
        build_combined_reviewer_system,
        build_challenge_system,
        parse_combined_critiques,
        merge_challenge_addenda,
        parse_decision,
        extract_applicant_revision,
        build_revised_text,
        COVER_LETTER_START,
        COVER_LETTER_END,
        NARRATIVE_START,
        NARRATIVE_END,
    )
    _HELPERS_OK = True
except ImportError:
    _HELPERS_OK = False
    banner = format_critiques = convert_to_pdf = load_prompt = None
    build_combined_reviewer_system = build_challenge_system = None
    parse_combined_critiques = merge_challenge_addenda = None
    parse_decision = extract_applicant_revision = build_revised_text = None
    COVER_LETTER_START = COVER_LETTER_END = NARRATIVE_START = NARRATIVE_END = None

# ── Anthropic SDK ─────────────────────────────────────────────────────────────

try:
    import anthropic as _anthropic
    _ANTHROPIC_OK = True
except ImportError:
    _anthropic = None  # type: ignore[assignment]
    _ANTHROPIC_OK = False

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_MODEL = "claude-sonnet-4-20250514"
PDF_MIME      = "application/pdf"

MODEL_FALLBACK_CHAIN = [
    "claude-sonnet-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
]

MAX_TOKENS        = 16384
_529_RETRY_DELAYS = [15, 30, 60]
_429_RETRY_DELAYS = [60, 120]

# ── Module-level system prompts ───────────────────────────────────────────────

COMBINED_REVIEWER_SYSTEM = build_combined_reviewer_system() if build_combined_reviewer_system else ""
CHALLENGE_SYSTEM         = build_challenge_system()          if build_challenge_system          else ""
PANEL_CHAIR_SYSTEM       = load_prompt("panel_chair")        if load_prompt                     else ""
APPLICANT_SYSTEM         = load_prompt("applicant")          if load_prompt                     else ""


# ── Claude API layer ──────────────────────────────────────────────────────────

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


def build_content(
    prompt: str,
    pdf_base64: str,
) -> list[dict]:
    """
    Build the Claude content block list for a messages request.

    Foundation review is single-pass — the PDF is always sent as a
    base64-encoded document block alongside the prompt.
    """
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


def stream_agent(
    client: "_anthropic.Anthropic",
    model: str,
    system: str,
    content: list[dict],
    label: str,
) -> tuple[str, str]:
    """
    Stream a Claude response and return (full_text, model_used).

    - 529 overloaded: retry same model up to 3 times with backoff
    - 429 rate limit: retry with 60/120s backoff (per-minute, recoverable)
    - Other errors: re-raise
    """
    models_to_try = [model] + [m for m in MODEL_FALLBACK_CHAIN if m != model]
    last_exc: Exception | None = None

    for attempt_model in models_to_try:
        if attempt_model != model:
            print(f"\n[Fallback] Switching → '{attempt_model}'\n")

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
                status  = getattr(exc, "status_code", None)

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
                    print(
                        f"\n[529] '{attempt_model}' is overloaded. "
                        f"Retrying in {wait} s… "
                        f"(attempt {retry_num + 1}/{len(_529_RETRY_DELAYS)})"
                    )
                    time.sleep(wait)
                    continue

                if is_rate_limit:
                    for i, wait in enumerate(_429_RETRY_DELAYS):
                        print(
                            f"\n[429] Rate limit hit for '{attempt_model}'. "
                            f"Waiting {wait} s… (attempt {i + 1}/{len(_429_RETRY_DELAYS)})"
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
                        except Exception:
                            if i == len(_429_RETRY_DELAYS) - 1:
                                raise
                    break

                if is_overloaded:
                    more = attempt_model != models_to_try[-1]
                    print(
                        f"\n[Fallback] '{attempt_model}' still overloaded after retries."
                        + (" Trying next model…" if more else " No more fallbacks.")
                    )
                    last_exc = exc
                    break

                raise

            break

    raise RuntimeError(
        f"All models exhausted ({models_to_try}). Last: {last_exc}"
    ) from last_exc


# ── Orchestration ─────────────────────────────────────────────────────────────

def run(pdf_path: str, model: str, output_path: str) -> None:
    """
    Run the single-pass foundation grant review using Claude.

    Flow:
      1. Encode PDF as base64 (no server upload)
      2. Combined reviewers (3 roles, 1 API call)
      3. Independence audit / challenge pass (1 API call)
      4. Panel Chair recommendation letter (1 API call)
      5. Applicant revision — cover letter + revised narrative (1 API call)
      6. Save markdown + PDF

    Total: 4 API calls
    """
    client = _anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env
    log: list[str] = []

    def record(heading: str, content: str, level: int = 3) -> None:
        log.append(f"{'#' * level} {heading}\n\n{content}\n")

    # ── Encode PDF ────────────────────────────────────────────────────────────
    pdf = Path(pdf_path)
    if not pdf.exists():
        sys.exit(f"Error: file not found — {pdf_path}")

    print(banner(f"Loading {pdf.name}  [model: {model}]", "═"))
    pdf_base64 = base64.standard_b64encode(pdf.read_bytes()).decode("ascii")
    print(f"  Encoded {pdf.stat().st_size // 1024} KB → base64  OK\n")

    log.append(
        f"# Foundation Grant Review (Claude): {pdf.name}\n\nModel: `{model}`\n"
    )

    # ── Step 1: Combined Reviewers ────────────────────────────────────────────
    combined_out, model = stream_agent(
        client, model, COMBINED_REVIEWER_SYSTEM,
        build_content(
            "Review the grant proposal above. Produce all three "
            "reviewer critiques between their sentinel markers, "
            "following each role's instructions exactly.",
            pdf_base64=pdf_base64,
        ),
        "Review Panel (3 reviewers — combined)",
    )
    critiques = parse_combined_critiques(combined_out)
    for name, text in critiques.items():
        record(name, text)

    # ── Step 2: Independence Audit ────────────────────────────────────────────
    challenge_out, model = stream_agent(
        client, model, CHALLENGE_SYSTEM,
        build_content(
            "Below are three reviewer critiques of the same grant proposal. "
            "Evaluate each for independence biases and produce addenda as instructed.\n\n"
            + format_critiques(critiques),
            pdf_base64=pdf_base64,
        ),
        "Independence Auditor — Challenge Pass",
    )
    critiques = merge_challenge_addenda(critiques, challenge_out)
    for name, text in critiques.items():
        if "### Independence Auditor" in text:
            record(f"{name} (with addendum)", text)

    critiques_text = format_critiques(critiques)

    # ── Step 3: Panel Chair Recommendation ───────────────────────────────────
    chair_out, model = stream_agent(
        client, model, PANEL_CHAIR_SYSTEM,
        build_content(
            "The following critiques have been submitted by the review panel "
            "for this grant proposal.\n\n"
            f"{critiques_text}\n\n"
            "Synthesize these into a Panel Recommendation Letter following "
            "your output format exactly.",
            pdf_base64=pdf_base64,
        ),
        "Panel Chair — Recommendation Letter",
    )
    record("Panel Chair — Recommendation Letter", chair_out, level=2)

    decision = parse_decision(chair_out)
    print(f"\n{'━'*72}")
    print(f"  Panel Decision: {decision}")
    print(f"{'━'*72}\n")
    log.append(f"**Panel Decision: {decision}**\n")

    # ── Step 4: Applicant Revision ────────────────────────────────────────────
    applicant_out, model = stream_agent(
        client, model, APPLICANT_SYSTEM,
        build_content(
            "You have received the review panel critiques and the Panel Chair's "
            "Recommendation Letter below. Revise your application accordingly.\n\n"
            f"{critiques_text}\n\n"
            f"Panel Chair Recommendation Letter:\n{chair_out}\n\n"
            "Follow your output format exactly, including the sentinel markers.\n"
            f"Place the cover letter between {COVER_LETTER_START} and {COVER_LETTER_END}.\n"
            f"Place the revised narrative between {NARRATIVE_START} and {NARRATIVE_END}.",
            pdf_base64=pdf_base64,
        ),
        "Project Director — Revision",
    )
    record("Project Director — Response & Revision", applicant_out)

    # Extract and display revised sections
    sections = extract_applicant_revision(applicant_out)
    revised_text = build_revised_text(sections, fallback=applicant_out)

    print(banner("Revised Application", "─"))
    if sections.get("cover_letter"):
        print("── Cover Letter ──\n")
        print(sections["cover_letter"])
        print()
    if sections.get("narrative"):
        print("── Revised Narrative ──\n")
        print(sections["narrative"][:800])
        if len(sections["narrative"]) > 800:
            print(f"\n[… {len(sections['narrative']) - 800} more chars — see output file …]\n")
        print()

    record("Revised Application", revised_text, level=2)

    _save(log, output_path)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _save(log: list[str], output_path: str) -> None:
    Path(output_path).write_text("\n".join(log), encoding="utf-8")
    print(f"Output saved → {output_path}")
    pdf_out = str(Path(output_path).with_suffix(".pdf"))
    if convert_to_pdf(output_path, pdf_out):
        print(f"PDF saved    → {pdf_out}\n")
    else:
        print("[PDF] Could not export PDF. Install with: pip install reportlab\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _pick_pdf() -> str:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select PDF grant proposal",
        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
    )
    root.destroy()
    if not path:
        sys.exit("No file selected. Exiting.")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Private Foundation Grant Review — Claude Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s                                          # opens file picker\n"
            "  %(prog)s proposal.pdf                             # review that PDF\n"
            "  %(prog)s proposal.pdf --output review.md\n"
            "  %(prog)s proposal.pdf --model claude-opus-4-20250514"
        ),
    )
    parser.add_argument("pdf", nargs="?", help="PDF grant proposal (omit to open file picker)")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help=f"Claude model (default: {DEFAULT_MODEL})")
    parser.add_argument("--output", "-o", metavar="FILE",
                        help="Output markdown file (default: <pdf_stem>_review.md)")
    args = parser.parse_args()

    validate_startup()

    pdf_path = str(Path(args.pdf or _pick_pdf()).resolve())
    output_path = (
        str(Path(args.output).resolve()) if args.output
        else str(Path(pdf_path).parent / f"{Path(pdf_path).stem}_review.md")
    )
    run(pdf_path, args.model, output_path)


if __name__ == "__main__":
    main()
