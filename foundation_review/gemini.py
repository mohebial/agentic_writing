#!/usr/bin/env python3
"""
Multi-Agent Private Foundation Grant Review — Gemini Backend

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

Requirements:
    pip install google-genai
    export GEMINI_API_KEY=AIza...
"""

from __future__ import annotations

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Any

# ── Allow direct execution: `python foundation_review/gemini.py` ──────────────
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

# ── Gemini SDK ────────────────────────────────────────────────────────────────

try:
    from google import genai
    from google.genai import types
    _GENAI_OK = True
except ImportError:
    genai = types = None
    _GENAI_OK = False

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gemini-2.0-flash"
PDF_MIME      = "application/pdf"

MODEL_FALLBACK_CHAIN = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview-06-17",
]

_503_RETRY_DELAYS = [15, 30, 60]

# ── Module-level system prompts ───────────────────────────────────────────────

COMBINED_REVIEWER_SYSTEM = build_combined_reviewer_system() if build_combined_reviewer_system else ""
CHALLENGE_SYSTEM         = build_challenge_system()          if build_challenge_system          else ""
PANEL_CHAIR_SYSTEM       = load_prompt("panel_chair")        if load_prompt                     else ""
APPLICANT_SYSTEM         = load_prompt("applicant")          if load_prompt                     else ""


# ── Gemini API layer ──────────────────────────────────────────────────────────

def validate_startup() -> None:
    """Check that google-genai is installed and GEMINI_API_KEY is set."""
    if not _GENAI_OK:
        sys.exit(
            "Error: missing dependency 'google-genai'.\n"
            "Install it with: pip install google-genai"
        )
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit(
            "Error: GEMINI_API_KEY environment variable is not set.\n"
            "Export it with: export GEMINI_API_KEY=AIza..."
        )


def make_config(system: str) -> Any:
    """Create a Gemini GenerateContentConfig with a system instruction."""
    return types.GenerateContentConfig(
        system_instruction=system,
        temperature=1.0,
    )


def file_part(file_uri: str) -> Any:
    """Create a Gemini file part from a remote URI."""
    return types.Part.from_uri(file_uri=file_uri, mime_type=PDF_MIME)


def text_part(text: str) -> Any:
    """Create a Gemini text part."""
    return types.Part.from_text(text=text)


def build_parts(
    prompt: str,
    file_uri: str | None = None,
) -> list[Any]:
    """Build the Part list for a Gemini request (single-pass: always uses file_uri)."""
    return [file_part(file_uri), text_part(prompt)]


def stream_agent(
    client: Any,
    model: str,
    system: str,
    parts: list,
    label: str,
) -> tuple[str, str]:
    """
    Stream a Gemini response and return (full_text, model_used).

    - 503 UNAVAILABLE: retry same model up to 3 times
    - 429 RESOURCE_EXHAUSTED: raise immediately (daily quota — no point retrying)
    - Other errors: re-raise
    """
    models_to_try = [model] + [m for m in MODEL_FALLBACK_CHAIN if m != model]
    last_exc: Exception | None = None

    for attempt_model in models_to_try:
        if attempt_model != model:
            print(f"\n[Fallback] Switching → '{attempt_model}'\n")

        for retry_num in range(1 + len(_503_RETRY_DELAYS)):
            print(banner(f"{label}  [model: {attempt_model}]"))
            collected: list[str] = []
            try:
                for chunk in client.models.generate_content_stream(
                    model=attempt_model,
                    contents=[types.Content(parts=parts, role="user")],
                    config=make_config(system),
                ):
                    if chunk.text:
                        print(chunk.text, end="", flush=True)
                        collected.append(chunk.text)
                print("\n")
                return "".join(collected), attempt_model

            except Exception as exc:  # noqa: BLE001
                exc_str = str(exc)
                status  = getattr(exc, "status_code", None)
                is_503  = status == 503 or "503" in exc_str or "UNAVAILABLE" in exc_str
                is_429  = status == 429 or "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str

                if is_503 and retry_num < len(_503_RETRY_DELAYS):
                    wait = _503_RETRY_DELAYS[retry_num]
                    print(
                        f"\n[503] '{attempt_model}' is under high demand. "
                        f"Retrying in {wait} s… (attempt {retry_num + 1}/{len(_503_RETRY_DELAYS)})"
                    )
                    time.sleep(wait)
                    continue

                if is_429:
                    print(
                        f"\n[429] Daily request quota exhausted for '{attempt_model}'.\n"
                        "All free-tier models share the same RPD quota — "
                        "no point trying fallbacks.\n"
                        "Wait until tomorrow or upgrade your API plan."
                    )
                    raise RuntimeError(
                        f"Daily quota exhausted (429). Model: {attempt_model}."
                    ) from exc

                if is_503:
                    more = attempt_model != models_to_try[-1]
                    print(
                        f"\n[Fallback] '{attempt_model}' still unavailable after retries."
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
    Run the single-pass foundation grant review using Gemini.

    Flow:
      1. Upload PDF
      2. Combined reviewers (3 roles, 1 API call)
      3. Independence audit / challenge pass (1 API call)
      4. Panel Chair recommendation letter (1 API call)
      5. Applicant revision — cover letter + revised narrative (1 API call)
      6. Save markdown + PDF

    Total: 4 API calls (+ 1 file upload, 1 file delete)
    """
    api_key = os.environ["GEMINI_API_KEY"]
    client  = genai.Client(api_key=api_key)
    log: list[str] = []

    def record(heading: str, content: str, level: int = 3) -> None:
        log.append(f"{'#' * level} {heading}\n\n{content}\n")

    # ── Upload PDF ────────────────────────────────────────────────────────────
    pdf = Path(pdf_path)
    if not pdf.exists():
        sys.exit(f"Error: file not found — {pdf_path}")

    print(banner(f"Uploading {pdf.name}  [model: {model}]", "═"))
    uploaded  = client.files.upload(
        file=pdf_path,
        config=types.UploadFileConfig(mime_type=PDF_MIME, display_name=pdf.name),
    )
    file_uri  = uploaded.uri
    file_name = uploaded.name
    print(f"  Uploaded → {file_uri}\n")

    log.append(
        f"# Foundation Grant Review (Gemini): {pdf.name}\n\n"
        f"Model: `{model}`  \nFile URI: `{file_uri}`\n"
    )

    try:
        # ── Step 1: Combined Reviewers ────────────────────────────────────────
        combined_out, model = stream_agent(
            client, model, COMBINED_REVIEWER_SYSTEM,
            build_parts(
                "Review the grant proposal above. Produce all three "
                "reviewer critiques between their sentinel markers, "
                "following each role's instructions exactly.",
                file_uri=file_uri,
            ),
            "Review Panel (3 reviewers — combined)",
        )
        critiques = parse_combined_critiques(combined_out)
        for name, text in critiques.items():
            record(name, text)

        # ── Step 2: Independence Audit ────────────────────────────────────────
        challenge_out, model = stream_agent(
            client, model, CHALLENGE_SYSTEM,
            build_parts(
                "Below are three reviewer critiques of the same grant proposal. "
                "Evaluate each for independence biases and produce addenda as instructed.\n\n"
                + format_critiques(critiques),
                file_uri=file_uri,
            ),
            "Independence Auditor — Challenge Pass",
        )
        critiques = merge_challenge_addenda(critiques, challenge_out)
        for name, text in critiques.items():
            if "### Independence Auditor" in text:
                record(f"{name} (with addendum)", text)

        critiques_text = format_critiques(critiques)

        # ── Step 3: Panel Chair Recommendation ───────────────────────────────
        chair_out, model = stream_agent(
            client, model, PANEL_CHAIR_SYSTEM,
            build_parts(
                "The following critiques have been submitted by the review panel "
                "for this grant proposal.\n\n"
                f"{critiques_text}\n\n"
                "Synthesize these into a Panel Recommendation Letter following "
                "your output format exactly.",
                file_uri=file_uri,
            ),
            "Panel Chair — Recommendation Letter",
        )
        record("Panel Chair — Recommendation Letter", chair_out, level=2)

        decision = parse_decision(chair_out)
        print(f"\n{'━'*72}")
        print(f"  Panel Decision: {decision}")
        print(f"{'━'*72}\n")
        log.append(f"**Panel Decision: {decision}**\n")

        # ── Step 4: Applicant Revision ────────────────────────────────────────
        applicant_out, model = stream_agent(
            client, model, APPLICANT_SYSTEM,
            build_parts(
                "You have received the review panel critiques and the Panel Chair's "
                "Recommendation Letter below. Revise your application accordingly.\n\n"
                f"{critiques_text}\n\n"
                f"Panel Chair Recommendation Letter:\n{chair_out}\n\n"
                "Follow your output format exactly, including the sentinel markers.\n"
                f"Place the cover letter between {COVER_LETTER_START} and {COVER_LETTER_END}.\n"
                f"Place the revised narrative between {NARRATIVE_START} and {NARRATIVE_END}.",
                file_uri=file_uri,
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

    finally:
        # ── Cleanup ───────────────────────────────────────────────────────────
        try:
            client.files.delete(name=file_name)
            print(f"\nCleaned up remote file: {file_name}")
        except Exception:
            pass
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
        description="Multi-Agent Private Foundation Grant Review — Gemini Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s                                      # opens file picker\n"
            "  %(prog)s proposal.pdf                         # review that PDF\n"
            "  %(prog)s proposal.pdf --output review.md\n"
            "  %(prog)s proposal.pdf --model gemini-2.5-flash"
        ),
    )
    parser.add_argument("pdf", nargs="?", help="PDF grant proposal (omit to open file picker)")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help=f"Gemini model (default: {DEFAULT_MODEL})")
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
