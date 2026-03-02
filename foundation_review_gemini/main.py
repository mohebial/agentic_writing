#!/usr/bin/env python3
"""
Multi-Agent Private Foundation Grant Review System — Gemini Edition

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

Powered by Google Gemini via the google-genai SDK.

Usage:
    python -m foundation_review proposal.pdf
    python -m foundation_review proposal.pdf --output review.md
    python -m foundation_review proposal.pdf --model gemini-2.5-flash

Requirements:
    pip install google-genai
    export GEMINI_API_KEY=AIza...
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

# ── Allow direct script execution: `python foundation_review_gemini/main.py` ──
# When run as a script (not via `python -m`), __package__ is None and relative
# imports fail. Fix by inserting the repo root onto sys.path and setting the
# package name before any relative imports are attempted.
if __name__ == "__main__" and __package__ in (None, ""):
    _pkg_root = Path(__file__).resolve().parent.parent
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))
    __package__ = "foundation_review_gemini"

# ── Import helpers ────────────────────────────────────────────────────────────

try:
    from .helpers import (
        banner,
        stream_agent,
        build_parts,
        format_critiques,
        convert_to_pdf,
        validate_startup,
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
    _HELPERS_IMPORT_ERROR = None
except ImportError as exc:
    _HELPERS_IMPORT_ERROR = exc
    banner = stream_agent = build_parts = None
    format_critiques = convert_to_pdf = validate_startup = load_prompt = None
    build_combined_reviewer_system = build_challenge_system = None
    parse_combined_critiques = merge_challenge_addenda = None
    parse_decision = extract_applicant_revision = build_revised_text = None
    COVER_LETTER_START = COVER_LETTER_END = NARRATIVE_START = NARRATIVE_END = None

# ── Gemini API imports ────────────────────────────────────────────────────────

try:
    from google import genai
    from google.genai import types
    _GENAI_IMPORT_ERROR = None
except ImportError as exc:
    genai = None
    types = None
    _GENAI_IMPORT_ERROR = exc

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gemini-2.0-flash"
PDF_MIME      = "application/pdf"

MODEL_FALLBACK_CHAIN = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview-06-17",
]

# ── Agent System Prompts ──────────────────────────────────────────────────────
# Edit .txt files in prompts/ and instructions/ to change agent behaviour.

COMBINED_REVIEWER_SYSTEM = build_combined_reviewer_system() if build_combined_reviewer_system else ""
CHALLENGE_SYSTEM         = build_challenge_system()          if build_challenge_system          else ""
PANEL_CHAIR_SYSTEM       = load_prompt("panel_chair")        if load_prompt                     else ""
APPLICANT_SYSTEM         = load_prompt("applicant")          if load_prompt                     else ""


# ── Main review orchestration ─────────────────────────────────────────────────

def run(pdf_path: str, model: str, output_path: str) -> None:
    """
    Run the single-pass foundation grant review.

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
        f"# Foundation Grant Review: {pdf.name}\n\n"
        f"Model: `{model}`  \nFile URI: `{file_uri}`\n"
    )

    try:
        # ── Step 1: Combined Reviewers ────────────────────────────────────────
        review_parts = build_parts(
            prompt=(
                "Review the grant proposal above. Produce all three "
                "reviewer critiques between their sentinel markers, "
                "following each role's instructions exactly."
            ),
            file_uri=file_uri,
        )
        combined_out, model = stream_agent(
            client, model, COMBINED_REVIEWER_SYSTEM, review_parts,
            "Review Panel (3 reviewers — combined)",
        )
        critiques = parse_combined_critiques(combined_out)
        for name, text in critiques.items():
            record(name, text)

        # ── Step 2: Independence Audit ────────────────────────────────────────
        challenge_prompt = (
            "Below are three reviewer critiques of the same grant proposal. "
            "Evaluate each for independence biases and produce addenda as instructed.\n\n"
            + format_critiques(critiques)
        )
        challenge_parts = build_parts(
            prompt=challenge_prompt,
            file_uri=file_uri,
        )
        challenge_out, model = stream_agent(
            client, model, CHALLENGE_SYSTEM, challenge_parts,
            "Independence Auditor — Challenge Pass",
        )
        critiques = merge_challenge_addenda(critiques, challenge_out)
        for name, text in critiques.items():
            if "### Independence Auditor" in text:
                record(f"{name} (with addendum)", text)

        critiques_text = format_critiques(critiques)

        # ── Step 3: Panel Chair Recommendation ───────────────────────────────
        chair_prompt = (
            "The following critiques have been submitted by the review panel "
            "for this grant proposal.\n\n"
            f"{critiques_text}\n\n"
            "Synthesize these into a Panel Recommendation Letter following "
            "your output format exactly."
        )
        chair_parts = build_parts(
            prompt=chair_prompt,
            file_uri=file_uri,
        )
        chair_out, model = stream_agent(
            client, model, PANEL_CHAIR_SYSTEM, chair_parts,
            "Panel Chair — Recommendation Letter",
        )
        record("Panel Chair — Recommendation Letter", chair_out, level=2)

        decision = parse_decision(chair_out)
        print(f"\n{'━'*72}")
        print(f"  Panel Decision: {decision}")
        print(f"{'━'*72}\n")
        log.append(f"**Panel Decision: {decision}**\n")

        # ── Step 4: Applicant Revision ────────────────────────────────────────
        applicant_prompt = (
            "You have received the review panel critiques and the Panel Chair's "
            "Recommendation Letter below. Revise your application accordingly.\n\n"
            f"{critiques_text}\n\n"
            f"Panel Chair Recommendation Letter:\n{chair_out}\n\n"
            "Follow your output format exactly, including the sentinel markers.\n"
            f"Place the cover letter between {COVER_LETTER_START} and {COVER_LETTER_END}.\n"
            f"Place the revised narrative between {NARRATIVE_START} and {NARRATIVE_END}."
        )
        applicant_parts = build_parts(
            prompt=applicant_prompt,
            file_uri=file_uri,
        )
        applicant_out, model = stream_agent(
            client, model, APPLICANT_SYSTEM, applicant_parts,
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
            print(sections["narrative"])
            print()

        record("Revised Application", revised_text, level=2)

    finally:
        # ── Cleanup ───────────────────────────────────────────────────────────
        try:
            client.files.delete(name=file_name)
            print(f"\nCleaned up remote file: {file_name}")
        except Exception:
            pass

        # ── Save output ───────────────────────────────────────────────────────
        Path(output_path).write_text("\n".join(log), encoding="utf-8")
        print(f"Output saved → {output_path}")

        # ── Convert to PDF ────────────────────────────────────────────────────
        pdf_out = str(Path(output_path).with_suffix(".pdf"))
        ok = convert_to_pdf(output_path, pdf_out)
        if ok:
            print(f"PDF saved    → {pdf_out}\n")
        else:
            print("[PDF] Could not export PDF. Install with: pip install reportlab\n")


# ── File picker ───────────────────────────────────────────────────────────────

def _pick_pdf() -> str:
    """Open a native file-picker dialog and return the chosen PDF path."""
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


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Private Foundation Grant Review System — Gemini Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                               # opens file picker
  %(prog)s proposal.pdf                  # review that PDF
  %(prog)s proposal.pdf --output out.md  # specify output file
  %(prog)s proposal.pdf --model gemini-2.5-flash
        """.strip(),
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        help="Path to the PDF grant proposal (omit to open a file picker)",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Output markdown file (default: <pdf_folder>/<name>_review.md)",
    )

    args = parser.parse_args()

    validate_startup()

    pdf_path = args.pdf or _pick_pdf()
    pdf_path = str(Path(pdf_path).resolve())

    if args.output:
        output_path = str(Path(args.output).resolve())
    else:
        p = Path(pdf_path)
        output_path = str(p.parent / f"{p.stem}_review.md")

    run(pdf_path, args.model, output_path)


if __name__ == "__main__":
    main()
