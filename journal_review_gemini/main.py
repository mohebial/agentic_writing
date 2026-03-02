#!/usr/bin/env python3
"""
Multi-Agent Journal Peer Review System — Gemini Edition

Accepts a PDF manuscript and runs peer review through up to two rounds:

  Reviewers (Round 1 & 2):
    - Domain Expert          (deep field knowledge; novelty vs. literature)
    - Technical Reviewer     (methods, statistics, reproducibility, rigor)
    - Novelty & Impact Reviewer (originality, transformative potential,
                                 high-profile journal fit)
  Editor                    (synthesizes reviews into a decision letter)
  Author / PI               (point-by-point response + revised manuscript)

Round 1 decisions: Accept / Minor Revision / Major Revision
  - Accept  → Author produces a final polished manuscript. Done (4 API calls).
  - Minor/Major Revision → Author revises. Round 2 begins.

Round 2 decisions: Accept / Minor Revision / Reject
  - Author always produces a final version after Round 2. Done (8 API calls).

Powered by Google Gemini via the google-genai SDK.

Usage:
    python -m journal_review_gemini manuscript.pdf
    python -m journal_review_gemini manuscript.pdf --output review.md
    python -m journal_review_gemini manuscript.pdf --model gemini-2.5-flash

Requirements:
    pip install google-genai
    export GEMINI_API_KEY=AIza...
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

# ── Allow direct script execution: `python journal_review_gemini/main.py` ────
# When run as a script (not via `python -m`), __package__ is None and relative
# imports fail. Fix by inserting the repo root onto sys.path and setting the
# package name before any relative imports are attempted.
if __name__ == "__main__" and __package__ in (None, ""):
    _pkg_root = Path(__file__).resolve().parent.parent
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))
    __package__ = "journal_review_gemini"

# ── Import helpers from this package ─────────────────────────────────────────

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
        extract_author_revision,
        build_revised_text,
        RESPONSE_LETTER_START,
        RESPONSE_LETTER_END,
        REVISED_MANUSCRIPT_START,
        REVISED_MANUSCRIPT_END,
    )
    _HELPERS_IMPORT_ERROR = None
except ImportError as exc:
    _HELPERS_IMPORT_ERROR = exc
    banner = stream_agent = build_parts = format_critiques = None
    convert_to_pdf = validate_startup = load_prompt = None
    build_combined_reviewer_system = build_challenge_system = None
    parse_combined_critiques = merge_challenge_addenda = None
    parse_decision = extract_author_revision = build_revised_text = None
    RESPONSE_LETTER_START = RESPONSE_LETTER_END = None
    REVISED_MANUSCRIPT_START = REVISED_MANUSCRIPT_END = None

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
# Edit the .txt files in prompts/ and instructions/ to change agent behaviour.
# Guards handle the case where the package is run directly as a script rather
# than via `python -m journal_review_gemini` (relative imports would fail).

COMBINED_REVIEWER_SYSTEM = build_combined_reviewer_system() if build_combined_reviewer_system else ""
CHALLENGE_SYSTEM         = build_challenge_system()          if build_challenge_system          else ""
EDITOR_SYSTEM            = load_prompt("editor")             if load_prompt                     else ""
AUTHOR_SYSTEM            = load_prompt("author")             if load_prompt                     else ""


# ── Main review orchestration ─────────────────────────────────────────────────

def run(pdf_path: str, model: str, output_path: str) -> None:
    """
    Run the multi-agent journal peer review system.

    Orchestrates:
      Round 1: PDF upload → reviewers → editor decision
               Accept      → author polish                 → save (4 calls)
               Minor/Major → author revision → Round 2
      Round 2: revised text → reviewers → editor final decision
               → author final version                      → save (8 calls)
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
    uploaded = client.files.upload(
        file=pdf_path,
        config=types.UploadFileConfig(
            mime_type=PDF_MIME,
            display_name=pdf.name,
        ),
    )
    file_uri  = uploaded.uri
    file_name = uploaded.name
    print(f"  Uploaded → {file_uri}\n")

    log.append(
        f"# Journal Peer Review (Gemini): {pdf.name}\n\n"
        f"Model: `{model}`  \nFile URI: `{file_uri}`\n"
    )

    # ── After-round state ─────────────────────────────────────────────────────
    current_file_uri: str | None  = file_uri
    current_text:     str | None  = None   # holds revised manuscript after R1
    prev_editor_out:  str         = ""

    try:
        # ══════════════════════════════════════════════════════════════════════
        # ROUND 1
        # ══════════════════════════════════════════════════════════════════════
        print(banner("REVIEW ROUND 1 — Initial Submission", "═"))
        log.append("## Review Round 1 — Initial Submission\n")

        # ── 1A. Combined Reviewers ────────────────────────────────────────────
        review_parts = build_parts(
            prompt=(
                "Review the manuscript above. Produce all three reviewer critiques "
                "between their sentinel markers, following each role's instructions exactly."
            ),
            file_uri=current_file_uri,
            manuscript_text=current_text,
        )
        combined_out, model = stream_agent(
            client, model, COMBINED_REVIEWER_SYSTEM, review_parts,
            "Review Panel (3 reviewers — combined)",
        )
        critiques = parse_combined_critiques(combined_out)
        for name, text in critiques.items():
            record(name, text)

        # ── 1B. Independence Audit ────────────────────────────────────────────
        challenge_prompt = (
            "Below are three reviewer critiques of the same manuscript. "
            "Evaluate each for independence biases and produce addenda as instructed.\n\n"
            + format_critiques(critiques)
        )
        challenge_parts = build_parts(
            prompt=challenge_prompt,
            file_uri=current_file_uri,
            manuscript_text=current_text,
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

        # ── 1C. Editor Decision Letter ────────────────────────────────────────
        editor_prompt = (
            "The following three peer-reviewer critiques were submitted for this "
            "manuscript.\n\n"
            f"{critiques_text}\n\n"
            "Synthesize these into an official editorial decision letter following "
            "your output format. This is Round 1; valid decisions are Accept, "
            "Minor Revision, or Major Revision."
        )
        editor_parts = build_parts(
            prompt=editor_prompt,
            file_uri=current_file_uri,
            manuscript_text=current_text,
        )
        editor_out, model = stream_agent(
            client, model, EDITOR_SYSTEM, editor_parts,
            "Editor — Decision Letter (Round 1)",
        )
        prev_editor_out = editor_out
        record("Editorial Decision Letter (Round 1)", editor_out)

        decision = parse_decision(editor_out)
        print(f"\n{'━'*72}")
        print(f"  Editorial Decision (Round 1): {decision}")
        print(f"{'━'*72}\n")
        log.append(f"**Editorial Decision (Round 1): {decision}**\n")

        # ── 1D. Author Response ───────────────────────────────────────────────
        if decision == "Accept":
            # Accept: author polishes without needing a response letter.
            author_prompt = (
                "The editor has accepted your manuscript. Congratulations.\n\n"
                "Produce a final polished version of your manuscript. "
                "Apply only light copyediting, clarify any passages flagged "
                "by the reviewers, and correct any minor issues mentioned in "
                "the decision letter. No response letter is required.\n\n"
                "Reviewer feedback for reference:\n"
                f"{critiques_text}\n\n"
                f"Editorial Decision Letter:\n{editor_out}\n\n"
                f"Place the final manuscript between {REVISED_MANUSCRIPT_START} "
                f"and {REVISED_MANUSCRIPT_END}."
            )
        else:
            # Minor or Major Revision: full point-by-point response + revised manuscript.
            author_prompt = (
                "You have received the following reviewer critiques and editorial "
                "decision letter for your manuscript.\n\n"
                f"Reviewer Critiques:\n{critiques_text}\n\n"
                f"Editorial Decision Letter:\n{editor_out}\n\n"
                "Revise your manuscript accordingly and produce:\n"
                "  1. A point-by-point response letter addressing every concern "
                "in the decision letter and reviewer critiques.\n"
                "  2. A full revised manuscript.\n\n"
                "Follow your output format exactly, using the sentinel markers:\n"
                f"  Response letter: {RESPONSE_LETTER_START} ... {RESPONSE_LETTER_END}\n"
                f"  Revised manuscript: {REVISED_MANUSCRIPT_START} ... {REVISED_MANUSCRIPT_END}"
            )

        author_parts = build_parts(
            prompt=author_prompt,
            file_uri=current_file_uri,
            manuscript_text=current_text,
        )
        author_out, model = stream_agent(
            client, model, AUTHOR_SYSTEM, author_parts,
            f"Author — {'Final Polish' if decision == 'Accept' else 'Revision (Round 1)'}",
        )
        record(
            f"Author — {'Final Polished Manuscript' if decision == 'Accept' else 'Response & Revised Manuscript (Round 1)'}",
            author_out,
        )

        sections = extract_author_revision(author_out)
        current_text = build_revised_text(sections, fallback=author_out)
        current_file_uri = None  # subsequent calls use text, not the original PDF

        _print_revision(sections, rnd=1, decision=decision)

        # Accept path ends here
        if decision == "Accept":
            print(banner("✓ MANUSCRIPT ACCEPTED — Round 1", "═"))
            log.append(f"\n# Final Manuscript\n\n{current_text}\n")
            return  # jump to finally

        # ══════════════════════════════════════════════════════════════════════
        # ROUND 2
        # ══════════════════════════════════════════════════════════════════════
        print(banner(f"REVIEW ROUND 2 — Revised Submission ({decision})", "═"))
        log.append(f"## Review Round 2 — Revised Submission ({decision})\n")

        # ── 2A. Combined Reviewers (revised manuscript) ───────────────────────
        review_parts_r2 = build_parts(
            prompt=(
                "Review the REVISED manuscript below. Produce all three reviewer "
                "critiques between their sentinel markers, following each role's "
                "instructions exactly. Note that this is a revised submission — "
                "assess whether the authors have adequately addressed prior concerns."
            ),
            file_uri=current_file_uri,
            manuscript_text=current_text,
        )
        combined_out_r2, model = stream_agent(
            client, model, COMBINED_REVIEWER_SYSTEM, review_parts_r2,
            "Review Panel (Round 2 — 3 reviewers combined)",
        )
        critiques_r2 = parse_combined_critiques(combined_out_r2)
        for name, text in critiques_r2.items():
            record(f"{name} (Round 2)", text)

        # ── 2B. Independence Audit ────────────────────────────────────────────
        challenge_prompt_r2 = (
            "Below are three reviewer critiques (Round 2) of a revised manuscript. "
            "Evaluate each for independence biases and produce addenda as instructed.\n\n"
            + format_critiques(critiques_r2)
        )
        challenge_parts_r2 = build_parts(
            prompt=challenge_prompt_r2,
            file_uri=None,
            manuscript_text=current_text,
        )
        challenge_out_r2, model = stream_agent(
            client, model, CHALLENGE_SYSTEM, challenge_parts_r2,
            "Independence Auditor — Challenge Pass (Round 2)",
        )
        critiques_r2 = merge_challenge_addenda(critiques_r2, challenge_out_r2)
        for name, text in critiques_r2.items():
            if "### Independence Auditor" in text:
                record(f"{name} (Round 2, with addendum)", text)

        critiques_text_r2 = format_critiques(critiques_r2)

        # ── 2C. Editor Final Decision Letter ─────────────────────────────────
        editor_prompt_r2 = (
            "The following are Round 2 reviewer critiques of the revised manuscript. "
            "This is the final review round.\n\n"
            f"Round 1 Decision Letter (for context):\n{prev_editor_out}\n\n"
            f"Round 2 Reviewer Critiques:\n{critiques_text_r2}\n\n"
            "Synthesize these into a final editorial decision letter following your "
            "output format. This is Round 2 (final); valid decisions are Accept, "
            "Minor Revision, or Reject."
        )
        editor_parts_r2 = build_parts(
            prompt=editor_prompt_r2,
            file_uri=None,
            manuscript_text=current_text,
        )
        editor_out_r2, model = stream_agent(
            client, model, EDITOR_SYSTEM, editor_parts_r2,
            "Editor — Final Decision Letter (Round 2)",
        )
        record("Editorial Decision Letter (Round 2 — Final)", editor_out_r2)

        decision_r2 = parse_decision(editor_out_r2)
        print(f"\n{'━'*72}")
        print(f"  Editorial Decision (Round 2 — Final): {decision_r2}")
        print(f"{'━'*72}\n")
        log.append(f"**Editorial Decision (Round 2 — Final): {decision_r2}**\n")

        # ── 2D. Author Final Version ──────────────────────────────────────────
        if decision_r2 == "Reject":
            author_prompt_r2 = (
                "The editor has rejected your manuscript after Round 2 review. "
                "However, based on the feedback in the decision letter and reviewer "
                "critiques, produce a final fully revised version of your manuscript "
                "that addresses all concerns. This version will be available for "
                "submission to another journal or future resubmission.\n\n"
                f"Round 2 Reviewer Critiques:\n{critiques_text_r2}\n\n"
                f"Final Editorial Decision Letter:\n{editor_out_r2}\n\n"
                f"Place the final manuscript between {REVISED_MANUSCRIPT_START} "
                f"and {REVISED_MANUSCRIPT_END}. No response letter is required."
            )
        else:
            # Accept or Minor Revision after Round 2
            author_prompt_r2 = (
                "You have received the final editorial decision for your revised "
                "manuscript.\n\n"
                f"Round 2 Reviewer Critiques:\n{critiques_text_r2}\n\n"
                f"Final Editorial Decision Letter:\n{editor_out_r2}\n\n"
                "Produce:\n"
                "  1. A point-by-point response letter addressing any remaining "
                "concerns from the final decision letter.\n"
                "  2. A final revised manuscript incorporating all required changes.\n\n"
                f"  Response letter: {RESPONSE_LETTER_START} ... {RESPONSE_LETTER_END}\n"
                f"  Final manuscript: {REVISED_MANUSCRIPT_START} ... {REVISED_MANUSCRIPT_END}"
            )

        author_parts_r2 = build_parts(
            prompt=author_prompt_r2,
            file_uri=None,
            manuscript_text=current_text,
        )
        author_out_r2, model = stream_agent(
            client, model, AUTHOR_SYSTEM, author_parts_r2,
            f"Author — Final Version (Round 2 — {decision_r2})",
        )
        record(f"Author — Final Version (Round 2 — {decision_r2})", author_out_r2)

        sections_r2 = extract_author_revision(author_out_r2)
        final_text = build_revised_text(sections_r2, fallback=author_out_r2)

        _print_revision(sections_r2, rnd=2, decision=decision_r2)

        log.append(f"\n# Final Manuscript (after Round 2)\n\n{final_text}\n")

        if decision_r2 in ("Accept", "Minor Revision"):
            print(banner(f"✓ MANUSCRIPT {'ACCEPTED' if decision_r2 == 'Accept' else 'CONDITIONALLY ACCEPTED'} — Round 2", "═"))
        else:
            print(banner("✗ MANUSCRIPT REJECTED — Final version saved for resubmission", "═"))

    finally:
        # ── Cleanup ───────────────────────────────────────────────────────────
        try:
            client.files.delete(name=file_name)
            print(f"\nCleaned up remote file: {file_name}")
        except Exception:
            pass

        # ── Save markdown output ──────────────────────────────────────────────
        Path(output_path).write_text("\n".join(log), encoding="utf-8")
        print(f"Output saved → {output_path}")

        # ── Convert to PDF ────────────────────────────────────────────────────
        pdf_out = str(Path(output_path).with_suffix(".pdf"))
        ok = convert_to_pdf(output_path, pdf_out)
        if ok:
            print(f"PDF saved    → {pdf_out}\n")
        else:
            print(
                "[PDF] Could not export PDF. "
                "Install with: pip install reportlab\n"
            )


def _print_revision(
    sections: dict[str, str | None],
    rnd: int,
    decision: str,
) -> None:
    """Print a revision summary to stdout."""
    print(banner(f"Author Output — Round {rnd} ({decision})", "─"))
    if sections.get("response_letter"):
        print("── Response to Reviewers ──\n")
        print(sections["response_letter"][:1000])
        if len(sections["response_letter"]) > 1000:
            print(f"\n[... {len(sections['response_letter']) - 1000} more characters — see output file ...]\n")
        print()
    if sections.get("manuscript"):
        print("── Revised Manuscript (excerpt) ──\n")
        print(sections["manuscript"][:800])
        if len(sections["manuscript"]) > 800:
            print(f"\n[... {len(sections['manuscript']) - 800} more characters — see output file ...]\n")
        print()


# ── File picker ───────────────────────────────────────────────────────────────

def _pick_pdf() -> str:
    """Open a native file-picker dialog and return the chosen PDF path."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select PDF manuscript",
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
        description="Multi-Agent Journal Peer Review System — Gemini Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                # opens file picker
  %(prog)s manuscript.pdf                 # review that PDF
  %(prog)s manuscript.pdf --output r.md   # specify output file
  %(prog)s manuscript.pdf --model gemini-2.5-flash
        """.strip(),
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        help="Path to the PDF manuscript (omit to open a file picker)",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Output markdown file (default: <pdf_folder>/<name>_peer_review.md)",
    )

    args = parser.parse_args()

    validate_startup()

    pdf_path = args.pdf or _pick_pdf()
    pdf_path = str(Path(pdf_path).resolve())

    if args.output:
        output_path = str(Path(args.output).resolve())
    else:
        pdf = Path(pdf_path)
        output_path = str(pdf.parent / f"{pdf.stem}_peer_review.md")

    run(pdf_path, args.model, output_path)


if __name__ == "__main__":
    main()
