#!/usr/bin/env python3
"""
Multi-Agent NIH Grant Peer Review System — Gemini Edition

Accepts a PDF grant application and runs iterative peer review through:
  - Primary Reviewer       (deep domain expertise, full written critique)
  - Secondary Reviewer     (complementary expertise, full written critique)
  - Tertiary Reviewer      (broad perspective, brief critique)
  - Biostatistics & Rigor Reviewer (methodology and statistical power)
  - Program Officer        (NIH mission alignment, portfolio fit)
  - Scientific Review Officer / SRO  (summary statement, fundability decision)
  - Principal Investigator / Author  (revision: Introduction + Research Strategy)

Scoring follows the NIH 9-point scale (1 = Exceptional, 9 = Poor).
Iteration stops when the SRO deems the application Fundable or max rounds
is reached.

Powered by Google Gemini via the google-genai SDK.

Usage:
    python grant_review_gemini.py proposal.pdf
    python grant_review_gemini.py proposal.pdf --max-rounds 3 --output summary.md
    python grant_review_gemini.py proposal.pdf --model gemini-2-flash

Requirements:
    pip install google-genai
    export GEMINI_API_KEY=AIza...
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

# ── Allow direct script execution: `python NIH_review_gemini/main.py` ────────
# When run as a script (not via `python -m`), __package__ is None and relative
# imports fail. Fix by inserting the repo root onto sys.path and setting the
# package name before any relative imports are attempted.
if __name__ == "__main__" and __package__ in (None, ""):
    _pkg_root = Path(__file__).resolve().parent.parent
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))
    __package__ = "NIH_review_gemini"

# ── Import helpers from submodule ────────────────────────────────────────────

try:
    from .helpers import (
        banner,
        stream_agent,
        build_parts,
        extract_revised_grant,
        build_revised_text,
        parse_decision,
        format_critiques,
        convert_to_pdf,
        validate_startup,
        load_prompt,
        build_combined_reviewer_system,
        build_challenge_system,
        parse_combined_critiques,
        merge_challenge_addenda,
    )
    _HELPERS_IMPORT_ERROR = None
except ImportError as exc:
    _HELPERS_IMPORT_ERROR = exc
    banner = stream_agent = build_parts = None
    extract_revised_grant = build_revised_text = parse_decision = None
    format_critiques = convert_to_pdf = validate_startup = load_prompt = None
    build_combined_reviewer_system = build_challenge_system = None
    parse_combined_critiques = merge_challenge_addenda = None

# ── Gemini API imports ───────────────────────────────────────────────────────

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
PDF_MIME = "application/pdf"

MODEL_FALLBACK_CHAIN = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview-06-17",
]

# Sentinels
AIMS_START  = "<<<SPECIFIC_AIMS_START>>>"
AIMS_END    = "<<<SPECIFIC_AIMS_END>>>"
STRAT_START = "<<<RESEARCH_STRATEGY_START>>>"
STRAT_END   = "<<<RESEARCH_STRATEGY_END>>>"
INTRO_START = "<<<INTRO_REVISED_APP_START>>>"
INTRO_END   = "<<<INTRO_REVISED_APP_END>>>"

# ── Agent System Prompts ──────────────────────────────────────────────────────
# Reviewer prompts are combined into a single API call (saves RPD quota).
# Edit the .txt files in prompts/ and instructions/ to change agent behaviour.

COMBINED_REVIEWER_SYSTEM = build_combined_reviewer_system() if build_combined_reviewer_system else ""
CHALLENGE_SYSTEM         = build_challenge_system()          if build_challenge_system          else ""
SRO_SYSTEM               = load_prompt("sro", scoring=True)  if load_prompt                     else ""
PI_SYSTEM                = load_prompt("pi")                 if load_prompt                     else ""

# ── Main review orchestration ──────────────────────────────────────────────────

def run(pdf_path: str, model: str, max_rounds: int, output_path: str) -> None:
    """
    Run the multi-agent NIH grant review system.
    
    Orchestrates: PDF upload → reviewers → SRO → PI revision loop.
    All implementation details are delegated to helpers module.
    """
    api_key = os.environ["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
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
        f"# NIH Grant Review (Gemini): {pdf.name}\n\n"
        f"Model: `{model}`  \nFile URI: `{file_uri}`\n"
    )

    # ── State management ──────────────────────────────────────────────────────
    current_file_uri: str | None = file_uri
    current_text: str | None = None
    prev_sro_out: str = ""

    try:
        for rnd in range(1, max_rounds + 1):
            print(banner(f"REVIEW ROUND {rnd} / {max_rounds}", "═"))
            log.append(f"## Review Round {rnd}\n")

            # ── Combined Reviewers (single API call) ─────────────────────────
            review_parts = build_parts(
                prompt=(
                    "Review the grant application above. Produce all five "
                    "reviewer critiques between their sentinel markers, "
                    "following each role's instructions exactly."
                ),
                file_uri=current_file_uri,
                manuscript_text=current_text,
            )
            combined_out, model = stream_agent(
                client, model, COMBINED_REVIEWER_SYSTEM, review_parts,
                "Review Panel (5 reviewers — combined)",
            )
            critiques = parse_combined_critiques(combined_out)
            for name, text in critiques.items():
                record(name, text)

            # ── Challenge / Independence Audit (single API call) ──────────────
            challenge_prompt = (
                "Below are five reviewer critiques of the same NIH grant "
                "application. Evaluate each for independence biases and "
                "produce addenda as instructed.\n\n"
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

            # ── Scientific Review Officer (SRO) ───────────────────────────────
            sro_prompt = (
                "The following critiques have been submitted by the review panel "
                "for this grant application.\n\n"
                f"{critiques_text}\n\n"
                "Synthesize these into an official NIH Summary Statement and issue "
                "your fundability decision following your output format."
            )
            if prev_sro_out:
                sro_prompt = (
                    f"Previous Summary Statement (prior submission):\n{prev_sro_out}\n\n"
                    + sro_prompt
                )

            sro_parts = build_parts(
                prompt=sro_prompt,
                file_uri=current_file_uri,
                manuscript_text=current_text,
            )
            sro_out, model = stream_agent(
                client, model, SRO_SYSTEM, sro_parts,
                "Scientific Review Officer (SRO) — Summary Statement",
            )
            prev_sro_out = sro_out
            record("Summary Statement (SRO)", sro_out)

            decision = parse_decision(sro_out)
            print(f"\n{'━'*72}")
            print(f"  SRO Decision: {decision}")
            print(f"{'━'*72}\n")
            log.append(f"**SRO Decision: {decision}**\n")

            # ── Terminal conditions ───────────────────────────────────────────
            if decision == "Fundable":
                print(banner("✓ APPLICATION DEEMED FUNDABLE", "═"))
                if current_text:
                    print(banner("FINAL REVISED APPLICATION", "─"))
                    print(current_text)
                    print()
                    log.append(f"\n# Final Revised Application\n\n{current_text}\n")
                else:
                    note = f"[Original PDF funded without revision: {pdf.name}]"
                    print(note)
                    log.append(f"\n# Final Revised Application\n\n{note}\n")
                break

            if decision == "NRFC":
                print(banner("✗ NOT RECOMMENDED FOR FURTHER CONSIDERATION (NRFC)", "═"))
                log.append(
                    f"\n**Application Not Recommended for Further Consideration "
                    f"after round {rnd}.**\n"
                )
                if current_text:
                    print(banner("Last Revised Application (NRFC)", "─"))
                    print(current_text)
                    print()
                    log.append(f"\n# Last Revised Application\n\n{current_text}\n")
                break

            if rnd == max_rounds:
                print(banner(f"MAX ROUNDS ({max_rounds}) REACHED — not yet fundable", "═"))
                log.append(
                    f"\n**Stopped: max rounds ({max_rounds}) reached. "
                    f"Final decision: {decision}**\n"
                )
                if current_text:
                    print(banner("Last Revised Application", "─"))
                    print(current_text)
                    print()
                    log.append(f"\n# Last Revised Application\n\n{current_text}\n")
                break

            # ── PI Revision ───────────────────────────────────────────────────
            pi_prompt = (
                "You have received the review panel critiques and the official "
                "Summary Statement below. Revise your application accordingly.\n\n"
                f"{critiques_text}\n\n"
                f"Summary Statement:\n{sro_out}\n\n"
                "Follow your output format exactly, including all sentinel markers.\n"
                f"Place the Introduction between {INTRO_START} and {INTRO_END}.\n"
                f"Place revised Specific Aims between {AIMS_START} and {AIMS_END}.\n"
                f"Place revised Research Strategy between {STRAT_START} and {STRAT_END}."
            )
            pi_parts = build_parts(
                prompt=pi_prompt,
                file_uri=current_file_uri,
                manuscript_text=current_text,
            )
            pi_out, model = stream_agent(
                client, model, PI_SYSTEM, pi_parts,
                "Principal Investigator — Revision",
            )
            record("PI Response & Revised Application", pi_out)

            # Prepare for next round
            markers = {
                "intro":    (INTRO_START, INTRO_END),
                "aims":     (AIMS_START,  AIMS_END),
                "strategy": (STRAT_START, STRAT_END),
            }
            sections = extract_revised_grant(pi_out, markers)
            current_text = build_revised_text(sections, fallback=pi_out)
            current_file_uri = None

            # Print revised sections
            print(banner(f"Revised Application — Round {rnd}", "─"))
            if sections.get("intro"):
                print("── Introduction to Revised Application ──\n")
                print(sections["intro"])
                print()
            if sections.get("aims"):
                print("── Specific Aims ──\n")
                print(sections["aims"])
                print()
            if sections.get("strategy"):
                print("── Research Strategy ──\n")
                print(sections["strategy"])
                print()
            record(f"Revised Application — Round {rnd}", current_text, level=2)

    finally:
        # ── Cleanup ────────────────────────────────────────────────────────────
        try:
            client.files.delete(name=file_name)
            print(f"\nCleaned up remote file: {file_name}")
        except Exception:
            pass

        # ── Save output ────────────────────────────────────────────────────────
        Path(output_path).write_text("\n".join(log), encoding="utf-8")
        print(f"Output saved → {output_path}")

        # ── Convert to PDF ────────────────────────────────────────────────────────
        pdf_out = str(Path(output_path).with_suffix(".pdf"))
        ok = convert_to_pdf(output_path, pdf_out)
        if ok:
            print(f"PDF saved    → {pdf_out}\n")
        else:
            print(
                "[PDF] Could not export PDF. "
                "Install with: pip install reportlab\n"
            )


# ── File picker ───────────────────────────────────────────────────────────────

def _pick_pdf() -> str:
    """Open a native file-picker dialog and return the chosen PDF path."""
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select PDF grant application",
        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
    )
    root.destroy()
    if not path:
        sys.exit("No file selected. Exiting.")
    return path


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent NIH Grant Peer Review System — Gemini Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                               # opens file picker
  %(prog)s proposal.pdf                  # review that PDF
  %(prog)s proposal.pdf --max-rounds 3   # max 3 rounds
  %(prog)s proposal.pdf --output out.md  # specify output file
        """.strip(),
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        help="Path to the PDF grant application (omit to open a file picker)",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-rounds", "-r",
        type=int,
        default=2,
        metavar="N",
        help="Maximum review rounds (default: 2; NIH allows only 1 resubmission)",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Output markdown file (default: <pdf_folder>/<name>_output.md)",
    )

    args = parser.parse_args()

    # Validate startup
    validate_startup()

    # Get PDF path
    pdf_path = args.pdf or _pick_pdf()
    pdf_path = str(Path(pdf_path).resolve())

    # Determine output path
    if args.output:
        output_path = str(Path(args.output).resolve())
    else:
        pdf = Path(pdf_path)
        output_path = str(pdf.parent / f"{pdf.stem}_output.md")

    # Run the review
    run(pdf_path, args.model, args.max_rounds, output_path)


if __name__ == "__main__":
    main()
