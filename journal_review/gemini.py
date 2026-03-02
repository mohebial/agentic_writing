#!/usr/bin/env python3
"""
Multi-Agent Journal Peer Review — Gemini Backend

Accepts a PDF manuscript and runs peer review through up to two rounds using
the Google Gemini API.  See __main__.py for usage.

Reviewers: Domain Expert · Technical Reviewer · Novelty & Impact Reviewer
Editor:    synthesizes reviews into a decision letter
Author/PI: point-by-point response + revised manuscript

Round 1 decisions: Accept / Minor Revision / Major Revision
Round 2 decisions: Accept / Minor Revision / Reject

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

# ── Allow direct execution: `python journal_review/gemini.py` ────────────────
if __name__ == "__main__" and __package__ in (None, ""):
    _pkg_root = Path(__file__).resolve().parent.parent
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))
    __package__ = "journal_review"

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
        extract_author_revision,
        build_revised_text,
        RESPONSE_LETTER_START,
        RESPONSE_LETTER_END,
        REVISED_MANUSCRIPT_START,
        REVISED_MANUSCRIPT_END,
    )
    _HELPERS_OK = True
except ImportError:
    _HELPERS_OK = False
    banner = format_critiques = convert_to_pdf = load_prompt = None
    build_combined_reviewer_system = build_challenge_system = None
    parse_combined_critiques = merge_challenge_addenda = None
    parse_decision = extract_author_revision = build_revised_text = None
    RESPONSE_LETTER_START = RESPONSE_LETTER_END = None
    REVISED_MANUSCRIPT_START = REVISED_MANUSCRIPT_END = None

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
EDITOR_SYSTEM            = load_prompt("editor")             if load_prompt                     else ""
AUTHOR_SYSTEM            = load_prompt("author")             if load_prompt                     else ""


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
    manuscript_text: str | None = None,
) -> list[Any]:
    """
    Build the Part list for a Gemini request.

    Round 1 (file_uri set):  PDF part + prompt text part
    Round 2+ (text set):     combined text part
    """
    if file_uri:
        return [file_part(file_uri), text_part(prompt)]
    combined = f"## Manuscript (Revised)\n\n{manuscript_text}\n\n{prompt}"
    return [text_part(combined)]


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

            break  # success path already returned; break retry loop if needed

    raise RuntimeError(
        f"All models exhausted ({models_to_try}). Last: {last_exc}"
    ) from last_exc


# ── Orchestration ─────────────────────────────────────────────────────────────

def run(pdf_path: str, model: str, output_path: str) -> None:
    """Run the full two-round peer review pipeline using Gemini."""
    api_key = os.environ["GEMINI_API_KEY"]
    client  = genai.Client(api_key=api_key)
    log: list[str] = []

    def record(heading: str, content: str, level: int = 3) -> None:
        log.append(f"{'#' * level} {heading}\n\n{content}\n")

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
        f"# Journal Peer Review (Gemini): {pdf.name}\n\n"
        f"Model: `{model}`  \nFile URI: `{file_uri}`\n"
    )

    current_file_uri: str | None = file_uri
    current_text:     str | None = None
    prev_editor_out:  str        = ""

    try:
        # ══════════════════════════════════════════════════════════════════════
        # ROUND 1
        # ══════════════════════════════════════════════════════════════════════
        print(banner("REVIEW ROUND 1 — Initial Submission", "═"))
        log.append("## Review Round 1 — Initial Submission\n")

        combined_out, model = stream_agent(
            client, model, COMBINED_REVIEWER_SYSTEM,
            build_parts(
                "Review the manuscript above. Produce all three reviewer critiques "
                "between their sentinel markers, following each role's instructions exactly.",
                file_uri=current_file_uri, manuscript_text=current_text,
            ),
            "Review Panel (3 reviewers — combined)",
        )
        critiques = parse_combined_critiques(combined_out)
        for name, text in critiques.items():
            record(name, text)

        challenge_out, model = stream_agent(
            client, model, CHALLENGE_SYSTEM,
            build_parts(
                "Below are three reviewer critiques of the same manuscript. "
                "Evaluate each for independence biases and produce addenda as instructed.\n\n"
                + format_critiques(critiques),
                file_uri=current_file_uri, manuscript_text=current_text,
            ),
            "Independence Auditor — Challenge Pass",
        )
        critiques = merge_challenge_addenda(critiques, challenge_out)
        for name, text in critiques.items():
            if "### Independence Auditor" in text:
                record(f"{name} (with addendum)", text)
        critiques_text = format_critiques(critiques)

        editor_out, model = stream_agent(
            client, model, EDITOR_SYSTEM,
            build_parts(
                f"The following three peer-reviewer critiques were submitted for this manuscript.\n\n"
                f"{critiques_text}\n\n"
                "Synthesize these into an official editorial decision letter following "
                "your output format. This is Round 1; valid decisions are Accept, "
                "Minor Revision, or Major Revision.",
                file_uri=current_file_uri, manuscript_text=current_text,
            ),
            "Editor — Decision Letter (Round 1)",
        )
        prev_editor_out = editor_out
        record("Editorial Decision Letter (Round 1)", editor_out)

        decision = parse_decision(editor_out)
        print(f"\n{'━'*72}\n  Editorial Decision (Round 1): {decision}\n{'━'*72}\n")
        log.append(f"**Editorial Decision (Round 1): {decision}**\n")

        if decision == "Accept":
            author_prompt = (
                "The editor has accepted your manuscript. Congratulations.\n\n"
                "Produce a final polished version. Apply only light copyediting, "
                "clarify any passages flagged by reviewers, and correct minor issues "
                "mentioned in the decision letter. No response letter is required.\n\n"
                f"Reviewer feedback for reference:\n{critiques_text}\n\n"
                f"Editorial Decision Letter:\n{editor_out}\n\n"
                f"Place the final manuscript between {REVISED_MANUSCRIPT_START} "
                f"and {REVISED_MANUSCRIPT_END}."
            )
        else:
            author_prompt = (
                "You have received the following reviewer critiques and editorial "
                "decision letter for your manuscript.\n\n"
                f"Reviewer Critiques:\n{critiques_text}\n\n"
                f"Editorial Decision Letter:\n{editor_out}\n\n"
                "Revise your manuscript accordingly and produce:\n"
                "  1. A point-by-point response letter.\n"
                "  2. A full revised manuscript.\n\n"
                f"  Response letter: {RESPONSE_LETTER_START} ... {RESPONSE_LETTER_END}\n"
                f"  Revised manuscript: {REVISED_MANUSCRIPT_START} ... {REVISED_MANUSCRIPT_END}"
            )

        author_out, model = stream_agent(
            client, model, AUTHOR_SYSTEM,
            build_parts(author_prompt, file_uri=current_file_uri, manuscript_text=current_text),
            f"Author — {'Final Polish' if decision == 'Accept' else 'Revision (Round 1)'}",
        )
        record(
            f"Author — {'Final Polished Manuscript' if decision == 'Accept' else 'Response & Revised Manuscript (Round 1)'}",
            author_out,
        )

        sections = extract_author_revision(author_out)
        current_text     = build_revised_text(sections, fallback=author_out)
        current_file_uri = None
        _print_revision(sections, rnd=1, decision=decision)

        if decision == "Accept":
            print(banner("✓ MANUSCRIPT ACCEPTED — Round 1", "═"))
            log.append(f"\n# Final Manuscript\n\n{current_text}\n")
            return

        # ══════════════════════════════════════════════════════════════════════
        # ROUND 2
        # ══════════════════════════════════════════════════════════════════════
        print(banner(f"REVIEW ROUND 2 — Revised Submission ({decision})", "═"))
        log.append(f"## Review Round 2 — Revised Submission ({decision})\n")

        combined_out_r2, model = stream_agent(
            client, model, COMBINED_REVIEWER_SYSTEM,
            build_parts(
                "Review the REVISED manuscript below. Produce all three reviewer "
                "critiques between their sentinel markers. Note that this is a "
                "revised submission — assess whether prior concerns were addressed.",
                file_uri=None, manuscript_text=current_text,
            ),
            "Review Panel (Round 2 — 3 reviewers combined)",
        )
        critiques_r2 = parse_combined_critiques(combined_out_r2)
        for name, text in critiques_r2.items():
            record(f"{name} (Round 2)", text)

        challenge_out_r2, model = stream_agent(
            client, model, CHALLENGE_SYSTEM,
            build_parts(
                "Below are three Round 2 reviewer critiques of a revised manuscript. "
                "Evaluate each for independence biases and produce addenda as instructed.\n\n"
                + format_critiques(critiques_r2),
                file_uri=None, manuscript_text=current_text,
            ),
            "Independence Auditor — Challenge Pass (Round 2)",
        )
        critiques_r2 = merge_challenge_addenda(critiques_r2, challenge_out_r2)
        for name, text in critiques_r2.items():
            if "### Independence Auditor" in text:
                record(f"{name} (Round 2, with addendum)", text)
        critiques_text_r2 = format_critiques(critiques_r2)

        editor_out_r2, model = stream_agent(
            client, model, EDITOR_SYSTEM,
            build_parts(
                f"Round 1 Decision Letter (for context):\n{prev_editor_out}\n\n"
                f"Round 2 Reviewer Critiques:\n{critiques_text_r2}\n\n"
                "Synthesize into a final editorial decision letter. "
                "This is Round 2 (final); valid decisions are Accept, Minor Revision, or Reject.",
                file_uri=None, manuscript_text=current_text,
            ),
            "Editor — Final Decision Letter (Round 2)",
        )
        record("Editorial Decision Letter (Round 2 — Final)", editor_out_r2)

        decision_r2 = parse_decision(editor_out_r2)
        print(f"\n{'━'*72}\n  Editorial Decision (Round 2 — Final): {decision_r2}\n{'━'*72}\n")
        log.append(f"**Editorial Decision (Round 2 — Final): {decision_r2}**\n")

        if decision_r2 == "Reject":
            author_prompt_r2 = (
                "The editor has rejected your manuscript after Round 2. "
                "Produce a fully revised version addressing all concerns, "
                "for future resubmission elsewhere.\n\n"
                f"Round 2 Reviewer Critiques:\n{critiques_text_r2}\n\n"
                f"Final Editorial Decision Letter:\n{editor_out_r2}\n\n"
                f"Place the final manuscript between {REVISED_MANUSCRIPT_START} "
                f"and {REVISED_MANUSCRIPT_END}. No response letter required."
            )
        else:
            author_prompt_r2 = (
                "You have received the final editorial decision.\n\n"
                f"Round 2 Reviewer Critiques:\n{critiques_text_r2}\n\n"
                f"Final Editorial Decision Letter:\n{editor_out_r2}\n\n"
                "Produce:\n"
                "  1. A point-by-point response letter for remaining concerns.\n"
                "  2. A final revised manuscript.\n\n"
                f"  Response letter: {RESPONSE_LETTER_START} ... {RESPONSE_LETTER_END}\n"
                f"  Final manuscript: {REVISED_MANUSCRIPT_START} ... {REVISED_MANUSCRIPT_END}"
            )

        author_out_r2, model = stream_agent(
            client, model, AUTHOR_SYSTEM,
            build_parts(author_prompt_r2, file_uri=None, manuscript_text=current_text),
            f"Author — Final Version (Round 2 — {decision_r2})",
        )
        record(f"Author — Final Version (Round 2 — {decision_r2})", author_out_r2)

        sections_r2 = extract_author_revision(author_out_r2)
        final_text  = build_revised_text(sections_r2, fallback=author_out_r2)
        _print_revision(sections_r2, rnd=2, decision=decision_r2)
        log.append(f"\n# Final Manuscript (after Round 2)\n\n{final_text}\n")

        if decision_r2 in ("Accept", "Minor Revision"):
            print(banner(
                f"✓ MANUSCRIPT {'ACCEPTED' if decision_r2 == 'Accept' else 'CONDITIONALLY ACCEPTED'} — Round 2",
                "═",
            ))
        else:
            print(banner("✗ MANUSCRIPT REJECTED — Final version saved for resubmission", "═"))

    finally:
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


def _print_revision(sections: dict[str, str | None], rnd: int, decision: str) -> None:
    print(banner(f"Author Output — Round {rnd} ({decision})", "─"))
    if sections.get("response_letter"):
        print("── Response to Reviewers ──\n")
        print(sections["response_letter"][:1000])
        if len(sections["response_letter"]) > 1000:
            print(f"\n[… {len(sections['response_letter']) - 1000} more chars — see output file …]\n")
        print()
    if sections.get("manuscript"):
        print("── Revised Manuscript (excerpt) ──\n")
        print(sections["manuscript"][:800])
        if len(sections["manuscript"]) > 800:
            print(f"\n[… {len(sections['manuscript']) - 800} more chars — see output file …]\n")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _pick_pdf() -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Journal Peer Review — Gemini Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s                                 # opens file picker\n"
            "  %(prog)s manuscript.pdf                  # review that PDF\n"
            "  %(prog)s manuscript.pdf --output r.md\n"
            "  %(prog)s manuscript.pdf --model gemini-2.5-flash"
        ),
    )
    parser.add_argument("pdf", nargs="?", help="PDF manuscript (omit to open file picker)")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help=f"Gemini model (default: {DEFAULT_MODEL})")
    parser.add_argument("--output", "-o", metavar="FILE",
                        help="Output markdown file (default: <pdf_stem>_peer_review.md)")
    args = parser.parse_args()

    validate_startup()

    pdf_path = str(Path(args.pdf or _pick_pdf()).resolve())
    output_path = (
        str(Path(args.output).resolve()) if args.output
        else str(Path(pdf_path).parent / f"{Path(pdf_path).stem}_peer_review.md")
    )
    run(pdf_path, args.model, output_path)


if __name__ == "__main__":
    main()
