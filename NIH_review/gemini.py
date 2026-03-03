#!/usr/bin/env python3
"""
Multi-Agent NIH Grant Peer Review — Gemini Backend

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

# ── Allow direct execution: `python NIH_review/gemini.py` ────────────────────
if __name__ == "__main__" and __package__ in (None, ""):
    _pkg_root = Path(__file__).resolve().parent.parent
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))
    __package__ = "NIH_review"

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
        extract_revised_grant,
        build_revised_text,
        AIMS_START, AIMS_END,
        STRAT_START, STRAT_END,
        INTRO_START, INTRO_END,
    )
    _HELPERS_OK = True
except ImportError:
    _HELPERS_OK = False
    banner = format_critiques = convert_to_pdf = load_prompt = None
    build_combined_reviewer_system = build_challenge_system = None
    parse_combined_critiques = merge_challenge_addenda = None
    parse_decision = extract_revised_grant = build_revised_text = None
    AIMS_START = AIMS_END = STRAT_START = STRAT_END = None
    INTRO_START = INTRO_END = None

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
SRO_SYSTEM               = load_prompt("sro", scoring=True)  if load_prompt                     else ""
PI_SYSTEM                = load_prompt("pi")                 if load_prompt                     else ""


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
    combined = f"## Grant Application (Revised)\n\n{manuscript_text}\n\n{prompt}"
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

            break

    raise RuntimeError(
        f"All models exhausted ({models_to_try}). Last: {last_exc}"
    ) from last_exc


# ── Orchestration ─────────────────────────────────────────────────────────────

def run(pdf_path: str, model: str, max_rounds: int, output_path: str) -> None:
    """Run the multi-agent NIH grant review using Gemini."""
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
        f"# NIH Grant Review (Gemini): {pdf.name}\n\n"
        f"Model: `{model}`  \nFile URI: `{file_uri}`\n"
    )

    # ── State management ──────────────────────────────────────────────────────
    current_file_uri: str | None = file_uri
    current_text:     str | None = None
    prev_sro_out:     str        = ""

    try:
        for rnd in range(1, max_rounds + 1):
            print(banner(f"REVIEW ROUND {rnd} / {max_rounds}", "═"))
            log.append(f"## Review Round {rnd}\n")

            # ── Combined Reviewers (single API call) ──────────────────────────
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

            # ── Challenge / Independence Audit ────────────────────────────────
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
            sections     = extract_revised_grant(pi_out, markers)
            current_text = build_revised_text(sections, fallback=pi_out)
            current_file_uri = None

            _print_revision(sections, rnd=rnd)
            record(f"Revised Application — Round {rnd}", current_text, level=2)

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
    from pathlib import Path as _Path
    _Path(output_path).write_text("\n".join(log), encoding="utf-8")
    print(f"Output saved → {output_path}")
    pdf_out = str(_Path(output_path).with_suffix(".pdf"))
    if convert_to_pdf(output_path, pdf_out):
        print(f"PDF saved    → {pdf_out}\n")
    else:
        print("[PDF] Could not export PDF. Install with: pip install reportlab\n")


def _print_revision(sections: dict[str, str | None], rnd: int) -> None:
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
        print(sections["strategy"][:800])
        if len(sections["strategy"]) > 800:
            print(f"\n[… {len(sections['strategy']) - 800} more chars — see output file …]\n")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _pick_pdf() -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Agent NIH Grant Peer Review — Gemini Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s                                      # opens file picker\n"
            "  %(prog)s proposal.pdf                         # review that PDF\n"
            "  %(prog)s proposal.pdf --max-rounds 3\n"
            "  %(prog)s proposal.pdf --output out.md\n"
            "  %(prog)s proposal.pdf --model gemini-2.5-flash"
        ),
    )
    parser.add_argument("pdf", nargs="?", help="PDF grant application (omit to open file picker)")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help=f"Gemini model (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-rounds", "-r", type=int, default=2, metavar="N",
                        help="Maximum review rounds (default: 2)")
    parser.add_argument("--output", "-o", metavar="FILE",
                        help="Output markdown file (default: <pdf_stem>_review.md)")
    args = parser.parse_args()

    validate_startup()

    pdf_path = str(Path(args.pdf or _pick_pdf()).resolve())
    output_path = (
        str(Path(args.output).resolve()) if args.output
        else str(Path(pdf_path).parent / f"{Path(pdf_path).stem}_review.md")
    )
    run(pdf_path, args.model, args.max_rounds, output_path)


if __name__ == "__main__":
    main()
