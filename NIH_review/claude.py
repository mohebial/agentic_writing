#!/usr/bin/env python3
"""
Multi-Agent NIH Grant Peer Review — Claude Backend

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

# ── Allow direct execution: `python NIH_review/claude.py` ────────────────────
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

# ── Anthropic SDK ─────────────────────────────────────────────────────────────

try:
    import anthropic as _anthropic
    _ANTHROPIC_OK = True
except ImportError:
    _anthropic = None  # type: ignore[assignment]
    _ANTHROPIC_OK = False

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_MODEL = "claude-haiku-4-5-20251001"  # or keep the snapshot below

PDF_MIME = "application/pdf"

MODEL_FALLBACK_CHAIN = [
    # Claude 4.6 generation (Feb 2026) — latest
    "claude-opus-4-6",               # Most capable, 1M context, agent teams
    "claude-sonnet-4-6",             # Frontier intelligence at scale, preferred for coding

    # Claude 4.5 generation (Sept–Nov 2025)
    "claude-opus-4-5",               # Premium intelligence (67% cheaper than earlier Opus)
    "claude-sonnet-4-5-20250929",    # Best for agents, coding, computer use
    "claude-haiku-4-5-20251001",     # Fast and cheap

    # Claude 4 generation (May 2025)
    "claude-opus-4-20250514",        # Agentic search, complex coding
    "claude-sonnet-4-20250514",      # Your current default

    # Claude 3.7 (Feb 2025) — legacy but still active
    "claude-sonnet-3-7-20250219",    # Extended thinking / hybrid reasoning

    # Claude 3.5 (Oct 2024) — legacy
    "claude-3-5-sonnet-20241022",    # Your current fallback #2
    "claude-3-5-haiku-20241022",     # Your current fallback #3 — RETIRED Feb 2026 ⚠️
]

MAX_TOKENS        = 16384
_529_RETRY_DELAYS = [15, 30, 60]
_429_RETRY_DELAYS = [60, 120]

# ── Module-level system prompts ───────────────────────────────────────────────

COMBINED_REVIEWER_SYSTEM = build_combined_reviewer_system() if build_combined_reviewer_system else ""
CHALLENGE_SYSTEM         = build_challenge_system()          if build_challenge_system          else ""
SRO_SYSTEM               = load_prompt("sro", scoring=True)  if load_prompt                     else ""
PI_SYSTEM                = load_prompt("pi")                 if load_prompt                     else ""


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
    pdf_base64: str | None = None,
    manuscript_text: str | None = None,
) -> list[dict]:
    """
    Build the Claude content block list for a messages request.

    Round 1 (pdf_base64 set):  PDF document block + prompt text block
    Round 2+ (text set):       combined text block (manuscript + prompt)
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
        f"## Grant Application (Revised)\n\n{manuscript_text}\n\n{prompt}"
        if manuscript_text else prompt
    )
    return [{"type": "text", "text": combined}]


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

def run(pdf_path: str, model: str, max_rounds: int, output_path: str) -> None:
    """Run the multi-agent NIH grant review using Claude."""
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
        f"# NIH Grant Review (Claude): {pdf.name}\n\nModel: `{model}`\n"
    )

    # ── State management ──────────────────────────────────────────────────────
    current_pdf_b64: str | None = pdf_base64
    current_text:    str | None = None
    prev_sro_out:    str        = ""

    for rnd in range(1, max_rounds + 1):
        print(banner(f"REVIEW ROUND {rnd} / {max_rounds}", "═"))
        log.append(f"## Review Round {rnd}\n")

        # ── Combined Reviewers ────────────────────────────────────────────────
        combined_out, model = stream_agent(
            client, model, COMBINED_REVIEWER_SYSTEM,
            build_content(
                "Review the grant application above. Produce all five "
                "reviewer critiques between their sentinel markers, "
                "following each role's instructions exactly.",
                pdf_base64=current_pdf_b64, manuscript_text=current_text,
            ),
            "Review Panel (5 reviewers — combined)",
        )
        critiques = parse_combined_critiques(combined_out)
        for name, text in critiques.items():
            record(name, text)

        # ── Challenge / Independence Audit ────────────────────────────────────
        challenge_out, model = stream_agent(
            client, model, CHALLENGE_SYSTEM,
            build_content(
                "Below are five reviewer critiques of the same NIH grant "
                "application. Evaluate each for independence biases and "
                "produce addenda as instructed.\n\n"
                + format_critiques(critiques),
                pdf_base64=current_pdf_b64, manuscript_text=current_text,
            ),
            "Independence Auditor — Challenge Pass",
        )
        critiques = merge_challenge_addenda(critiques, challenge_out)
        for name, text in critiques.items():
            if "### Independence Auditor" in text:
                record(f"{name} (with addendum)", text)

        critiques_text = format_critiques(critiques)

        # ── Scientific Review Officer (SRO) ───────────────────────────────────
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

        sro_out, model = stream_agent(
            client, model, SRO_SYSTEM,
            build_content(sro_prompt, pdf_base64=current_pdf_b64, manuscript_text=current_text),
            "Scientific Review Officer (SRO) — Summary Statement",
        )
        prev_sro_out = sro_out
        record("Summary Statement (SRO)", sro_out)

        decision = parse_decision(sro_out)
        print(f"\n{'━'*72}")
        print(f"  SRO Decision: {decision}")
        print(f"{'━'*72}\n")
        log.append(f"**SRO Decision: {decision}**\n")

        # ── Terminal conditions ───────────────────────────────────────────────
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

        # ── PI Revision ───────────────────────────────────────────────────────
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
        pi_out, model = stream_agent(
            client, model, PI_SYSTEM,
            build_content(pi_prompt, pdf_base64=current_pdf_b64, manuscript_text=current_text),
            "Principal Investigator — Revision",
        )
        record("PI Response & Revised Application", pi_out)

        # Prepare for next round
        markers = {
            "intro":    (INTRO_START, INTRO_END),
            "aims":     (AIMS_START,  AIMS_END),
            "strategy": (STRAT_START, STRAT_END),
        }
        sections        = extract_revised_grant(pi_out, markers)
        current_text    = build_revised_text(sections, fallback=pi_out)
        current_pdf_b64 = None   # Round 2+ uses text, not the original PDF

        _print_revision(sections, rnd=rnd)
        record(f"Revised Application — Round {rnd}", current_text, level=2)

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
        description="Multi-Agent NIH Grant Peer Review — Claude Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s                                          # opens file picker\n"
            "  %(prog)s proposal.pdf                             # review that PDF\n"
            "  %(prog)s proposal.pdf --max-rounds 3\n"
            "  %(prog)s proposal.pdf --output out.md\n"
            "  %(prog)s proposal.pdf --model claude-opus-4-20250514"
        ),
    )
    parser.add_argument("pdf", nargs="?", help="PDF grant application (omit to open file picker)")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help=f"Claude model (default: {DEFAULT_MODEL})")
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
