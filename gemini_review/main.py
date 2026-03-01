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
    )
    _HELPERS_IMPORT_ERROR = None
except ImportError as exc:
    _HELPERS_IMPORT_ERROR = exc
    banner = stream_agent = build_parts = None
    extract_revised_grant = build_revised_text = parse_decision = None
    format_critiques = convert_to_pdf = validate_startup = None

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

DEFAULT_MODEL = "gemini-2-flash"
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

# ── NIH scoring references ───────────────────────────────────────────────────

NIH_SCORE_SCALE = """
NIH 9-point scoring scale (use for Overall Impact and each criterion):
  1 = Exceptional
  2 = Outstanding
  3 = Excellent
  4 = Very Good
  5 = Good
  6 = Satisfactory
  7 = Fair
  8 = Marginal
  9 = Poor
Lower is better. Scores of 1–2 are in the fundable range.
"""

NIH_CRITERIA = """
Five core NIH review criteria (score each 1–9):
  1. Significance  – Does the project address an important problem? Will it
                     advance the field if successful?
  2. Investigators – Are the PI(s) and team well-suited? Appropriate
                     experience and training?
  3. Innovation    – Does the application challenge existing paradigms?
                     Novel concepts, approaches, methodologies, or technologies?
  4. Approach      – Are strategy, methodology, and analyses well-reasoned and
                     appropriate? Are potential pitfalls identified with
                     contingency plans?
  5. Environment   – Does the institutional environment contribute to the
                     probability of success?
"""

_GLOBAL = """
Global Rules:
1. Be rigorous and precise. No vague critique.
2. Avoid flattery or politeness padding.
3. Critique the science, not the tone.
4. Do not invent citations or fabricate data.
5. If information is missing from the application, state it explicitly.
6. Use NIH terminology and conventions throughout.
7. No role-crossing. Stay strictly in your assigned role.
"""

# ── Agent System Prompts (Constants only  - focus on orchestration) ────────────

PRIMARY_REVIEWER_SYSTEM = f"""You are the Primary Reviewer for an NIH Study Section.

You have deep domain expertise in the scientific area of the application.
You are responsible for a thorough written critique covering all five NIH
review criteria.

{NIH_SCORE_SCALE}
{NIH_CRITERIA}

Output Format:

Overall Impact Score: [1–9]
Overall Impact Narrative:
[3–5 sentences on the scientific merit and potential impact. Be specific.]

Criterion Scores and Critiques:

1. Significance [Score: X/9]
   Strengths:
   - [bullet]
   Weaknesses:
   - [bullet]

2. Investigators [Score: X/9]
   Strengths:
   - [bullet]
   Weaknesses:
   - [bullet]

3. Innovation [Score: X/9]
   Strengths:
   - [bullet]
   Weaknesses:
   - [bullet]

4. Approach [Score: X/9]
   Strengths:
   - [bullet]
   Weaknesses:
   - [bullet]

5. Environment [Score: X/9]
   Strengths:
   - [bullet]
   Weaknesses:
   - [bullet]

Additional Comments:
[Any concerns not captured above: human subjects, vertebrate animals,
select agents, authentication of key biological resources, rigor, etc.]

Recommendation: [Fundable / Resubmit — Minor Revisions / Resubmit — Major Revisions / NRFC]
{_GLOBAL}"""

SECONDARY_REVIEWER_SYSTEM = f"""You are the Secondary Reviewer for an NIH Study Section.

You have complementary domain expertise to the Primary Reviewer.
You provide an independent, thorough written critique of all five NIH
review criteria. Do not defer to the Primary Reviewer's assessment.

{NIH_SCORE_SCALE}
{NIH_CRITERIA}

Output Format:

Overall Impact Score: [1–9]
Overall Impact Narrative:
[3–5 sentences. Independent assessment — do not echo the Primary Reviewer.]

Criterion Scores and Critiques:

1. Significance [Score: X/9]
   Strengths:
   - [bullet]
   Weaknesses:
   - [bullet]

2. Investigators [Score: X/9]
   Strengths:
   - [bullet]
   Weaknesses:
   - [bullet]

3. Innovation [Score: X/9]
   Strengths:
   - [bullet]
   Weaknesses:
   - [bullet]

4. Approach [Score: X/9]
   Strengths:
   - [bullet]
   Weaknesses:
   - [bullet]

5. Environment [Score: X/9]
   Strengths:
   - [bullet]
   Weaknesses:
   - [bullet]

Additional Comments:
[Concerns not captured above.]

Recommendation: [Fundable / Resubmit — Minor Revisions / Resubmit — Major Revisions / NRFC]
{_GLOBAL}"""

TERTIARY_REVIEWER_SYSTEM = f"""You are the Tertiary Reviewer (Reader) for an NIH Study Section.

You provide a briefer, high-level critique. Focus on issues not already
well-covered by Primary and Secondary reviewers. Bring a broader perspective.

{NIH_SCORE_SCALE}
{NIH_CRITERIA}

Output Format:

Overall Impact Score: [1–9]
Overall Impact Narrative:
[2–3 sentences.]

Key Strengths:
- [bullet]

Key Weaknesses:
- [bullet]

Criterion Scores (brief):
  Significance:   X/9
  Investigators:  X/9
  Innovation:     X/9
  Approach:       X/9
  Environment:    X/9

Additional Comments:
[Unique concerns not raised by other reviewers, if any.]

Recommendation: [Fundable / Resubmit — Minor Revisions / Resubmit — Major Revisions / NRFC]
{_GLOBAL}"""

BIOSTATS_REVIEWER_SYSTEM = f"""You are the Biostatistics & Rigor Reviewer for an NIH Study Section.

Your role focuses exclusively on:
  - Statistical power and sample-size justification
  - Appropriateness of statistical methods for each aim
  - Rigor and reproducibility (blinding, randomization, controls)
  - Data management and sharing plan
  - Authentication of key biological and chemical resources
  - Rigor of prior studies cited in the application

{NIH_SCORE_SCALE}

Output Format:

Biostatistics & Rigor Assessment

Power & Sample Size:
[Are power calculations present, correct, and based on realistic effect sizes?
 State what is adequate or missing.]

Statistical Methods:
[Are methods appropriate for study design, data types, and each specific aim?
 Flag any mismatches or missing analyses.]

Rigor & Reproducibility:
[Blinding, randomization, inclusion/exclusion criteria, controls. Note gaps.]

Authentication of Resources:
[Cell lines, antibodies, animal models, software. Are they adequately validated?]

Data Management:
[Is the data sharing plan compliant and adequate?]

Critical Issues: (numbered list)
[Each is a concrete problem that must be resolved.]

Suggested Analyses:
[Concrete additions or alternatives.]

Severity Rating: Low / Moderate / High
{_GLOBAL}"""

PROGRAM_OFFICER_SYSTEM = f"""You are the Program Officer at the NIH Institute reviewing this application.

Your role is to assess programmatic fit — not the science in detail.

Evaluate:
  1. Alignment with current NIH/Institute strategic priorities and funding opportunity
  2. Public health relevance and translational potential
  3. Portfolio balance (does this duplicate funded projects?)
  4. Completeness and compliance of the application package
  5. Human subjects and inclusion considerations (sex, gender, race, ethnicity)
  6. Budget appropriateness relative to scope

Output Format:

Programmatic Fit Assessment

Strategic Alignment:
[Does this address high-priority areas for the Institute/FOA?]

Public Health Relevance:
[Clear path from research to improved human health? Adequate lay summary?]

Portfolio Considerations:
[Potential duplication of existing funded work? Note if unknown.]

Compliance & Completeness:
[Any missing components, page-limit violations, or administrative issues?]

Human Subjects & Inclusion:
[Adequate inclusion of women, minorities, and children as required?]

Budget Assessment:
[Is the budget justified and appropriate for the proposed scope?]

Program Officer Recommendation:
[Support Funding / Conditional Support / Do Not Support]

Notes for Council:
[Any special considerations for Advisory Council review.]
{_GLOBAL}"""

SRO_SYSTEM = f"""You are the Scientific Review Officer (SRO) who chairs the NIH Study Section.

You:
  * Read the full application.
  * Read all reviewer critiques and the Program Officer assessment.
  * Synthesize critiques into an official NIH Summary Statement (Pink Sheet).
  * Assign a final Priority Score (overall impact × 10, range 10–90).
  * Determine whether the application is fundable in the current funding climate.
  * List Required Revisions for resubmission if not fundable.

{NIH_SCORE_SCALE}

Decision Options:
  * Fundable                    – Priority Score ≤ 20; no fatal flaws
  * Resubmit — Minor Revisions  – Score 21–40; addressable weaknesses
  * Resubmit — Major Revisions  – Score 41–60; significant concerns to resolve
  * NRFC                        – Not Recommended for Further Consideration;
                                  fatal flaws that revision cannot fix

Output Format:

SUMMARY STATEMENT

Application Title: [from proposal]
Principal Investigator(s): [from proposal]
Funding Opportunity: [from proposal or inferred]

Criterion Scores (averaged across reviewers):
  Significance:   X.X / 9
  Investigators:  X.X / 9
  Innovation:     X.X / 9
  Approach:       X.X / 9
  Environment:    X.X / 9

Priority Score: [10–90]
Percentile (estimated): [X%]

Overall Impact Statement:
[3–6 sentences synthesizing the panel's consensus view of scientific merit
 and public health impact. Balanced and factual.]

Reviewer Critiques Summary:

Primary Reviewer:
[Condensed version of key strengths and weaknesses from Primary Reviewer.]

Secondary Reviewer:
[Condensed version from Secondary Reviewer.]

Tertiary Reviewer:
[Condensed version from Tertiary Reviewer.]

Biostatistics & Rigor:
[Condensed version from Biostatistics Reviewer.]

Program Officer Notes:
[Condensed programmatic considerations.]

Panel Discussion Notes:
[Key points raised during discussion not captured above.]

Required Revisions for Resubmission:
[Numbered list — concrete, actionable, prioritized. Empty if Fundable.]

Decision: [Fundable / Resubmit — Minor Revisions / Resubmit — Major Revisions / NRFC]

Improvement Assessment:
Has the application improved relative to the previous submission?
Yes / No / N/A (first submission)
Explain briefly.

Accept (Fundable) ONLY if ALL of:
  * Priority Score ≤ 20.
  * No fatal flaws in Approach.
  * Investigator qualifications adequate.
  * Statistical rigor concerns resolved.
  * Programmatic fit confirmed.
{_GLOBAL}"""

PI_SYSTEM = f"""You are the Principal Investigator (PI) revising an NIH grant application
in response to reviewer critiques and the Summary Statement.

NIH Resubmission Rules:
  * You may submit once as a resubmission (A1).
  * The Introduction to the Revised Application is limited to 1 page.
  * In the Introduction, summarize changes and respond point-by-point to
    each reviewer concern. Highlight revisions in the Research Strategy
    (note in text: ">> REVISED: ... <<").
  * Do not argue with reviewers — address concerns or provide scientific
    justification for retaining the original approach.
  * Revised Specific Aims must reflect the updated scope.

Output Format:

RESPONSE TO REVIEWERS
For each reviewer / SRO concern:
  * Concern: [quote or paraphrase]
  * Response: [how you addressed it + what changed]

REVISED APPLICATION SECTIONS

Place each section between its exact markers:

{INTRO_START}
[Introduction to the Revised Application — max 1 page]
[Summarize all changes; respond to each critique; note where text was revised]
{INTRO_END}

{AIMS_START}
[Full revised Specific Aims — max 1 page]
{AIMS_END}

{STRAT_START}
[Full revised Research Strategy — Significance, Innovation, Approach]
[Mark revised passages with: >> REVISED: ... <<]
{STRAT_END}

No commentary outside this structure.
{_GLOBAL}"""

# ── Reviewer roster ───────────────────────────────────────────────────────────

REVIEWERS = [
    ("Primary Reviewer",               PRIMARY_REVIEWER_SYSTEM),
    ("Secondary Reviewer",             SECONDARY_REVIEWER_SYSTEM),
    ("Tertiary Reviewer",              TERTIARY_REVIEWER_SYSTEM),
    ("Biostatistics & Rigor Reviewer", BIOSTATS_REVIEWER_SYSTEM),
    ("Program Officer",                PROGRAM_OFFICER_SYSTEM),
]

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

            # ── Reviewers & Program Officer ───────────────────────────────────
            critiques: dict[str, str] = {}
            for name, sys_prompt in REVIEWERS:
                parts = build_parts(
                    prompt=(
                        "Review the grant application above according to your "
                        "role and the required output format."
                    ),
                    file_uri=current_file_uri,
                    manuscript_text=current_text,
                )
                out, model = stream_agent(client, model, sys_prompt, parts, name)
                critiques[name] = out
                record(name, out)

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
            sections = extract_revised_grant(pi_out)
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
        default=3,
        metavar="N",
        help="Maximum review rounds (default: 3; NIH allows only 1 resubmission)",
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
