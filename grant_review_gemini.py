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
    python grant_review_gemini.py proposal.pdf --model gemini-2.5-pro

Requirements:
    pip install google-genai
    export GEMINI_API_KEY=AIza...
"""

from __future__ import annotations

import os
import re
import sys
import argparse
from typing import Any
from pathlib import Path
from datetime import datetime

# ── Optional PDF-export deps (imported lazily) ────────────────────────────────
# Primary:  pip install markdown weasyprint
# Fallback: pip install reportlab markdown

try:
    from google import genai
    from google.genai import types
    _GENAI_IMPORT_ERROR = None
except ImportError as exc:
    genai = None
    types = None
    _GENAI_IMPORT_ERROR = exc

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gemini-3-flash-preview"
PDF_MIME = "application/pdf"

# Fallback model chain — tried in order when a 429 rate-limit is hit.
# The first entry is the preferred model; each subsequent entry is a step down.
MODEL_FALLBACK_CHAIN = [
    "gemini-2.0-flash",       # Gemini 2.0 Flash (primary fallback)
    "gemini-2.5-flash",       # Gemini 2.5 Flash (secondary fallback)
    "gemini-2.5-flash-lite-preview-06-17",  # Gemini 2.5 Flash Lite (last resort)
]

# Sentinels the PI uses to delimit the revised sections
AIMS_START  = "<<<SPECIFIC_AIMS_START>>>"
AIMS_END    = "<<<SPECIFIC_AIMS_END>>>"
STRAT_START = "<<<RESEARCH_STRATEGY_START>>>"
STRAT_END   = "<<<RESEARCH_STRATEGY_END>>>"
INTRO_START = "<<<INTRO_REVISED_APP_START>>>"
INTRO_END   = "<<<INTRO_REVISED_APP_END>>>"

# ── NIH scoring anchor text ───────────────────────────────────────────────────

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

# ── Shared rules ──────────────────────────────────────────────────────────────

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

# ── Agent System Prompts ──────────────────────────────────────────────────────

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

# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(title: str, char: str = "─", width: int = 72) -> str:
    bar = char * width
    return f"\n{bar}\n  {title}\n{bar}\n"


def _make_config(system: str) -> Any:
    return types.GenerateContentConfig(
        system_instruction=system,
        temperature=1.0,      # recommended for creative / analytical tasks
    )


def _file_part(file_uri: str) -> Any:
    return types.Part.from_uri(file_uri=file_uri, mime_type=PDF_MIME)


def _text_part(text: str) -> Any:
    return types.Part.from_text(text=text)


def stream_agent(
    client: Any,
    model: str,
    system: str,
    parts: list,
    label: str,
) -> tuple[str, str]:
    """
    Stream a Gemini response to stdout and return (full_text, model_used).

    On a 429 rate-limit error, automatically retries each model in
    MODEL_FALLBACK_CHAIN (skipping any already tried) before giving up.
    """
    # Build ordered list of models to attempt: requested model first,
    # then any fallbacks not yet in the sequence.
    models_to_try: list[str] = [model] + [
        m for m in MODEL_FALLBACK_CHAIN if m != model
    ]

    last_exc: Exception | None = None
    for attempt_model in models_to_try:
        if attempt_model != model:
            print(
                f"\n[Rate limit] Switching from '{model}' "
                f"→ '{attempt_model}'\n"
            )
        print(_banner(f"{label}  [model: {attempt_model}]"))
        collected: list[str] = []
        try:
            for chunk in client.models.generate_content_stream(
                model=attempt_model,
                contents=[types.Content(parts=parts, role="user")],
                config=_make_config(system),
            ):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    collected.append(chunk.text)
            print("\n")
            return "".join(collected), attempt_model
        except Exception as exc:  # noqa: BLE001
            # 429 RESOURCE_EXHAUSTED → try next model in chain
            is_429 = (
                getattr(exc, "status_code", None) == 429
                or "429" in str(exc)
                or "RESOURCE_EXHAUSTED" in str(exc)
            )
            if is_429:
                print(
                    f"\n[Rate limit] '{attempt_model}' quota exhausted."
                    + (" Trying next model…" if attempt_model != models_to_try[-1] else " No more fallbacks.")
                )
                last_exc = exc
                continue
            raise  # non-429 errors are re-raised immediately

    raise RuntimeError(
        f"All models exhausted ({models_to_try}). Last error: {last_exc}"
    ) from last_exc


def _build_parts(
    prompt: str,
    file_uri: str | None = None,
    manuscript_text: str | None = None,
) -> list[Any]:
    """
    Build the Part list for a single agent call.

    Round 1 (file_uri set):     PDF part + prompt text part
    Round 2+ (text set):        combined text part (manuscript + prompt)
    """
    if file_uri:
        return [_file_part(file_uri), _text_part(prompt)]
    else:
        combined = f"## Grant Application (Revised)\n\n{manuscript_text}\n\n{prompt}"
        return [_text_part(combined)]


def _extract_section(text: str, start: str, end: str) -> str | None:
    s = text.find(start)
    e = text.find(end)
    if s != -1 and e != -1:
        return text[s + len(start) : e].strip()
    return None


def _extract_revised_grant(pi_text: str) -> dict[str, str | None]:
    return {
        "intro":    _extract_section(pi_text, INTRO_START,  INTRO_END),
        "aims":     _extract_section(pi_text, AIMS_START,   AIMS_END),
        "strategy": _extract_section(pi_text, STRAT_START,  STRAT_END),
    }


def _build_revised_text(sections: dict[str, str | None], fallback: str) -> str:
    parts = []
    if sections.get("intro"):
        parts.append("## Introduction to the Revised Application\n\n" + sections["intro"])
    if sections.get("aims"):
        parts.append("## Specific Aims\n\n" + sections["aims"])
    if sections.get("strategy"):
        parts.append("## Research Strategy\n\n" + sections["strategy"])
    return "\n\n---\n\n".join(parts) if parts else fallback


def _parse_decision(sro_text: str) -> str:
    m = re.search(
        r"Decision\s*[:\-]?\s*(Fundable|Resubmit\s*[—\-]+\s*Minor Revisions"
        r"|Resubmit\s*[—\-]+\s*Major Revisions|NRFC)",
        sro_text,
        re.IGNORECASE,
    )
    if m:
        raw = m.group(1).strip()
        raw = re.sub(r"\s*[—\-]+\s*", " — ", raw)
        return raw if "nrfc" in raw.lower() else raw.title()
    if re.search(r"^\s*Fundable\s*$", sro_text, re.MULTILINE | re.IGNORECASE):
        return "Fundable"
    return "Unknown"


def _format_critiques(critiques: dict[str, str]) -> str:
    sep = "─" * 60
    return "\n\n".join(
        f"{sep}\n{name}\n{sep}\n\n{text}" for name, text in critiques.items()
    )


# ── Reviewer roster ───────────────────────────────────────────────────────────

REVIEWERS = [
    ("Primary Reviewer",               PRIMARY_REVIEWER_SYSTEM),
    ("Secondary Reviewer",             SECONDARY_REVIEWER_SYSTEM),
    ("Tertiary Reviewer",              TERTIARY_REVIEWER_SYSTEM),
    ("Biostatistics & Rigor Reviewer", BIOSTATS_REVIEWER_SYSTEM),
    ("Program Officer",                PROGRAM_OFFICER_SYSTEM),
]

# ── Main Review Loop ──────────────────────────────────────────────────────────

def _validate_startup() -> None:
    if _GENAI_IMPORT_ERROR is not None:
        py = sys.executable
        sys.exit(
            "Error: missing dependency 'google-genai' for this Python interpreter.\n"
            f"Interpreter: {py}\n"
            "Install with:\n"
            f"  {py} -m pip install -r requirements.txt"
        )

    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit(
            "Error: GEMINI_API_KEY environment variable is not set.\n"
            "Set it with:\n"
            "  export GEMINI_API_KEY=your_api_key"
        )

def run(pdf_path: str, model: str, max_rounds: int, output_path: str) -> None:
    api_key = os.environ["GEMINI_API_KEY"]

    client = genai.Client(api_key=api_key)
    log: list[str] = []

    def record(heading: str, content: str, level: int = 3) -> None:
        log.append(f"{'#' * level} {heading}\n\n{content}\n")

    # ── Upload PDF ────────────────────────────────────────────────────────────
    pdf = Path(pdf_path)
    if not pdf.exists():
        sys.exit(f"Error: file not found — {pdf_path}")

    print(_banner(f"Uploading {pdf.name}  [model: {model}]", "═"))
    uploaded = client.files.upload(
        file=pdf_path,
        config=types.UploadFileConfig(
            mime_type=PDF_MIME,
            display_name=pdf.name,
        ),
    )
    file_uri  = uploaded.uri
    file_name = uploaded.name   # used for deletion
    print(f"  Uploaded → {file_uri}\n")

    log.append(
        f"# NIH Grant Review (Gemini): {pdf.name}\n\n"
        f"Model: `{model}`  \nFile URI: `{file_uri}`\n"
    )

    # ── State ─────────────────────────────────────────────────────────────────
    current_file_uri: str | None = file_uri   # round 1 → use File API
    current_text:     str | None = None        # round 2+ → use text
    prev_sro_out:     str = ""

    try:
        for rnd in range(1, max_rounds + 1):
            print(_banner(f"REVIEW ROUND {rnd} / {max_rounds}", "═"))
            log.append(f"## Review Round {rnd}\n")

            # ── Reviewers & Program Officer ───────────────────────────────────
            critiques: dict[str, str] = {}
            for name, sys_prompt in REVIEWERS:
                parts = _build_parts(
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

            critiques_text = _format_critiques(critiques)

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

            sro_parts = _build_parts(
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

            decision = _parse_decision(sro_out)
            print(f"\n{'━'*72}")
            print(f"  SRO Decision: {decision}")
            print(f"{'━'*72}\n")
            log.append(f"**SRO Decision: {decision}**\n")

            # ── Terminal conditions ───────────────────────────────────────────
            if decision == "Fundable":
                print(_banner("✓ APPLICATION DEEMED FUNDABLE", "═"))
                if current_text:
                    print(_banner("FINAL REVISED APPLICATION", "─"))
                    print(current_text)
                    print()
                    log.append(f"\n# Final Revised Application\n\n{current_text}\n")
                else:
                    note = f"[Original PDF funded without revision: {pdf.name}]"
                    print(note)
                    log.append(f"\n# Final Revised Application\n\n{note}\n")
                break

            if decision == "NRFC":
                print(_banner("✗ NOT RECOMMENDED FOR FURTHER CONSIDERATION (NRFC)", "═"))
                log.append(
                    f"\n**Application Not Recommended for Further Consideration "
                    f"after round {rnd}.**\n"
                )
                if current_text:
                    print(_banner("Last Revised Application (NRFC)", "─"))
                    print(current_text)
                    print()
                    log.append(f"\n# Last Revised Application\n\n{current_text}\n")
                break

            if rnd == max_rounds:
                print(_banner(f"MAX ROUNDS ({max_rounds}) REACHED — not yet fundable", "═"))
                log.append(
                    f"\n**Stopped: max rounds ({max_rounds}) reached. "
                    f"Final decision: {decision}**\n"
                )
                if current_text:
                    print(_banner("Last Revised Application", "─"))
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
            pi_parts = _build_parts(
                prompt=pi_prompt,
                file_uri=current_file_uri,
                manuscript_text=current_text,
            )
            pi_out, model = stream_agent(
                client, model, PI_SYSTEM, pi_parts,
                "Principal Investigator — Revision",
            )
            record("PI Response & Revised Application", pi_out)

            # Prepare for next round — switch from File API to text
            sections = _extract_revised_grant(pi_out)
            current_text = _build_revised_text(sections, fallback=pi_out)
            current_file_uri = None   # use text from round 2 onward

            # ── Print revised sections prominently ────────────────────────────
            print(_banner(f"Revised Application — Round {rnd}", "─"))
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
        # ── Delete uploaded file from Gemini File API ─────────────────────────
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
        ok = _convert_to_pdf(output_path, pdf_out)
        if ok:
            print(f"PDF saved    → {pdf_out}\n")
        else:
            print(
                "[PDF] Could not export PDF. "
                "Install with: pip install reportlab\n"
            )


# ── PDF Export ───────────────────────────────────────────────────────────────

_PDF_CSS = """
@page {
    size: letter;
    margin: 0.9in 0.85in 0.85in 0.85in;
    @top-left {
        content: "NIH Grant Peer Review";
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 8pt;
        color: #9aa3af;
        padding-top: 6pt;
    }
    @top-right {
        content: string(docdate);
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 8pt;
        color: #9aa3af;
        padding-top: 6pt;
    }
    @bottom-right {
        content: counter(page) " / " counter(pages);
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 8pt;
        color: #9aa3af;
        padding-bottom: 6pt;
    }
    border-top: 2.5pt solid #1a3a5c;
}

body {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 10.2pt;
    line-height: 1.72;
    color: #1e2126;
    margin: 0;
    padding: 0;
}

/* ── Cover / Title block ── */
.doc-title {
    string-set: docdate attr(data-date);
    background: linear-gradient(135deg, #1a3a5c 0%, #2563a8 100%);
    color: #ffffff;
    padding: 28pt 32pt 24pt 32pt;
    margin: -0pt -1pt 28pt -1pt;
    border-radius: 0 0 6pt 6pt;
    page-break-inside: avoid;
}
.doc-title h1 {
    margin: 0 0 6pt 0;
    font-size: 18pt;
    font-weight: 700;
    letter-spacing: -0.3pt;
    color: #ffffff;
    border: none;
    padding: 0;
}
.doc-title .subtitle {
    font-size: 9pt;
    color: #a8c4e0;
    margin: 0;
    font-weight: 400;
}

/* ── Headings ── */
h1 {
    font-size: 16pt;
    font-weight: 700;
    color: #1a3a5c;
    margin: 28pt 0 8pt 0;
    padding-bottom: 5pt;
    border-bottom: 2pt solid #2563a8;
    page-break-after: avoid;
}
h2 {
    font-size: 12.5pt;
    font-weight: 700;
    color: #2563a8;
    margin: 20pt 0 5pt 0;
    padding: 5pt 10pt;
    background: #eef4fb;
    border-left: 4pt solid #2563a8;
    border-radius: 0 3pt 3pt 0;
    page-break-after: avoid;
}
h3 {
    font-size: 10.5pt;
    font-weight: 700;
    color: #1a3a5c;
    margin: 14pt 0 4pt 0;
    page-break-after: avoid;
}
h4 {
    font-size: 10pt;
    font-weight: 700;
    color: #3a5a7c;
    margin: 10pt 0 3pt 0;
    page-break-after: avoid;
}

/* ── Body text ── */
p {
    margin: 0 0 8pt 0;
    text-align: justify;
    orphans: 3;
    widows: 3;
}

/* ── Horizontal rules → section dividers ── */
hr {
    border: none;
    border-top: 1pt solid #c8d8ea;
    margin: 18pt 0;
}

/* ── Lists ── */
ul, ol {
    margin: 4pt 0 8pt 0;
    padding-left: 18pt;
}
li {
    margin-bottom: 3pt;
}

/* ── Score / callout blocks (blockquote) ── */
blockquote {
    margin: 10pt 0 10pt 0;
    padding: 10pt 14pt;
    background: #f0f6ff;
    border-left: 5pt solid #2563a8;
    border-radius: 0 4pt 4pt 0;
    color: #1e2126;
    font-size: 9.5pt;
}
blockquote p { margin: 0; text-align: left; }

/* ── Inline code & code blocks ── */
code {
    font-family: 'Courier New', Courier, monospace;
    font-size: 8.5pt;
    background: #f4f6f8;
    padding: 1pt 3pt;
    border-radius: 2pt;
    color: #c0392b;
}
pre {
    background: #f4f6f8;
    border: 0.5pt solid #dde3ea;
    border-left: 4pt solid #7fb3d3;
    padding: 8pt 10pt;
    border-radius: 3pt;
    font-size: 8pt;
    overflow-x: auto;
    page-break-inside: avoid;
}
pre code {
    background: transparent;
    padding: 0;
    color: #1e2126;
}

/* ── Tables ── */
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 9pt;
    margin: 10pt 0 14pt 0;
    page-break-inside: avoid;
}
th {
    background: #1a3a5c;
    color: #ffffff;
    padding: 6pt 8pt;
    text-align: left;
    font-weight: 600;
}
td {
    padding: 5pt 8pt;
    border-bottom: 0.5pt solid #dde3ea;
    vertical-align: top;
}
tr:nth-child(even) td {
    background: #f5f8fc;
}

/* ── Strong / emphasis ── */
strong { color: #1a3a5c; font-weight: 700; }
em { color: #2563a8; }
"""


def _convert_to_pdf(md_path: str, pdf_path: str) -> bool:
    """Convert *md_path* to a stylish PDF at *pdf_path*.

    Attempts weasyprint first (highest quality), then falls back to
    reportlab.  Returns True on success, False if no PDF library is
    available.
    """
    md_text = Path(md_path).read_text(encoding="utf-8")

    # ── Attempt 1: markdown + weasyprint ─────────────────────────────────────
    try:
        import markdown as _md
        from weasyprint import HTML, CSS  # type: ignore

        html_body = _md.markdown(
            md_text,
            extensions=["tables", "fenced_code", "toc", "nl2br", "sane_lists"],
        )

        # Inject a styled title banner at the top
        doc_date = datetime.now().strftime("%B %d, %Y")
        title_block = (
            f'<div class="doc-title" data-date="{doc_date}">'
            f'<h1>NIH Grant Peer Review — Summary</h1>'
            f'<p class="subtitle">Generated {doc_date} · Multi-Agent Review System</p>'
            f"</div>\n"
        )

        full_html = (
            "<!DOCTYPE html>\n<html lang=\"en\">\n"
            "<head><meta charset=\"utf-8\"></head>\n"
            f"<body>{title_block}{html_body}</body>\n</html>"
        )

        HTML(string=full_html).write_pdf(
            pdf_path,
            stylesheets=[CSS(string=_PDF_CSS)],
        )
        return True

    except ImportError:
        pass  # fall through to reportlab
    except Exception as exc:  # noqa: BLE001
        # Silently skip weasyprint if system GTK/Cairo libs are missing (common on macOS);
        # only warn for unexpected errors.
        err_str = str(exc)
        if any(lib in err_str for lib in ("libgobject", "libcairo", "libpango", "cannot load library")):
            pass  # GTK not installed — fall through quietly
        else:
            print(f"[PDF] weasyprint error: {exc}. Trying reportlab fallback…", file=sys.stderr)

    # ── Attempt 2: markdown + reportlab ──────────────────────────────────────
    try:
        import textwrap
        import markdown as _md
        from reportlab.lib.pagesizes import LETTER  # type: ignore
        from reportlab.lib.units import inch  # type: ignore
        from reportlab.lib import colors  # type: ignore
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # type: ignore
        from reportlab.platypus import (  # type: ignore
            SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
            Table, TableStyle, Preformatted,
        )
        from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER  # type: ignore

        NAVY   = colors.HexColor("#1a3a5c")
        BLUE   = colors.HexColor("#2563a8")
        LTBLUE = colors.HexColor("#eef4fb")
        GREY   = colors.HexColor("#9aa3af")
        DARK   = colors.HexColor("#1e2126")

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=LETTER,
            leftMargin=0.85 * inch,
            rightMargin=0.85 * inch,
            topMargin=0.9 * inch,
            bottomMargin=0.85 * inch,
            title="NIH Grant Peer Review",
        )

        base = getSampleStyleSheet()
        sty = {
            "body":  ParagraphStyle("body",  parent=base["Normal"],
                                    fontSize=10, leading=16, textColor=DARK,
                                    alignment=TA_JUSTIFY, spaceAfter=6),
            "h1":    ParagraphStyle("h1",    parent=base["Heading1"],
                                    fontSize=16, textColor=NAVY, leading=22,
                                    spaceBefore=20, spaceAfter=8,
                                    borderPad=4, borderColor=BLUE,
                                    borderWidth=0, underlineColor=BLUE),
            "h2":    ParagraphStyle("h2",    parent=base["Heading2"],
                                    fontSize=12, textColor=BLUE, leading=16,
                                    spaceBefore=14, spaceAfter=6,
                                    backColor=LTBLUE, leftIndent=-6,
                                    borderPad=(4, 6, 4, 10)),
            "h3":    ParagraphStyle("h3",    parent=base["Heading3"],
                                    fontSize=10.5, textColor=NAVY, leading=15,
                                    spaceBefore=10, spaceAfter=4, fontName="Helvetica-Bold"),
            "code":  ParagraphStyle("code",  parent=base["Code"],
                                    fontSize=8, leading=12, textColor=DARK,
                                    backColor=colors.HexColor("#f4f6f8"),
                                    leftIndent=10, rightIndent=10,
                                    spaceBefore=4, spaceAfter=4),
            "bullet":ParagraphStyle("bullet", parent=base["Normal"],
                                    fontSize=10, leading=15, textColor=DARK,
                                    leftIndent=18, spaceAfter=3,
                                    bulletIndent=6),
        }

        elems = []

        # Title banner (simulated with a table)
        doc_date = datetime.now().strftime("%B %d, %Y")
        banner_data = [[
            Paragraph(
                f'<font color="white" size="16"><b>NIH Grant Peer Review — Summary</b></font><br/>'
                f'<font color="#a8c4e0" size="8">Generated {doc_date} · Multi-Agent Review System</font>',
                ParagraphStyle("banner", parent=base["Normal"], leading=20),
            )
        ]]
        banner_tbl = Table(banner_data, colWidths=[doc.width])
        banner_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), NAVY),
            ("TOPPADDING",    (0, 0), (-1, -1), 16),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ("LEFTPADDING",   (0, 0), (-1, -1), 20),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 20),
            ("ROUNDEDCORNERS", [4]),
        ]))
        elems.append(banner_tbl)
        elems.append(Spacer(1, 0.2 * inch))

        for line in md_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#### "):
                elems.append(Paragraph(stripped[5:], sty["h3"]))
            elif stripped.startswith("### "):
                elems.append(Paragraph(stripped[4:], sty["h3"]))
            elif stripped.startswith("## "):
                elems.append(Paragraph(stripped[3:], sty["h2"]))
            elif stripped.startswith("# "):
                elems.append(Paragraph(stripped[2:], sty["h1"]))
            elif stripped.startswith("---") or stripped.startswith("___"):
                elems.append(HRFlowable(width="100%", thickness=0.5,
                                        color=colors.HexColor("#c8d8ea"),
                                        spaceAfter=8, spaceBefore=8))
            elif stripped.startswith(("- ", "* ", "+ ")):
                elems.append(Paragraph("• " + stripped[2:], sty["bullet"]))
            elif stripped.startswith("`") and stripped.endswith("`"):
                elems.append(Preformatted(stripped[1:-1], sty["code"]))
            elif stripped == "":
                elems.append(Spacer(1, 4))
            else:
                elems.append(Paragraph(stripped, sty["body"]))

        doc.build(elems)
        return True

    except ImportError:
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"[PDF] reportlab error: {exc}", file=sys.stderr)
        return False


# ── Entry Point ───────────────────────────────────────────────────────────────

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Agent NIH Grant Peer Review System — Gemini Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python grant_review_gemini.py               # opens file picker\n"
            "  python grant_review_gemini.py proposal.pdf\n"
            "  python grant_review_gemini.py proposal.pdf --max-rounds 2\n"
            "  python grant_review_gemini.py proposal.pdf --model gemini-2.5-pro\n"
            "  python grant_review_gemini.py proposal.pdf --output summary.md\n"
        ),
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        default=None,
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
        default=None,
        metavar="FILE",
        help="Output markdown file (default: <pdf_folder>/<name>_output.md)",
    )
    args = parser.parse_args()

    _validate_startup()

    if not args.pdf:
        args.pdf = _pick_pdf()

    if not args.output:
        pdf = Path(args.pdf)
        args.output = str(pdf.parent / f"{pdf.stem}_output.md")

    run(args.pdf, args.model, args.max_rounds, args.output)


if __name__ == "__main__":
    main()
