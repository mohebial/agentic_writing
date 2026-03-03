#!/usr/bin/env python3
"""
Model-agnostic domain logic for the Foundation Grant Review system.

Provides:
  - Sentinel markers for structured agent output
  - Prompt loader (reads from foundation_review/prompts/ and instructions/)
  - Combined reviewer & challenge system-prompt builders
  - Parsers: combined critiques, decision, applicant revision
  - Generic text utilities (re-exported from _shared)

This module has NO dependency on any specific AI SDK.  Both gemini.py and
claude.py import from here; edit the .txt files in prompts/ and instructions/
to change agent behaviour for all backends simultaneously.
"""

from __future__ import annotations

import re
from pathlib import Path

# ── Re-export generic terminal / text / PDF utilities ────────────────────────
# These live in _shared and are shared across all review packages.

from _shared import (          # noqa: F401  (re-exported)
    banner,
    extract_section,
    format_critiques,
    preprocess_markdown,
    convert_to_pdf,
)

# ── Reviewer Output Sentinel Markers ─────────────────────────────────────────

_REVIEWER_SENTINELS: list[tuple[str, str, str]] = [
    ("Scientific Reviewer", "<<<SCIENTIFIC_REVIEWER_START>>>", "<<<SCIENTIFIC_REVIEWER_END>>>"),
    ("Innovation Reviewer", "<<<INNOVATION_REVIEWER_START>>>", "<<<INNOVATION_REVIEWER_END>>>"),
    ("Program Advisor",     "<<<PROGRAM_ADVISOR_START>>>",     "<<<PROGRAM_ADVISOR_END>>>"),
]

# Independence-auditor addendum sentinels
_CHALLENGE_SENTINELS: list[tuple[str, str, str]] = [
    ("Scientific Reviewer", "<<<SCIENTIFIC_REVIEWER_ADDENDUM_START>>>", "<<<SCIENTIFIC_REVIEWER_ADDENDUM_END>>>"),
    ("Innovation Reviewer", "<<<INNOVATION_REVIEWER_ADDENDUM_START>>>", "<<<INNOVATION_REVIEWER_ADDENDUM_END>>>"),
    ("Program Advisor",     "<<<PROGRAM_ADVISOR_ADDENDUM_START>>>",     "<<<PROGRAM_ADVISOR_ADDENDUM_END>>>"),
]

# Applicant / Project Director revision sentinels
COVER_LETTER_START = "<<<COVER_LETTER_START>>>"
COVER_LETTER_END   = "<<<COVER_LETTER_END>>>"
NARRATIVE_START    = "<<<REVISED_NARRATIVE_START>>>"
NARRATIVE_END      = "<<<REVISED_NARRATIVE_END>>>"


# ── Prompt Loader ─────────────────────────────────────────────────────────────

def load_prompt(
    name: str,
    *,
    scoring: bool = False,
    criteria: bool = False,
    rules: bool = True,
) -> str:
    """
    Load a system prompt from foundation_review/prompts/<name>.txt and append
    requested instruction blocks from foundation_review/instructions/.

    Args:
        name:     Filename stem under foundation_review/prompts/
        scoring:  Append instructions/foundation_scoring.txt
        criteria: Append instructions/foundation_criteria.txt
        rules:    Append instructions/global_rules.txt  (default True)
    """
    base = Path(__file__).parent
    text = (base / "prompts" / f"{name}.txt").read_text(encoding="utf-8").strip()
    extras: list[str] = []
    if scoring:
        extras.append(
            (base / "instructions" / "foundation_scoring.txt").read_text(encoding="utf-8").strip()
        )
    if criteria:
        extras.append(
            (base / "instructions" / "foundation_criteria.txt").read_text(encoding="utf-8").strip()
        )
    if rules:
        extras.append(
            (base / "instructions" / "global_rules.txt").read_text(encoding="utf-8").strip()
        )
    return "\n\n".join([text] + extras)


# ── Combined Reviewer Utilities ───────────────────────────────────────────────

def build_combined_reviewer_system() -> str:
    """
    Build a single system prompt that instructs the model to produce all
    three reviewer critiques in one response, separated by sentinels.
    """
    base = Path(__file__).parent
    template = (base / "prompts" / "combined_reviewers.txt").read_text(encoding="utf-8").strip()
    reviewer_prompts = {
        "scientific_reviewer": load_prompt("scientific_reviewer", scoring=True, criteria=True),
        "innovation_reviewer": load_prompt("innovation_reviewer", scoring=True, criteria=True),
        "program_advisor":     load_prompt("program_advisor"),
    }
    return template.format(**reviewer_prompts)


def build_challenge_system() -> str:
    """Load the independence-auditor system prompt."""
    base = Path(__file__).parent
    return (base / "prompts" / "challenge_pass.txt").read_text(encoding="utf-8").strip()


def parse_combined_critiques(raw: str) -> dict[str, str]:
    """Parse the combined reviewer response into individual reviewer critiques."""
    critiques: dict[str, str] = {}
    for name, start_tag, end_tag in _REVIEWER_SENTINELS:
        section = extract_section(raw, start_tag, end_tag)
        critiques[name] = (
            section if section
            else f"[Section missing — sentinel not found for {name}]"
        )
    return critiques


def merge_challenge_addenda(
    critiques: dict[str, str],
    challenge_out: str,
) -> dict[str, str]:
    """Merge independence-auditor addenda into the corresponding reviewer critiques."""
    for name, start_tag, end_tag in _CHALLENGE_SENTINELS:
        addendum = extract_section(challenge_out, start_tag, end_tag)
        if not addendum:
            continue
        if addendum.lower().strip().startswith("no addendum needed"):
            continue
        if name in critiques:
            critiques[name] += "\n\n### Independence Auditor — Addendum\n\n" + addendum
    return critiques


# ── Decision Parser ───────────────────────────────────────────────────────────

def parse_decision(chair_text: str) -> str:
    """
    Extract the Panel Chair's final decision from the recommendation letter.

    Returns one of: 'Fund' | 'Fund with Conditions' | 'Decline' | 'Unknown'
    """
    m = re.search(
        r"Final Decision\s*[:\-]?\s*(Fund with Conditions|Fund|Decline)",
        chair_text,
        re.IGNORECASE,
    )
    if m:
        raw = m.group(1).strip()
        if raw.lower() == "fund":
            return "Fund"
        if "condition" in raw.lower():
            return "Fund with Conditions"
        if raw.lower() == "decline":
            return "Decline"
        return raw.title()
    # Fallback: scan for bare decision keyword on its own line
    for line in chair_text.splitlines():
        ls = line.strip().lower()
        if ls in ("fund", "fund."):
            return "Fund"
        if ls.startswith("fund with condition"):
            return "Fund with Conditions"
        if ls in ("decline", "decline."):
            return "Decline"
    return "Unknown"


# ── Applicant Revision Extraction ─────────────────────────────────────────────

def extract_applicant_revision(text: str) -> dict[str, str | None]:
    """Extract cover letter and revised narrative from the applicant's response."""
    return {
        "cover_letter": extract_section(text, COVER_LETTER_START, COVER_LETTER_END),
        "narrative":    extract_section(text, NARRATIVE_START,    NARRATIVE_END),
    }


def build_revised_text(sections: dict[str, str | None], fallback: str) -> str:
    """Assemble cover letter and revised narrative into a single text block."""
    parts = []
    if sections.get("cover_letter"):
        parts.append("## Cover Letter — Response to Reviewers\n\n" + sections["cover_letter"])
    if sections.get("narrative"):
        parts.append("## Revised Project Narrative\n\n" + sections["narrative"])
    return "\n\n---\n\n".join(parts) if parts else fallback
