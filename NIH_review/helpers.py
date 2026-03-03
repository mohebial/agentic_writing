#!/usr/bin/env python3
"""
Model-agnostic domain logic for the NIH Grant Peer Review system.

Provides:
  - Sentinel markers for structured agent output
  - Prompt loader (reads from NIH_review/prompts/ and instructions/)
  - Combined reviewer & challenge system-prompt builders
  - Parsers: combined critiques, decision, PI revision sections
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
    ("Primary Reviewer",               "<<<PRIMARY_REVIEWER_START>>>",       "<<<PRIMARY_REVIEWER_END>>>"),
    ("Secondary Reviewer",             "<<<SECONDARY_REVIEWER_START>>>",     "<<<SECONDARY_REVIEWER_END>>>"),
    ("Tertiary Reviewer",              "<<<TERTIARY_REVIEWER_START>>>",      "<<<TERTIARY_REVIEWER_END>>>"),
    ("Biostatistics & Rigor Reviewer", "<<<BIOSTATISTICS_REVIEWER_START>>>", "<<<BIOSTATISTICS_REVIEWER_END>>>"),
    ("Program Officer",                "<<<PROGRAM_OFFICER_START>>>",        "<<<PROGRAM_OFFICER_END>>>"),
]

# Independence-auditor addendum sentinels
_CHALLENGE_SENTINELS: list[tuple[str, str, str]] = [
    ("Primary Reviewer",               "<<<PRIMARY_REVIEWER_ADDENDUM_START>>>",       "<<<PRIMARY_REVIEWER_ADDENDUM_END>>>"),
    ("Secondary Reviewer",             "<<<SECONDARY_REVIEWER_ADDENDUM_START>>>",     "<<<SECONDARY_REVIEWER_ADDENDUM_END>>>"),
    ("Tertiary Reviewer",              "<<<TERTIARY_REVIEWER_ADDENDUM_START>>>",      "<<<TERTIARY_REVIEWER_ADDENDUM_END>>>"),
    ("Biostatistics & Rigor Reviewer", "<<<BIOSTATISTICS_REVIEWER_ADDENDUM_START>>>", "<<<BIOSTATISTICS_REVIEWER_ADDENDUM_END>>>"),
    ("Program Officer",                "<<<PROGRAM_OFFICER_ADDENDUM_START>>>",        "<<<PROGRAM_OFFICER_ADDENDUM_END>>>"),
]

# PI / author revision sentinels
AIMS_START  = "<<<SPECIFIC_AIMS_START>>>"
AIMS_END    = "<<<SPECIFIC_AIMS_END>>>"
STRAT_START = "<<<RESEARCH_STRATEGY_START>>>"
STRAT_END   = "<<<RESEARCH_STRATEGY_END>>>"
INTRO_START = "<<<INTRO_REVISED_APP_START>>>"
INTRO_END   = "<<<INTRO_REVISED_APP_END>>>"


# ── Prompt Loader ─────────────────────────────────────────────────────────────

def load_prompt(
    name: str,
    *,
    scoring: bool = False,
    criteria: bool = False,
    rules: bool = True,
) -> str:
    """
    Load a system prompt from NIH_review/prompts/<name>.txt and append
    requested instruction blocks from NIH_review/instructions/.

    Args:
        name:     Filename stem under NIH_review/prompts/
        scoring:  Append instructions/nih_scoring_scale.txt
        criteria: Append instructions/nih_criteria.txt
        rules:    Append instructions/global_rules.txt  (default True)
    """
    base = Path(__file__).parent
    text = (base / "prompts" / f"{name}.txt").read_text(encoding="utf-8").strip()
    extras: list[str] = []
    if scoring:
        extras.append(
            (base / "instructions" / "nih_scoring_scale.txt").read_text(encoding="utf-8").strip()
        )
    if criteria:
        extras.append(
            (base / "instructions" / "nih_criteria.txt").read_text(encoding="utf-8").strip()
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
    five reviewer critiques in one response, separated by sentinels.
    """
    base = Path(__file__).parent
    template = (base / "prompts" / "combined_reviewers.txt").read_text(encoding="utf-8").strip()
    reviewer_prompts = {
        "primary_reviewer":   load_prompt("primary_reviewer",   scoring=True, criteria=True),
        "secondary_reviewer": load_prompt("secondary_reviewer", scoring=True, criteria=True),
        "tertiary_reviewer":  load_prompt("tertiary_reviewer",  scoring=True, criteria=True),
        "biostatistics":      load_prompt("biostatistics",      scoring=True),
        "program_officer":    load_prompt("program_officer"),
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

def parse_decision(sro_text: str) -> str:
    """
    Extract SRO fundability decision from the Summary Statement.

    Returns one of: 'Fundable' | 'Resubmit — Minor Revisions' |
                    'Resubmit — Major Revisions' | 'NRFC' | 'Unknown'
    """
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


# ── PI Revision Extraction ────────────────────────────────────────────────────

def extract_revised_grant(
    pi_text: str,
    markers: dict[str, tuple[str, str]],
) -> dict[str, str | None]:
    """
    Extract revised application sections from the PI's response.

    Args:
        pi_text:  PI's full response text
        markers:  Dict with keys 'intro', 'aims', 'strategy' containing
                  (start_sentinel, end_sentinel) tuples
    """
    return {
        "intro":    extract_section(pi_text, markers["intro"][0],    markers["intro"][1]),
        "aims":     extract_section(pi_text, markers["aims"][0],     markers["aims"][1]),
        "strategy": extract_section(pi_text, markers["strategy"][0], markers["strategy"][1]),
    }


def build_revised_text(sections: dict[str, str | None], fallback: str) -> str:
    """Assemble Introduction, Specific Aims, and Research Strategy into one block."""
    parts = []
    if sections.get("intro"):
        parts.append("## Introduction to the Revised Application\n\n" + sections["intro"])
    if sections.get("aims"):
        parts.append("## Specific Aims\n\n" + sections["aims"])
    if sections.get("strategy"):
        parts.append("## Research Strategy\n\n" + sections["strategy"])
    return "\n\n---\n\n".join(parts) if parts else fallback
