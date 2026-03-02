#!/usr/bin/env python3
"""
Helper functions for the Journal Peer Review system.

Re-uses all generic infrastructure from NIH_review_gemini.helpers and adds
journal-specific parsing, prompts, and decision logic on top.
"""

from __future__ import annotations

import re
from pathlib import Path

# ── Re-export generic infrastructure from NIH_review_gemini ──────────────────
# These functions are domain-agnostic: streaming, parts construction, PDF
# export, terminal formatting, etc.

from NIH_review_gemini.helpers import (        # noqa: F401  (re-exported)
    banner,
    stream_agent,
    build_parts,
    make_config,
    file_part,
    text_part,
    extract_section,
    format_critiques,
    preprocess_markdown,
    convert_to_pdf,
    validate_startup,
)

# ── Journal Sentinel Markers ──────────────────────────────────────────────────

# Reviewer output sentinels
_REVIEWER_SENTINELS: list[tuple[str, str, str]] = [
    ("Domain Expert",       "<<<DOMAIN_EXPERT_START>>>",       "<<<DOMAIN_EXPERT_END>>>"),
    ("Technical Reviewer",  "<<<TECHNICAL_REVIEWER_START>>>",  "<<<TECHNICAL_REVIEWER_END>>>"),
    ("Novelty Reviewer",    "<<<NOVELTY_REVIEWER_START>>>",    "<<<NOVELTY_REVIEWER_END>>>"),
]

# Independence-auditor addendum sentinels
_CHALLENGE_SENTINELS: list[tuple[str, str, str]] = [
    ("Domain Expert",       "<<<DOMAIN_EXPERT_ADDENDUM_START>>>",       "<<<DOMAIN_EXPERT_ADDENDUM_END>>>"),
    ("Technical Reviewer",  "<<<TECHNICAL_REVIEWER_ADDENDUM_START>>>",  "<<<TECHNICAL_REVIEWER_ADDENDUM_END>>>"),
    ("Novelty Reviewer",    "<<<NOVELTY_REVIEWER_ADDENDUM_START>>>",    "<<<NOVELTY_REVIEWER_ADDENDUM_END>>>"),
]

# Author revision sentinels
RESPONSE_LETTER_START    = "<<<RESPONSE_LETTER_START>>>"
RESPONSE_LETTER_END      = "<<<RESPONSE_LETTER_END>>>"
REVISED_MANUSCRIPT_START = "<<<REVISED_MANUSCRIPT_START>>>"
REVISED_MANUSCRIPT_END   = "<<<REVISED_MANUSCRIPT_END>>>"


# ── Prompt Loader ─────────────────────────────────────────────────────────────

def load_prompt(
    name: str,
    *,
    scoring: bool = False,
    criteria: bool = False,
    rules: bool = True,
) -> str:
    """
    Load a system prompt from journal_review_gemini/prompts/<name>.txt and
    append requested instruction blocks from journal_review_gemini/instructions/.

    Args:
        name:     Filename stem under journal_review_gemini/prompts/
        scoring:  Append instructions/journal_scoring.txt
        criteria: Append instructions/journal_criteria.txt
        rules:    Append instructions/global_rules.txt (default True)
    """
    base = Path(__file__).parent
    text = (base / "prompts" / f"{name}.txt").read_text(encoding="utf-8").strip()
    extras: list[str] = []
    if scoring:
        extras.append(
            (base / "instructions" / "journal_scoring.txt").read_text(encoding="utf-8").strip()
        )
    if criteria:
        extras.append(
            (base / "instructions" / "journal_criteria.txt").read_text(encoding="utf-8").strip()
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
    three reviewer critiques in one response.
    """
    base = Path(__file__).parent
    template = (base / "prompts" / "combined_reviewers.txt").read_text(encoding="utf-8").strip()
    reviewer_prompts = {
        "domain_expert":      load_prompt("domain_expert",      scoring=True, criteria=True),
        "technical_reviewer": load_prompt("technical_reviewer", scoring=True, criteria=True),
        "novelty_reviewer":   load_prompt("novelty_reviewer",   scoring=True, criteria=True),
    }
    return template.format(**reviewer_prompts)


def build_challenge_system() -> str:
    """Load the independence-auditor system prompt."""
    base = Path(__file__).parent
    return (base / "prompts" / "challenge_pass.txt").read_text(encoding="utf-8").strip()


def parse_combined_critiques(raw: str) -> dict[str, str]:
    """Parse the combined reviewer response into individual critiques."""
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

def parse_decision(editor_text: str) -> str:
    """
    Extract the Editor's decision from the decision letter.

    Matches the line:  Editorial Decision: <value>
    Returns one of:   'Accept' | 'Minor Revision' | 'Major Revision' | 'Reject'
    """
    m = re.search(
        r"Editorial Decision\s*[:\-]?\s*(Accept|Minor Revision|Major Revision|Reject)",
        editor_text,
        re.IGNORECASE,
    )
    if m:
        raw = m.group(1).strip().title()
        # Normalise
        if raw == "Accept":
            return "Accept"
        if "Minor" in raw:
            return "Minor Revision"
        if "Major" in raw:
            return "Major Revision"
        if raw == "Reject":
            return "Reject"
        return raw
    # Fallback: scan for bare decision keyword on its own line
    for line in editor_text.splitlines():
        ls = line.strip().lower()
        if ls in ("accept", "accept."):
            return "Accept"
        if ls.startswith("minor revision"):
            return "Minor Revision"
        if ls.startswith("major revision"):
            return "Major Revision"
        if ls in ("reject", "reject."):
            return "Reject"
    return "Unknown"


# ── Author Revision Extraction ────────────────────────────────────────────────

def extract_author_revision(text: str) -> dict[str, str | None]:
    """
    Extract the response letter and revised manuscript from the author's output.

    For 'Accept' decisions, response_letter will be None (no letter produced).
    """
    return {
        "response_letter": extract_section(text, RESPONSE_LETTER_START, RESPONSE_LETTER_END),
        "manuscript":       extract_section(text, REVISED_MANUSCRIPT_START, REVISED_MANUSCRIPT_END),
    }


def build_revised_text(sections: dict[str, str | None], fallback: str) -> str:
    """Assemble response letter and revised manuscript into a single text block."""
    parts = []
    if sections.get("response_letter"):
        parts.append("## Response to Reviewers\n\n" + sections["response_letter"])
    if sections.get("manuscript"):
        parts.append("## Revised Manuscript\n\n" + sections["manuscript"])
    return "\n\n---\n\n".join(parts) if parts else fallback
