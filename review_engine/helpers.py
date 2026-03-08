"""
Unified, config-driven domain logic for all review types.

Every function is parameterised by a ReviewConfig so one implementation
handles NIH, foundation, and journal reviews.

This module has NO dependency on any AI SDK.
"""

from __future__ import annotations

import re
from pathlib import Path

from _shared import (                          # noqa: F401  (re-exported)
    banner,
    extract_section,
    format_critiques,
    preprocess_markdown,
)
from _shared.pdf import convert_to_pdf         # noqa: F401  (re-exported)

from review_engine.config import ReviewConfig


# ── Prompt Loader ────────────────────────────────────────────────────────────

def load_prompt(
    config: ReviewConfig,
    name: str,
    *,
    scoring: bool = False,
    criteria: bool = False,
    rules: bool = True,
) -> str:
    """
    Load a system prompt from <config>/prompts/<name>.txt and optionally
    append scoring, criteria, and global-rules instruction blocks.
    """
    text = (config.prompts_dir() / f"{name}.txt").read_text(encoding="utf-8").strip()
    extras: list[str] = []
    if scoring and config.scoring_file:
        extras.append(
            (config.instructions_dir() / f"{config.scoring_file}.txt")
            .read_text(encoding="utf-8").strip()
        )
    if criteria and config.criteria_file:
        extras.append(
            (config.instructions_dir() / f"{config.criteria_file}.txt")
            .read_text(encoding="utf-8").strip()
        )
    if rules and config.rules_file:
        extras.append(
            (config.instructions_dir() / f"{config.rules_file}.txt")
            .read_text(encoding="utf-8").strip()
        )
    return "\n\n".join([text] + extras)


# ── Combined Reviewer Utilities ──────────────────────────────────────────────

def build_combined_reviewer_system(config: ReviewConfig) -> str:
    """
    Build a single system prompt that instructs the model to produce all
    reviewer critiques in one response, separated by sentinels.
    """
    template = (
        config.prompts_dir() / f"{config.combined_reviewers_prompt}.txt"
    ).read_text(encoding="utf-8").strip()

    reviewer_prompts = {
        r.prompt_file: load_prompt(
            config, r.prompt_file,
            scoring=r.include_scoring,
            criteria=r.include_criteria,
        )
        for r in config.reviewers
    }
    return template.format(**reviewer_prompts)


def build_challenge_system(config: ReviewConfig) -> str:
    """Load the independence-auditor system prompt."""
    return (
        config.prompts_dir() / f"{config.challenge_prompt}.txt"
    ).read_text(encoding="utf-8").strip()


# ── Parsers ──────────────────────────────────────────────────────────────────

def parse_combined_critiques(config: ReviewConfig, raw: str) -> dict[str, str]:
    """Parse the combined reviewer response into individual critiques."""
    critiques: dict[str, str] = {}
    for name, start_tag, end_tag in config.reviewer_sentinels:
        section = extract_section(raw, start_tag, end_tag)
        critiques[name] = (
            section if section
            else f"[Section missing — sentinel not found for {name}]"
        )
    return critiques


def merge_challenge_addenda(
    config: ReviewConfig,
    critiques: dict[str, str],
    challenge_out: str,
) -> dict[str, str]:
    """Merge independence-auditor addenda into corresponding critiques."""
    for name, start_tag, end_tag in config.challenge_sentinels:
        addendum = extract_section(challenge_out, start_tag, end_tag)
        if not addendum:
            continue
        if addendum.lower().strip().startswith("no addendum needed"):
            continue
        if name in critiques:
            critiques[name] += "\n\n### Independence Auditor — Addendum\n\n" + addendum
    return critiques


def parse_decision(config: ReviewConfig, text: str) -> str:
    """
    Extract the synthesizer's decision using the config's regex.

    Falls back to scanning for bare keyword lines if the regex misses.
    """
    m = re.search(config.decision.regex_pattern, text, re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
        # Normalise dash variants
        raw = re.sub(r"\s*[—\-]+\s*", " — ", raw)
        # NRFC stays uppercase
        if "nrfc" in raw.lower():
            return "NRFC"
        # Use fallback_keywords for canonical casing (longest match first)
        sorted_keywords = sorted(
            config.decision.fallback_keywords.items(),
            key=lambda kv: len(kv[0]),
            reverse=True,
        )
        for keyword, canonical in sorted_keywords:
            if raw.lower() == keyword or raw.lower().startswith(keyword):
                return canonical
        # Default: return as matched
        return raw

    # Fallback: scan for known keyword on its own line
    for line in text.splitlines():
        ls = line.strip().lower().rstrip(".")
        for keyword, decision in config.decision.fallback_keywords.items():
            if ls == keyword or ls.startswith(keyword):
                return decision
    return "Unknown"


# ── Revision Extraction ──────────────────────────────────────────────────────

def extract_revision(config: ReviewConfig, text: str) -> dict[str, str | None]:
    """Extract revised sections using the config's revision markers."""
    markers = config.revision_markers
    return {
        key: extract_section(text, start, end)
        for key, (start, end) in markers.items()
    }


def build_revised_text(
    config: ReviewConfig,
    sections: dict[str, str | None],
    fallback: str,
) -> str:
    """Assemble revised sections into a single text block."""
    parts = []
    for rs in config.revision_sections:
        content = sections.get(rs.key)
        if content:
            parts.append(f"{rs.heading}\n\n{content}")
    return "\n\n---\n\n".join(parts) if parts else fallback
