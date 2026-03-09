"""
ReviewConfig data model and registry.

Each review type (NIH, foundation, journal) is defined as a ReviewConfig
instance and registered at import time.  The orchestrator and backends read
all domain-specific parameters from the config — no hard-coded review logic
outside this module and the per-type config files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ── Building blocks ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReviewerRole:
    """One reviewer in the panel."""
    name: str                    # Display name: "Primary Reviewer"
    sentinel_prefix: str         # e.g. "PRIMARY_REVIEWER"
    prompt_file: str             # Stem under prompts/: "primary_reviewer"
    include_scoring: bool = True
    include_criteria: bool = True


@dataclass(frozen=True)
class RevisionSection:
    """One extractable section from the author/PI revision output."""
    key: str                     # Internal key: "aims", "cover_letter"
    sentinel_prefix: str         # e.g. "SPECIFIC_AIMS"
    heading: str                 # Markdown heading: "## Specific Aims"


@dataclass(frozen=True)
class DecisionConfig:
    """How to parse and act on the synthesizer's decision."""
    regex_pattern: str           # Regex to extract decision
    decision_label: str          # e.g. "SRO Decision", "Panel Decision"
    terminal_positive: list[str] # Stop with success: ["Fundable"]
    terminal_negative: list[str] # Stop with failure: ["NRFC"]
    fallback_keywords: dict[str, str]  # line_lower → decision


@dataclass(frozen=True)
class IterationConfig:
    """How iteration/rounds work for this review type."""
    mode: str                    # "iterative" | "single_pass" | "fixed_rounds"
    default_max_rounds: int      # 2 for NIH/journal, 1 for foundation
    # For journal R2: different valid decisions per round
    r2_valid_decisions: str = ""  # e.g. "Accept, Minor Revision, or Reject"


# ── Main config ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReviewConfig:
    """Complete configuration for one review type."""

    # Identity
    name: str                    # "nih", "foundation", "journal"
    display_name: str            # "NIH Grant Review"
    document_noun: str           # "grant application", "manuscript"

    # Paths
    base_dir: Path               # Absolute path to review_types/<type>/

    # Reviewer panel
    reviewers: list[ReviewerRole]
    combined_reviewers_prompt: str   # "combined_reviewers"
    challenge_prompt: str            # "challenge_pass"

    # Synthesizer (SRO / Panel Chair / Editor)
    synthesizer_role_name: str       # "Scientific Review Officer (SRO)"
    synthesizer_prompt_file: str     # "sro", "panel_chair", "editor"

    # Author/PI
    author_role_name: str            # "Principal Investigator"
    author_prompt_file: str          # "pi", "applicant", "author"

    # ── Fields with defaults below this line ──

    synthesizer_include_scoring: bool = False

    # Revision sections
    revision_sections: list[RevisionSection] = field(default_factory=list)
    revised_document_heading: str = "Document (Revised)"

    # Decision parsing
    decision: DecisionConfig = field(default_factory=lambda: DecisionConfig(
        regex_pattern="", decision_label="Decision",
        terminal_positive=[], terminal_negative=[], fallback_keywords={},
    ))

    # Iteration
    iteration: IterationConfig = field(default_factory=lambda: IterationConfig(
        mode="single_pass", default_max_rounds=1,
    ))

    # Instruction file stems
    scoring_file: str = ""
    criteria_file: str = ""
    rules_file: str = "global_rules"

    # Backend defaults
    claude_default_model: str = "claude-haiku-4-5-20251001"
    gemini_default_model: str = "gemini-2.0-flash"
    gemini_fallback_chain: list[str] = field(default_factory=lambda: [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite-preview-06-17",
    ])
    local_default_model: str = (
        "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF"
    )

    # ── Derived helpers ──

    @property
    def reviewer_sentinels(self) -> list[tuple[str, str, str]]:
        return [
            (r.name, f"<<<{r.sentinel_prefix}_START>>>", f"<<<{r.sentinel_prefix}_END>>>")
            for r in self.reviewers
        ]

    @property
    def challenge_sentinels(self) -> list[tuple[str, str, str]]:
        return [
            (r.name,
             f"<<<{r.sentinel_prefix}_ADDENDUM_START>>>",
             f"<<<{r.sentinel_prefix}_ADDENDUM_END>>>")
            for r in self.reviewers
        ]

    @property
    def revision_markers(self) -> dict[str, tuple[str, str]]:
        return {
            s.key: (f"<<<{s.sentinel_prefix}_START>>>", f"<<<{s.sentinel_prefix}_END>>>")
            for s in self.revision_sections
        }

    def prompts_dir(self) -> Path:
        return self.base_dir / "prompts"

    def instructions_dir(self) -> Path:
        return self.base_dir / "instructions"

    @property
    def reviewer_count_word(self) -> str:
        n = len(self.reviewers)
        return {3: "three", 5: "five"}.get(n, str(n))


# ── Registry ─────────────────────────────────────────────────────────────────

REVIEW_TYPES: dict[str, ReviewConfig] = {}


def register(config: ReviewConfig) -> None:
    """Register a review type configuration."""
    REVIEW_TYPES[config.name] = config


def get_config(name: str) -> ReviewConfig:
    """Look up a review type by name."""
    if name not in REVIEW_TYPES:
        available = ", ".join(REVIEW_TYPES) or "(none loaded)"
        raise ValueError(f"Unknown review type: {name!r}. Available: {available}")
    return REVIEW_TYPES[name]


def ensure_types_loaded() -> None:
    """Import all review type configs so they self-register."""
    if REVIEW_TYPES:
        return
    # Each config.py calls register() at module level
    import review_engine.review_types.nih.config       # noqa: F401
    import review_engine.review_types.foundation.config # noqa: F401
    import review_engine.review_types.journal.config    # noqa: F401
