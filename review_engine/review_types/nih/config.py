"""NIH Grant Review configuration."""

from pathlib import Path

from review_engine.config import (
    ReviewConfig, ReviewerRole, RevisionSection,
    DecisionConfig, IterationConfig, register,
)

NIH_CONFIG = ReviewConfig(
    name="nih",
    display_name="NIH Grant Review",
    document_noun="grant application",
    base_dir=Path(__file__).parent,
    reviewers=[
        ReviewerRole("Primary Reviewer",   "PRIMARY_REVIEWER",       "primary_reviewer",  True, True),
        ReviewerRole("Secondary Reviewer", "SECONDARY_REVIEWER",     "secondary_reviewer", True, True),
        ReviewerRole("Tertiary Reviewer",  "TERTIARY_REVIEWER",      "tertiary_reviewer",  True, True),
        ReviewerRole("Biostatistics & Rigor Reviewer", "BIOSTATISTICS_REVIEWER", "biostatistics", True, False),
        ReviewerRole("Program Officer",    "PROGRAM_OFFICER",        "program_officer",    False, False),
    ],
    combined_reviewers_prompt="combined_reviewers",
    challenge_prompt="challenge_pass",
    synthesizer_role_name="Scientific Review Officer (SRO)",
    synthesizer_prompt_file="sro",
    synthesizer_include_scoring=True,
    author_role_name="Principal Investigator",
    author_prompt_file="pi",
    revision_sections=[
        RevisionSection("intro",    "INTRO_REVISED_APP", "## Introduction to the Revised Application"),
        RevisionSection("aims",     "SPECIFIC_AIMS",     "## Specific Aims"),
        RevisionSection("strategy", "RESEARCH_STRATEGY", "## Research Strategy"),
    ],
    revised_document_heading="Grant Application (Revised)",
    decision=DecisionConfig(
        regex_pattern=(
            r"Decision\s*[:\-]?\s*(Fundable|Resubmit\s*[—\-]+\s*Minor Revisions"
            r"|Resubmit\s*[—\-]+\s*Major Revisions|NRFC)"
        ),
        decision_label="SRO Decision",
        terminal_positive=["Fundable"],
        terminal_negative=["NRFC"],
        fallback_keywords={"fundable": "Fundable"},
    ),
    iteration=IterationConfig(
        mode="iterative",
        default_max_rounds=2,
    ),
    scoring_file="nih_scoring_scale",
    criteria_file="nih_criteria",
)

register(NIH_CONFIG)
