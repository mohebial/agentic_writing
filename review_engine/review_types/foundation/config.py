"""Foundation Grant Review configuration."""

from pathlib import Path

from review_engine.config import (
    ReviewConfig, ReviewerRole, RevisionSection,
    DecisionConfig, IterationConfig, register,
)

FOUNDATION_CONFIG = ReviewConfig(
    name="foundation",
    display_name="Foundation Grant Review",
    document_noun="grant proposal",
    base_dir=Path(__file__).parent,
    reviewers=[
        ReviewerRole("Scientific Reviewer", "SCIENTIFIC_REVIEWER", "scientific_reviewer", True, True),
        ReviewerRole("Innovation Reviewer", "INNOVATION_REVIEWER", "innovation_reviewer", True, True),
        ReviewerRole("Program Advisor",     "PROGRAM_ADVISOR",     "program_advisor",     False, False),
    ],
    combined_reviewers_prompt="combined_reviewers",
    challenge_prompt="challenge_pass",
    synthesizer_role_name="Panel Chair",
    synthesizer_prompt_file="panel_chair",
    synthesizer_include_scoring=False,
    author_role_name="Project Director",
    author_prompt_file="applicant",
    revision_sections=[
        RevisionSection("cover_letter", "COVER_LETTER",      "## Cover Letter — Response to Reviewers"),
        RevisionSection("narrative",    "REVISED_NARRATIVE",  "## Revised Project Narrative"),
    ],
    revised_document_heading="Grant Proposal (Revised)",
    decision=DecisionConfig(
        regex_pattern=r"Final Decision\s*[:\-]?\s*(Fund with Conditions|Fund|Decline)",
        decision_label="Panel Decision",
        terminal_positive=["Fund"],
        terminal_negative=["Decline"],
        fallback_keywords={
            "fund": "Fund",
            "fund with condition": "Fund with Conditions",
            "decline": "Decline",
        },
    ),
    iteration=IterationConfig(
        mode="single_pass",
        default_max_rounds=1,
    ),
    scoring_file="foundation_scoring",
    criteria_file="foundation_criteria",
    gemini_default_model="gemini-2.5-flash",
    gemini_fallback_chain=[
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite-preview-06-17",
        "gemini-2.0-flash",
    ],
)

register(FOUNDATION_CONFIG)
