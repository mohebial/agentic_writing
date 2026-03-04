"""Journal Peer Review configuration."""

from pathlib import Path

from review_engine.config import (
    ReviewConfig, ReviewerRole, RevisionSection,
    DecisionConfig, IterationConfig, register,
)

JOURNAL_CONFIG = ReviewConfig(
    name="journal",
    display_name="Journal Peer Review",
    document_noun="manuscript",
    base_dir=Path(__file__).parent,
    reviewers=[
        ReviewerRole("Domain Expert",      "DOMAIN_EXPERT",       "domain_expert",      True, True),
        ReviewerRole("Technical Reviewer", "TECHNICAL_REVIEWER",  "technical_reviewer", True, True),
        ReviewerRole("Novelty Reviewer",   "NOVELTY_REVIEWER",    "novelty_reviewer",   True, True),
    ],
    combined_reviewers_prompt="combined_reviewers",
    challenge_prompt="challenge_pass",
    synthesizer_role_name="Editor",
    synthesizer_prompt_file="editor",
    synthesizer_include_scoring=False,
    author_role_name="Author",
    author_prompt_file="author",
    revision_sections=[
        RevisionSection("response_letter", "RESPONSE_LETTER",      "## Response to Reviewers"),
        RevisionSection("manuscript",      "REVISED_MANUSCRIPT",    "## Revised Manuscript"),
    ],
    revised_document_heading="Manuscript (Revised)",
    decision=DecisionConfig(
        regex_pattern=(
            r"Editorial Decision\s*[:\-]?\s*(Accept|Minor Revision|Major Revision|Reject)"
        ),
        decision_label="Editorial Decision",
        terminal_positive=["Accept"],
        terminal_negative=["Reject"],
        fallback_keywords={
            "accept": "Accept",
            "minor revision": "Minor Revision",
            "major revision": "Major Revision",
            "reject": "Reject",
        },
    ),
    iteration=IterationConfig(
        mode="fixed_rounds",
        default_max_rounds=2,
        r2_valid_decisions="Accept, Minor Revision, or Reject",
    ),
    scoring_file="journal_scoring",
    criteria_file="journal_criteria",
)

register(JOURNAL_CONFIG)
