"""Tests for review_engine.helpers — config-driven review logic functions."""

from __future__ import annotations

from pathlib import Path

import pytest
from review_engine.config import (
    ReviewConfig, ReviewerRole, RevisionSection,
    DecisionConfig, IterationConfig,
)
from review_engine.helpers import (
    parse_combined_critiques,
    merge_challenge_addenda,
    parse_decision,
    extract_revision,
    build_revised_text,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def nih_config():
    """Minimal NIH-like config for testing (no filesystem access needed)."""
    return ReviewConfig(
        name="test_nih",
        display_name="Test NIH Review",
        document_noun="grant application",
        base_dir=Path("/tmp/fake"),
        reviewers=[
            ReviewerRole("Primary Reviewer", "PRIMARY_REVIEWER", "primary_reviewer"),
            ReviewerRole("Secondary Reviewer", "SECONDARY_REVIEWER", "secondary_reviewer"),
        ],
        combined_reviewers_prompt="combined_reviewers",
        challenge_prompt="challenge_pass",
        synthesizer_role_name="SRO",
        synthesizer_prompt_file="sro",
        author_role_name="PI",
        author_prompt_file="pi",
        revision_sections=[
            RevisionSection("aims", "SPECIFIC_AIMS", "## Specific Aims"),
            RevisionSection("strategy", "RESEARCH_STRATEGY", "## Research Strategy"),
        ],
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
        iteration=IterationConfig(mode="iterative", default_max_rounds=2),
    )


@pytest.fixture
def foundation_config():
    """Minimal foundation-like config for testing."""
    return ReviewConfig(
        name="test_foundation",
        display_name="Test Foundation Review",
        document_noun="grant proposal",
        base_dir=Path("/tmp/fake"),
        reviewers=[
            ReviewerRole("Scientific Reviewer", "SCIENTIFIC_REVIEWER", "scientific_reviewer"),
            ReviewerRole("Innovation Reviewer", "INNOVATION_REVIEWER", "innovation_reviewer"),
        ],
        combined_reviewers_prompt="combined_reviewers",
        challenge_prompt="challenge_pass",
        synthesizer_role_name="Panel Chair",
        synthesizer_prompt_file="panel_chair",
        author_role_name="Project Director",
        author_prompt_file="applicant",
        revision_sections=[
            RevisionSection("cover_letter", "COVER_LETTER", "## Cover Letter"),
            RevisionSection("narrative", "REVISED_NARRATIVE", "## Revised Narrative"),
        ],
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
        iteration=IterationConfig(mode="single_pass", default_max_rounds=1),
    )


@pytest.fixture
def journal_config():
    """Minimal journal-like config for testing."""
    return ReviewConfig(
        name="test_journal",
        display_name="Test Journal Review",
        document_noun="manuscript",
        base_dir=Path("/tmp/fake"),
        reviewers=[
            ReviewerRole("Domain Expert", "DOMAIN_EXPERT", "domain_expert"),
            ReviewerRole("Technical Reviewer", "TECHNICAL_REVIEWER", "technical_reviewer"),
        ],
        combined_reviewers_prompt="combined_reviewers",
        challenge_prompt="challenge_pass",
        synthesizer_role_name="Editor",
        synthesizer_prompt_file="editor",
        author_role_name="Author",
        author_prompt_file="author",
        revision_sections=[
            RevisionSection("manuscript", "REVISED_MANUSCRIPT", "## Revised Manuscript"),
        ],
        decision=DecisionConfig(
            regex_pattern=r"Editorial Decision\s*[:\-]?\s*(Accept|Minor Revision|Major Revision|Reject)",
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
    )


# ── parse_combined_critiques ────────────────────────────────────────────────

class TestParseCombinedCritiques:
    def test_extracts_all_reviewers(self, nih_config):
        raw = (
            "preamble\n"
            "<<<PRIMARY_REVIEWER_START>>>\nPrimary feedback here.\n<<<PRIMARY_REVIEWER_END>>>\n"
            "<<<SECONDARY_REVIEWER_START>>>\nSecondary feedback here.\n<<<SECONDARY_REVIEWER_END>>>\n"
        )
        result = parse_combined_critiques(nih_config, raw)
        assert result["Primary Reviewer"] == "Primary feedback here."
        assert result["Secondary Reviewer"] == "Secondary feedback here."

    def test_missing_reviewer_gets_placeholder(self, nih_config):
        raw = "<<<PRIMARY_REVIEWER_START>>>\nSome text.\n<<<PRIMARY_REVIEWER_END>>>"
        result = parse_combined_critiques(nih_config, raw)
        assert "Some text." in result["Primary Reviewer"]
        assert "[Section missing" in result["Secondary Reviewer"]

    def test_all_missing(self, nih_config):
        result = parse_combined_critiques(nih_config, "no markers at all")
        for name in result.values():
            assert "[Section missing" in name

    def test_preserves_multiline_content(self, nih_config):
        raw = (
            "<<<PRIMARY_REVIEWER_START>>>\n"
            "Line 1\nLine 2\nLine 3\n"
            "<<<PRIMARY_REVIEWER_END>>>\n"
            "<<<SECONDARY_REVIEWER_START>>>\nOK\n<<<SECONDARY_REVIEWER_END>>>"
        )
        result = parse_combined_critiques(nih_config, raw)
        assert "Line 1\nLine 2\nLine 3" == result["Primary Reviewer"]


# ── merge_challenge_addenda ─────────────────────────────────────────────────

class TestMergeChallengeAddenda:
    def test_merges_addendum(self, nih_config):
        critiques = {
            "Primary Reviewer": "Original critique.",
            "Secondary Reviewer": "Original secondary.",
        }
        challenge_out = (
            "<<<PRIMARY_REVIEWER_ADDENDUM_START>>>\n"
            "Bias detected: groupthink on significance.\n"
            "<<<PRIMARY_REVIEWER_ADDENDUM_END>>>\n"
            "<<<SECONDARY_REVIEWER_ADDENDUM_START>>>\n"
            "No addendum needed.\n"
            "<<<SECONDARY_REVIEWER_ADDENDUM_END>>>"
        )
        result = merge_challenge_addenda(nih_config, critiques, challenge_out)
        assert "### Independence Auditor" in result["Primary Reviewer"]
        assert "groupthink" in result["Primary Reviewer"]
        # "No addendum needed" should be skipped
        assert "### Independence Auditor" not in result["Secondary Reviewer"]

    def test_no_addenda_found(self, nih_config):
        critiques = {"Primary Reviewer": "Original."}
        result = merge_challenge_addenda(nih_config, critiques, "no markers")
        assert result["Primary Reviewer"] == "Original."

    def test_skips_no_addendum_needed_case_insensitive(self, nih_config):
        critiques = {"Primary Reviewer": "Original."}
        challenge_out = (
            "<<<PRIMARY_REVIEWER_ADDENDUM_START>>>\n"
            "NO ADDENDUM NEEDED\n"
            "<<<PRIMARY_REVIEWER_ADDENDUM_END>>>"
        )
        result = merge_challenge_addenda(nih_config, critiques, challenge_out)
        assert "### Independence Auditor" not in result["Primary Reviewer"]


# ── parse_decision ──────────────────────────────────────────────────────────

class TestParseDecision:
    # NIH decisions
    def test_nih_fundable(self, nih_config):
        text = "After careful review...\n\nDecision: Fundable\n\nEnd."
        assert parse_decision(nih_config, text) == "Fundable"

    def test_nih_nrfc(self, nih_config):
        text = "The application...\nDecision: NRFC"
        assert parse_decision(nih_config, text) == "NRFC"

    def test_nih_resubmit_minor(self, nih_config):
        text = "Decision: Resubmit - Minor Revisions"
        result = parse_decision(nih_config, text)
        assert "Resubmit" in result

    def test_nih_resubmit_major(self, nih_config):
        text = "Decision: Resubmit — Major Revisions"
        result = parse_decision(nih_config, text)
        assert "Resubmit" in result

    def test_nih_fallback_keyword(self, nih_config):
        text = "Overall the application is\nfundable\nbased on merit."
        assert parse_decision(nih_config, text) == "Fundable"

    def test_nih_unknown(self, nih_config):
        text = "No decision mentioned anywhere."
        assert parse_decision(nih_config, text) == "Unknown"

    # Foundation decisions
    def test_foundation_fund(self, foundation_config):
        text = "Final Decision: Fund"
        assert parse_decision(foundation_config, text) == "Fund"

    def test_foundation_fund_with_conditions(self, foundation_config):
        text = "Final Decision: Fund with Conditions"
        assert parse_decision(foundation_config, text) == "Fund with Conditions"

    def test_foundation_decline(self, foundation_config):
        text = "Final Decision: Decline"
        assert parse_decision(foundation_config, text) == "Decline"

    def test_foundation_fallback_fund(self, foundation_config):
        text = "We recommend to\nfund\nthis proposal."
        assert parse_decision(foundation_config, text) == "Fund"

    # Journal decisions
    def test_journal_accept(self, journal_config):
        text = "Editorial Decision: Accept"
        assert parse_decision(journal_config, text) == "Accept"

    def test_journal_minor_revision(self, journal_config):
        text = "Editorial Decision: Minor Revision"
        assert parse_decision(journal_config, text) == "Minor Revision"

    def test_journal_major_revision(self, journal_config):
        text = "Editorial Decision: Major Revision"
        assert parse_decision(journal_config, text) == "Major Revision"

    def test_journal_reject(self, journal_config):
        text = "Editorial Decision: Reject"
        assert parse_decision(journal_config, text) == "Reject"

    def test_journal_fallback(self, journal_config):
        text = "After review:\nreject\nthe manuscript."
        assert parse_decision(journal_config, text) == "Reject"

    # Edge cases
    def test_case_insensitive_regex(self, nih_config):
        text = "decision: fundable"
        assert parse_decision(nih_config, text) == "Fundable"

    def test_colon_variations(self, foundation_config):
        text = "Final Decision - Fund"
        assert parse_decision(foundation_config, text) == "Fund"


# ── extract_revision ────────────────────────────────────────────────────────

class TestExtractRevision:
    def test_extracts_all_sections(self, nih_config):
        text = (
            "<<<SPECIFIC_AIMS_START>>>\nRevised aims.\n<<<SPECIFIC_AIMS_END>>>\n"
            "<<<RESEARCH_STRATEGY_START>>>\nRevised strategy.\n<<<RESEARCH_STRATEGY_END>>>"
        )
        result = extract_revision(nih_config, text)
        assert result["aims"] == "Revised aims."
        assert result["strategy"] == "Revised strategy."

    def test_missing_section_is_none(self, nih_config):
        text = "<<<SPECIFIC_AIMS_START>>>\nAims here.\n<<<SPECIFIC_AIMS_END>>>"
        result = extract_revision(nih_config, text)
        assert result["aims"] == "Aims here."
        assert result["strategy"] is None

    def test_all_missing(self, nih_config):
        result = extract_revision(nih_config, "no markers")
        assert all(v is None for v in result.values())

    def test_foundation_sections(self, foundation_config):
        text = (
            "<<<COVER_LETTER_START>>>\nDear Panel,\n<<<COVER_LETTER_END>>>\n"
            "<<<REVISED_NARRATIVE_START>>>\nNew narrative.\n<<<REVISED_NARRATIVE_END>>>"
        )
        result = extract_revision(foundation_config, text)
        assert result["cover_letter"] == "Dear Panel,"
        assert result["narrative"] == "New narrative."


# ── build_revised_text ──────────────────────────────────────────────────────

class TestBuildRevisedText:
    def test_joins_sections_with_separator(self, nih_config):
        sections = {"aims": "My aims.", "strategy": "My strategy."}
        result = build_revised_text(nih_config, sections, fallback="fallback")
        assert "## Specific Aims" in result
        assert "My aims." in result
        assert "## Research Strategy" in result
        assert "My strategy." in result
        assert "---" in result

    def test_skips_none_sections(self, nih_config):
        sections = {"aims": "My aims.", "strategy": None}
        result = build_revised_text(nih_config, sections, fallback="fallback")
        assert "## Specific Aims" in result
        assert "My aims." in result
        assert "## Research Strategy" not in result

    def test_uses_fallback_when_all_none(self, nih_config):
        sections = {"aims": None, "strategy": None}
        result = build_revised_text(nih_config, sections, fallback="raw output")
        assert result == "raw output"

    def test_single_section(self, journal_config):
        sections = {"manuscript": "The full text."}
        result = build_revised_text(journal_config, sections, fallback="")
        assert "## Revised Manuscript" in result
        assert "The full text." in result
        # Single section should not have separator
        assert "---" not in result


# ── Config derived properties ───────────────────────────────────────────────

class TestConfigProperties:
    def test_reviewer_sentinels(self, nih_config):
        sentinels = nih_config.reviewer_sentinels
        assert len(sentinels) == 2
        assert sentinels[0] == (
            "Primary Reviewer",
            "<<<PRIMARY_REVIEWER_START>>>",
            "<<<PRIMARY_REVIEWER_END>>>",
        )

    def test_challenge_sentinels(self, nih_config):
        sentinels = nih_config.challenge_sentinels
        assert sentinels[0][1] == "<<<PRIMARY_REVIEWER_ADDENDUM_START>>>"

    def test_revision_markers(self, nih_config):
        markers = nih_config.revision_markers
        assert "aims" in markers
        assert markers["aims"] == (
            "<<<SPECIFIC_AIMS_START>>>",
            "<<<SPECIFIC_AIMS_END>>>",
        )

    def test_reviewer_count_word(self, nih_config):
        # 2 reviewers in fixture, not in the {3, 5} map
        assert nih_config.reviewer_count_word == "2"

    def test_reviewer_count_word_three(self, foundation_config):
        # foundation_config fixture has 2 reviewers, but the real one has 3
        # Test the mapping works for known values
        from review_engine.config import ReviewConfig, ReviewerRole
        cfg = ReviewConfig(
            name="test", display_name="", document_noun="",
            base_dir=Path("/tmp"),
            reviewers=[ReviewerRole(f"R{i}", f"R{i}", f"r{i}") for i in range(3)],
            combined_reviewers_prompt="", challenge_prompt="",
            synthesizer_role_name="", synthesizer_prompt_file="",
            author_role_name="", author_prompt_file="",
        )
        assert cfg.reviewer_count_word == "three"

    def test_reviewer_count_word_five(self):
        from review_engine.config import ReviewConfig, ReviewerRole
        cfg = ReviewConfig(
            name="test", display_name="", document_noun="",
            base_dir=Path("/tmp"),
            reviewers=[ReviewerRole(f"R{i}", f"R{i}", f"r{i}") for i in range(5)],
            combined_reviewers_prompt="", challenge_prompt="",
            synthesizer_role_name="", synthesizer_prompt_file="",
            author_role_name="", author_prompt_file="",
        )
        assert cfg.reviewer_count_word == "five"
