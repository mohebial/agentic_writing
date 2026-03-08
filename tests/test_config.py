"""Tests for review_engine.config — config loading and registry."""

from __future__ import annotations

import pytest
from review_engine.config import (
    ensure_types_loaded,
    get_config,
    REVIEW_TYPES,
)


@pytest.fixture(autouse=True)
def _load_configs():
    """Ensure all review types are loaded for every test."""
    ensure_types_loaded()


class TestRegistry:
    def test_all_types_registered(self):
        assert "nih" in REVIEW_TYPES
        assert "foundation" in REVIEW_TYPES
        assert "journal" in REVIEW_TYPES

    def test_get_config_nih(self):
        cfg = get_config("nih")
        assert cfg.name == "nih"
        assert cfg.display_name == "NIH Grant Review"
        assert cfg.iteration.mode == "iterative"
        assert len(cfg.reviewers) == 5

    def test_get_config_foundation(self):
        cfg = get_config("foundation")
        assert cfg.name == "foundation"
        assert cfg.iteration.mode == "single_pass"
        assert len(cfg.reviewers) == 3

    def test_get_config_journal(self):
        cfg = get_config("journal")
        assert cfg.name == "journal"
        assert cfg.iteration.mode == "fixed_rounds"
        assert len(cfg.reviewers) == 3

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown review type"):
            get_config("nonexistent")

    def test_configs_are_frozen(self):
        cfg = get_config("nih")
        with pytest.raises(AttributeError):
            cfg.name = "changed"


class TestNIHConfig:
    def test_decision_config(self):
        cfg = get_config("nih")
        assert "Fundable" in cfg.decision.terminal_positive
        assert "NRFC" in cfg.decision.terminal_negative
        assert cfg.decision.decision_label == "SRO Decision"

    def test_revision_sections(self):
        cfg = get_config("nih")
        keys = [rs.key for rs in cfg.revision_sections]
        assert "intro" in keys
        assert "aims" in keys
        assert "strategy" in keys

    def test_reviewer_roles(self):
        cfg = get_config("nih")
        names = [r.name for r in cfg.reviewers]
        assert "Primary Reviewer" in names
        assert "Program Officer" in names

    def test_prompts_dir_exists(self):
        cfg = get_config("nih")
        assert cfg.prompts_dir().is_dir()

    def test_instructions_dir_exists(self):
        cfg = get_config("nih")
        assert cfg.instructions_dir().is_dir()


class TestFoundationConfig:
    def test_decision_config(self):
        cfg = get_config("foundation")
        assert "Fund" in cfg.decision.terminal_positive
        assert "Decline" in cfg.decision.terminal_negative

    def test_gemini_fallback_different(self):
        cfg = get_config("foundation")
        assert cfg.gemini_fallback_chain[0] == "gemini-2.5-flash"


class TestJournalConfig:
    def test_decision_config(self):
        cfg = get_config("journal")
        assert "Accept" in cfg.decision.terminal_positive
        assert "Reject" in cfg.decision.terminal_negative

    def test_r2_valid_decisions(self):
        cfg = get_config("journal")
        assert cfg.iteration.r2_valid_decisions != ""
