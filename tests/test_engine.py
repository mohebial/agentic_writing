"""Tests for review_engine.engine — ReviewSession class and entry point."""

from __future__ import annotations

from pathlib import Path

import pytest
from review_engine.config import (
    ReviewConfig, ReviewerRole, RevisionSection,
    DecisionConfig, IterationConfig, ensure_types_loaded, get_config,
)
from review_engine.engine import ReviewSession


@pytest.fixture(autouse=True)
def _load_configs():
    ensure_types_loaded()


class TestReviewSessionInit:
    def test_defaults(self):
        cfg = get_config("nih")
        session = ReviewSession(
            config=cfg,
            backend="claude",
            pdf_path="/tmp/test.pdf",
            model="test-model",
            max_rounds=2,
            output_path="/tmp/output.md",
        )
        assert session.config is cfg
        assert session.backend == "claude"
        assert session.model == "test-model"
        assert session.max_rounds == 2
        assert session.log == []

    def test_custom_callbacks(self):
        cfg = get_config("foundation")
        chunks = []
        statuses = []
        session = ReviewSession(
            config=cfg,
            backend="gemini",
            pdf_path="/tmp/test.pdf",
            model="test-model",
            max_rounds=1,
            output_path="/tmp/output.md",
            on_chunk=lambda t: chunks.append(t),
            on_status=lambda s: statuses.append(s),
        )
        session.out("hello")
        session.status("step 1")
        assert chunks == ["hello"]
        assert statuses == ["step 1"]


class TestReviewSessionHelpers:
    @pytest.fixture
    def session(self):
        cfg = get_config("nih")
        return ReviewSession(
            config=cfg,
            backend="claude",
            pdf_path="/tmp/test.pdf",
            model="test-model",
            max_rounds=2,
            output_path="/tmp/output.md",
            on_chunk=lambda t: None,
            on_status=lambda s: None,
        )

    def test_record(self, session):
        session.record("Test Heading", "Content here.")
        assert len(session.log) == 1
        assert "### Test Heading" in session.log[0]
        assert "Content here." in session.log[0]

    def test_record_custom_level(self, session):
        session.record("Title", "Body", level=2)
        assert "## Title" in session.log[0]

    def test_emit_decision(self, session):
        chunks = []
        session.out = lambda t: chunks.append(t)
        session.emit_decision("Fundable")
        assert any("Fundable" in c for c in chunks)
        assert any("Fundable" in entry for entry in session.log)

    def test_emit_decision_with_suffix(self, session):
        chunks = []
        session.out = lambda t: chunks.append(t)
        session.emit_decision("Accept", suffix=" (Round 1)")
        combined = "".join(chunks)
        assert "Editorial Decision (Round 1): Accept" in combined \
            or "SRO Decision (Round 1): Accept" in combined

    def test_marker_instructions(self, session):
        mi = session.marker_instructions
        assert "<<<SPECIFIC_AIMS_START>>>" in mi
        assert "<<<SPECIFIC_AIMS_END>>>" in mi
        assert "<<<RESEARCH_STRATEGY_START>>>" in mi

    def test_revision_sentinel_info(self, session):
        rsi = session.revision_sentinel_info
        assert "Specific Aims" in rsi
        assert "<<<SPECIFIC_AIMS_START>>>" in rsi

    def test_print_revision(self, session):
        chunks = []
        session.out = lambda t: chunks.append(t)
        sections = {
            "aims": "This is the revised aims section.",
            "strategy": None,
        }
        session.print_revision(sections, rnd=1, decision="Resubmit")
        combined = "".join(chunks)
        assert "Round 1" in combined
        assert "Resubmit" in combined
        assert "revised aims" in combined


class TestReviewSessionValidation:
    def test_run_raises_file_not_found(self):
        cfg = get_config("nih")
        session = ReviewSession(
            config=cfg,
            backend="claude",
            pdf_path="/nonexistent/file.pdf",
            model="test-model",
            max_rounds=2,
            output_path="/tmp/output.md",
        )
        with pytest.raises(FileNotFoundError, match="not found"):
            session.run()
