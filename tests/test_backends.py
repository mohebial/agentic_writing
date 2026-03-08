"""Tests for backend setup validation and content building."""

from __future__ import annotations

import os

import pytest


# ── Availability checks (evaluated at import time for skipif) ───────────────

def _anthropic_available() -> bool:
    from review_engine.backends.claude import _ANTHROPIC_OK
    return _ANTHROPIC_OK


def _gemini_available() -> bool:
    from review_engine.backends.gemini import _GENAI_OK
    return _GENAI_OK


# ── Claude backend ──────────────────────────────────────────────────────────

class TestClaudeBackend:
    def test_validate_startup_raises_on_error(self, monkeypatch):
        """validate_startup raises ClaudeSetupError when deps or key missing."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from review_engine.backends.claude import (
            ClaudeSetupError, validate_startup, _ANTHROPIC_OK,
        )
        if not _ANTHROPIC_OK:
            with pytest.raises(ClaudeSetupError, match="anthropic"):
                validate_startup()
        else:
            with pytest.raises(ClaudeSetupError, match="ANTHROPIC_API_KEY"):
                validate_startup()

    def test_build_content_with_pdf(self):
        from review_engine.backends.claude import build_content
        result = build_content("Do review.", pdf_base64="AAAA")
        assert len(result) == 2
        assert result[0]["type"] == "document"
        assert result[1]["type"] == "text"
        assert result[1]["text"] == "Do review."

    def test_build_content_with_text(self):
        from review_engine.backends.claude import build_content
        result = build_content(
            "Do review.",
            manuscript_text="Revised manuscript.",
            revised_heading="Document (Revised)",
        )
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert "## Document (Revised)" in result[0]["text"]
        assert "Revised manuscript." in result[0]["text"]
        assert "Do review." in result[0]["text"]

    def test_build_content_prompt_only(self):
        from review_engine.backends.claude import build_content
        result = build_content("Just a prompt.")
        assert len(result) == 1
        assert result[0]["text"] == "Just a prompt."

    def test_encode_pdf(self, tmp_path):
        from review_engine.backends.claude import encode_pdf
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")
        encoded = encode_pdf(str(pdf_file))
        assert isinstance(encoded, str)
        import base64
        decoded = base64.standard_b64decode(encoded)
        assert decoded == b"%PDF-1.4 fake content"


# ── Gemini backend ──────────────────────────────────────────────────────────

class TestGeminiBackend:
    def test_validate_startup_raises_on_error(self, monkeypatch):
        """validate_startup raises GeminiSetupError when deps or key missing."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from review_engine.backends.gemini import (
            GeminiSetupError, validate_startup, _GENAI_OK,
        )
        if not _GENAI_OK:
            with pytest.raises(GeminiSetupError, match="google-genai"):
                validate_startup()
        else:
            with pytest.raises(GeminiSetupError, match="GEMINI_API_KEY"):
                validate_startup()

    @pytest.mark.skipif(
        not _gemini_available(), reason="google-genai not installed"
    )
    def test_build_parts_prompt_only(self):
        from review_engine.backends.gemini import build_parts
        result = build_parts("Just a prompt.")
        assert len(result) == 1

    @pytest.mark.skipif(
        not _gemini_available(), reason="google-genai not installed"
    )
    def test_build_parts_with_text(self):
        from review_engine.backends.gemini import build_parts
        result = build_parts(
            "Do review.",
            manuscript_text="Revised text.",
            revised_heading="Doc (Revised)",
        )
        assert len(result) == 1
