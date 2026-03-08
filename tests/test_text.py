"""Tests for _shared.text — text processing and terminal formatting utilities."""

from __future__ import annotations

import pytest
from _shared.text import banner, extract_section, format_critiques, preprocess_markdown


# ── banner ──────────────────────────────────────────────────────────────────

class TestBanner:
    def test_default_char_and_width(self):
        result = banner("Hello")
        assert "Hello" in result
        assert "─" * 72 in result

    def test_custom_char(self):
        result = banner("Title", char="=")
        assert "=" * 72 in result

    def test_custom_width(self):
        result = banner("Title", width=40)
        assert "─" * 40 in result

    def test_structure(self):
        result = banner("Test Title", char="-", width=20)
        lines = result.strip().split("\n")
        assert len(lines) == 3
        assert lines[0] == "-" * 20
        assert "Test Title" in lines[1]
        assert lines[2] == "-" * 20


# ── extract_section ─────────────────────────────────────────────────────────

class TestExtractSection:
    def test_basic_extraction(self):
        text = "before <<<START>>> hello world <<<END>>> after"
        result = extract_section(text, "<<<START>>>", "<<<END>>>")
        assert result == "hello world"

    def test_strips_whitespace(self):
        text = "<<<A>>>   content with spaces   <<<B>>>"
        result = extract_section(text, "<<<A>>>", "<<<B>>>")
        assert result == "content with spaces"

    def test_missing_start(self):
        text = "no start marker <<<END>>> here"
        assert extract_section(text, "<<<START>>>", "<<<END>>>") is None

    def test_missing_end(self):
        text = "<<<START>>> no end marker here"
        assert extract_section(text, "<<<START>>>", "<<<END>>>") is None

    def test_missing_both(self):
        assert extract_section("plain text", "<<<A>>>", "<<<B>>>") is None

    def test_empty_content(self):
        text = "<<<START>>><<<END>>>"
        result = extract_section(text, "<<<START>>>", "<<<END>>>")
        assert result == ""

    def test_multiline_content(self):
        text = "<<<START>>>\nline 1\nline 2\n<<<END>>>"
        result = extract_section(text, "<<<START>>>", "<<<END>>>")
        assert "line 1" in result
        assert "line 2" in result

    def test_uses_first_occurrence(self):
        text = "<<<S>>>first<<<E>>> <<<S>>>second<<<E>>>"
        result = extract_section(text, "<<<S>>>", "<<<E>>>")
        assert result == "first"


# ── format_critiques ────────────────────────────────────────────────────────

class TestFormatCritiques:
    def test_single_critique(self):
        result = format_critiques({"Reviewer 1": "Good work."})
        assert "Reviewer 1" in result
        assert "Good work." in result

    def test_multiple_critiques_have_separators(self):
        critiques = {
            "Reviewer A": "Text A",
            "Reviewer B": "Text B",
        }
        result = format_critiques(critiques)
        assert "Reviewer A" in result
        assert "Reviewer B" in result
        assert "Text A" in result
        assert "Text B" in result
        # Should have separator lines
        assert "─" * 60 in result

    def test_empty_dict(self):
        result = format_critiques({})
        assert result == ""

    def test_preserves_order(self):
        critiques = {"First": "1", "Second": "2", "Third": "3"}
        result = format_critiques(critiques)
        assert result.index("First") < result.index("Second")
        assert result.index("Second") < result.index("Third")


# ── preprocess_markdown ─────────────────────────────────────────────────────

class TestPreprocessMarkdown:
    def test_normalises_star_bullets(self):
        result = preprocess_markdown("* Item one\n* Item two")
        assert "- Item one" in result
        assert "- Item two" in result
        assert "* " not in result

    def test_strips_sentinel_markers(self):
        result = preprocess_markdown("<<<COVER_LETTER_START>>>\nContent\n<<<COVER_LETTER_END>>>")
        lines = [l for l in result.split("\n") if l.strip()]
        assert all("<<<" not in l for l in lines)
        assert "Content" in result

    def test_strips_blockquote_close_marker(self):
        result = preprocess_markdown("some text <<")
        assert "<<" not in result
        assert "some text" in result

    def test_strips_italic_markers(self):
        result = preprocess_markdown("This is *italic* text")
        assert result.strip() == "This is italic text"

    def test_preserves_bold_markers(self):
        result = preprocess_markdown("This is **bold** text")
        assert "**bold**" in result

    def test_adds_blank_line_before_list(self):
        md = "Some paragraph text.\n- First item\n- Second item"
        result = preprocess_markdown(md)
        lines = result.split("\n")
        # Find the bullet line and check preceding line is blank
        for i, line in enumerate(lines):
            if line.strip().startswith("- First"):
                assert lines[i - 1].strip() == ""
                break

    def test_preserves_indented_bullets(self):
        md = "  * Sub-item"
        result = preprocess_markdown(md)
        assert "  - Sub-item" in result

    def test_numbered_lists_get_blank_line(self):
        md = "Paragraph.\n1. First\n2. Second"
        result = preprocess_markdown(md)
        lines = result.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("1."):
                assert lines[i - 1].strip() == ""
                break

    def test_empty_input(self):
        assert preprocess_markdown("") == ""

    def test_no_change_for_clean_markdown(self):
        md = "# Title\n\nA paragraph.\n\n- Bullet"
        result = preprocess_markdown(md)
        assert "# Title" in result
        assert "A paragraph." in result
        assert "- Bullet" in result
