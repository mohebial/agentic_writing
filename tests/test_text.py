"""Tests for _shared.text — text processing and terminal formatting utilities."""

from __future__ import annotations

import pytest
from _shared.text import (
    banner, extract_section, format_critiques, preprocess_markdown,
    generate_toc, _slugify,
)


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


# ── _slugify ───────────────────────────────────────────────────────────────

class TestSlugify:
    def test_basic(self):
        assert _slugify("Hello World") == "hello-world"

    def test_strips_special_chars(self):
        assert _slugify("Decision: Accept (Round 1)") == "decision-accept-round-1"

    def test_collapses_hyphens(self):
        assert _slugify("a --- b") == "a-b"

    def test_strips_leading_trailing_hyphens(self):
        assert _slugify("--hello--") == "hello"

    def test_unicode_letters_preserved(self):
        # \w matches word characters including underscores and unicode letters
        assert _slugify("café") == "café"

    def test_empty_string(self):
        assert _slugify("") == ""


# ── generate_toc ───────────────────────────────────────────────────────────

class TestGenerateToc:
    def test_basic_headings(self):
        md = "# Title\n\n## Section A\n\nContent.\n\n## Section B\n"
        toc = generate_toc(md)
        assert "- [Title](#title)" in toc
        assert "  - [Section A](#section-a)" in toc
        assert "  - [Section B](#section-b)" in toc

    def test_respects_max_depth(self):
        md = "# H1\n## H2\n### H3\n#### H4\n"
        toc = generate_toc(md, max_depth=2)
        assert "H1" in toc
        assert "H2" in toc
        assert "H3" not in toc
        assert "H4" not in toc

    def test_default_max_depth_includes_h3(self):
        md = "# H1\n## H2\n### H3\n#### H4\n"
        toc = generate_toc(md, max_depth=3)
        assert "H3" in toc
        assert "H4" not in toc

    def test_skips_headings_in_code_blocks(self):
        md = "# Real Heading\n\n```\n# Fake Heading\n```\n\n## Another Real\n"
        toc = generate_toc(md)
        assert "Real Heading" in toc
        assert "Fake Heading" not in toc
        assert "Another Real" in toc

    def test_duplicate_headings_get_suffix(self):
        md = "## Round 1\n\nContent.\n\n## Round 1\n"
        toc = generate_toc(md)
        assert "(#round-1)" in toc
        assert "(#round-1-1)" in toc

    def test_skips_toc_heading_itself(self):
        md = "# Title\n## Table of Contents\n## Section\n"
        toc = generate_toc(md)
        assert "Table of Contents" not in toc
        assert "Section" in toc

    def test_empty_input(self):
        assert generate_toc("") == ""

    def test_no_headings(self):
        assert generate_toc("Just plain text.\nNo headings here.") == ""

    def test_indentation_levels(self):
        md = "# L1\n## L2\n### L3\n"
        toc = generate_toc(md)
        lines = toc.split("\n")
        assert lines[0].startswith("- ")       # level 1: no indent
        assert lines[1].startswith("  - ")     # level 2: 2 spaces
        assert lines[2].startswith("    - ")   # level 3: 4 spaces

    def test_heading_with_special_characters(self):
        md = "## SRO Decision (Round 1): Fundable\n"
        toc = generate_toc(md)
        assert "[SRO Decision (Round 1): Fundable]" in toc
        assert "(#sro-decision-round-1-fundable)" in toc
