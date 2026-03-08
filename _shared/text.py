"""
Text processing and terminal formatting utilities.

This module has NO dependency on any AI SDK or review-domain logic.
"""

from __future__ import annotations

import re


# ── Terminal Utilities ───────────────────────────────────────────────────────

def banner(title: str, char: str = "─", width: int = 72) -> str:
    """Create a styled terminal banner."""
    bar = char * width
    return f"\n{bar}\n  {title}\n{bar}\n"


# ── Text Processing ─────────────────────────────────────────────────────────

def extract_section(text: str, start: str, end: str) -> str | None:
    """Extract text between two markers."""
    s = text.find(start)
    e = text.find(end)
    if s != -1 and e != -1 and s < e:
        return text[s + len(start) : e].strip()
    return None


# ── Table of Contents ────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    """Convert heading text to a GitHub-style anchor slug."""
    slug = text.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)   # strip non-word chars (keep hyphens)
    slug = re.sub(r"\s+", "-", slug)        # spaces -> hyphens
    slug = re.sub(r"-+", "-", slug)         # collapse multiple hyphens
    return slug.strip("-")


def generate_toc(markdown: str, *, max_depth: int = 3) -> str:
    """Generate a markdown Table of Contents from headings in *markdown*.

    Scans for ATX headings (``# …`` through ``######``) and builds a nested
    bullet list with anchor links.  Headings inside fenced code blocks are
    ignored.

    Args:
        markdown:   The full markdown document.
        max_depth:  Deepest heading level to include (default 3 = ``###``).

    Returns:
        A markdown string containing the TOC (without a surrounding heading).
    """
    lines = markdown.split("\n")
    entries: list[tuple[int, str]] = []
    in_fence = False
    slug_counts: dict[str, int] = {}

    for line in lines:
        stripped = line.strip()

        # Track fenced code blocks so we skip headings inside them.
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        m = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if not m:
            continue
        level = len(m.group(1))
        text = m.group(2).strip()

        if level > max_depth:
            continue

        # Skip the TOC heading itself if it already exists
        if text.lower() == "table of contents":
            continue

        slug = _slugify(text)
        # Handle duplicate headings (GitHub-style: append -1, -2, …)
        count = slug_counts.get(slug, 0)
        slug_counts[slug] = count + 1
        if count:
            slug = f"{slug}-{count}"

        indent = "  " * (level - 1)
        entries.append((level, f"{indent}- [{text}](#{slug})"))

    return "\n".join(entry for _, entry in entries)


def format_critiques(critiques: dict[str, str]) -> str:
    """Format reviewer critiques with separators."""
    sep = "─" * 60
    return "\n\n".join(
        f"{sep}\n{name}\n{sep}\n\n{text}" for name, text in critiques.items()
    )


# ── Markdown Processing ─────────────────────────────────────────────────────

def preprocess_markdown(md_text: str) -> str:
    """Clean markdown text for reliable conversion to PDF.

    Normalises bullet markers (* -> -) and ensures blank lines around lists.
    """
    lines = md_text.split("\n")
    out: list[str] = []
    prev_was_bullet = False

    for i, raw_line in enumerate(lines):
        line = raw_line.rstrip()

        # Normalise '* ' bullets to '- '
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if re.match(r"^\*\s+", stripped):
            line = indent + "- " + stripped[2:]
            stripped = line.lstrip()

        # Strip structural section-delimiter markers (e.g. <<<COVER_LETTER_START>>>)
        # that the AI writes as metadata but should not appear in the rendered PDF.
        if re.match(r"^<<<[A-Z_]+>>>$", stripped):
            line = ""
            stripped = ""

        # Strip trailing " <<" or "<<" close-marker from custom >> ... << blockquotes.
        # Applies to ALL lines (not just ">"-prefixed) because multi-line blockquotes
        # have the closing << on a continuation line that no longer starts with ">".
        if stripped.endswith("<<"):
            stripped = stripped[:-2].rstrip()
            line = indent + stripped

        # Strip lone *italic* markers — fpdf2 markdown=True only supports **bold**
        # and --underline--; single asterisks are not parsed and render literally.
        stripped = re.sub(r"(?<!\*)\*(?!\*)([^*]+?)\*(?!\*)", r"\1", stripped)
        line = indent + stripped

        is_bullet = bool(re.match(r"^[-+]\s+", stripped)) or bool(
            re.match(r"^\d+[.)]\s+", stripped)
        )

        # Ensure blank line before a list starts
        if is_bullet and not prev_was_bullet and out and out[-1].strip():
            out.append("")

        out.append(line)
        prev_was_bullet = is_bullet

    return "\n".join(out)
