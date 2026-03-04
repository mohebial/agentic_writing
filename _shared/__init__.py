#!/usr/bin/env python3
"""
Shared, domain-agnostic utilities used by all review packages.

Provides:
  - Terminal formatting  (banner)
  - Text processing      (extract_section, format_critiques)
  - Markdown processing  (preprocess_markdown)
  - PDF export           (convert_to_pdf)

This module has NO dependency on any AI SDK or review-domain logic.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from datetime import datetime


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
    if s != -1 and e != -1:
        return text[s + len(start) : e].strip()
    return None


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

        is_bullet = bool(re.match(r"^[-+]\s+", stripped)) or bool(
            re.match(r"^\d+[.)]\s+", stripped)
        )

        # Ensure blank line before a list starts
        if is_bullet and not prev_was_bullet and out and out[-1].strip():
            out.append("")

        out.append(line)
        prev_was_bullet = is_bullet

    return "\n".join(out)


# ── PDF Export ───────────────────────────────────────────────────────────────

_PDF_CSS = """
@page {
    size: letter;
    margin: 0.9in 0.85in 0.85in 0.85in;
    @top-left {
        content: "Grant Peer Review";
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 8pt;
        color: #9aa3af;
        padding-top: 6pt;
    }
    @top-right {
        content: string(docdate);
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 8pt;
        color: #9aa3af;
        padding-top: 6pt;
    }
    @bottom-right {
        content: counter(page) " / " counter(pages);
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 8pt;
        color: #9aa3af;
        padding-bottom: 6pt;
    }
    border-top: 2.5pt solid #1a3a5c;
}

body {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 10.2pt;
    line-height: 1.72;
    color: #1e2126;
    margin: 0;
    padding: 0;
}

div.doc-title {
    background-color: #1a3a5c;
    color: white;
    padding: 20px 15px;
    margin-bottom: 30px;
    border-radius: 0;
}

div.doc-title h1 {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 24pt;
    font-weight: bold;
    margin: 0;
    padding: 0;
    line-height: 1;
}

div.doc-title p.subtitle {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 8pt;
    color: #a8c4e0;
    margin: 8px 0 0 0;
    padding: 0;
}

h1 { font-size: 16pt; color: #1a3a5c; font-weight: bold;
     margin: 20px 0 10px 0; padding: 0; border-bottom: 2pt solid #2563a8;
     padding-bottom: 4px; }

h2 { font-size: 13pt; color: #2563a8; font-weight: bold;
     margin: 16px 0 8px 0; padding: 6px 8px;
     background-color: #eef4fb; border-left: 3pt solid #2563a8; }

h3 { font-size: 11pt; color: #1a3a5c; font-weight: bold;
     margin: 12px 0 6px 0; padding: 0; }

h4 { font-size: 10pt; color: #3a5a7c; font-weight: bold;
     margin: 10px 0 5px 0; padding: 0; }

p { margin: 0 0 10px 0; line-height: 1.7; }

ul, ol { margin: 8px 0 8px 20px; padding: 0; }

li { margin: 4px 0; line-height: 1.5; }

code { font-family: Courier, monospace; font-size: 9pt;
       color: #c0392b; background-color: #f4f6f8;
       padding: 2px 4px; border-radius: 2px; }

pre { background-color: #f4f6f8; padding: 10px 12px;
      margin: 8px 0; border-left: 2pt solid #7fb3d5;
      font-family: Courier, monospace; font-size: 8pt;
      overflow-x: auto; line-height: 1.4; }

blockquote { margin: 0 0 10px 0; padding: 10px 15px;
             border-left: 3pt solid #2563a8;
             background-color: #f0f6ff;
             font-style: italic; color: #3a3e43; }

table { border-collapse: collapse; margin: 10px 0; width: 100%; }
th { background-color: #eef4fb; padding: 8px; text-align: left;
     font-weight: bold; border: 1px solid #b8d4ed; }
td { padding: 8px; border: 1px solid #b8d4ed; }

hr { border: none; height: 1px; background-color: #c8d8ea;
     margin: 16px 0; }
"""


def convert_to_pdf(md_path: str, pdf_path: str) -> bool:
    """Convert markdown file to PDF.

    Strategy (in order of preference):
      1. markdown + weasyprint  (highest quality)
      2. fpdf2                  (pure-Python, robust)
      3. reportlab              (legacy fallback)

    Returns True on success, False if no PDF library available.
    """
    md_text = Path(md_path).read_text(encoding="utf-8")
    md_clean = preprocess_markdown(md_text)

    # ── Attempt 1: weasyprint ────────────────────────────────────────────────
    try:
        import os as _os
        import logging as _logging
        import markdown as _md

        def _silence_fds():
            """Redirect both fd 1 (stdout) and fd 2 (stderr) to /dev/null."""
            dn = _os.open(_os.devnull, _os.O_WRONLY)
            s1, s2 = _os.dup(1), _os.dup(2)
            _os.dup2(dn, 1)
            _os.dup2(dn, 2)
            _os.close(dn)
            return s1, s2

        def _restore_fds(s1: int, s2: int) -> None:
            _os.dup2(s1, 1); _os.close(s1)
            _os.dup2(s2, 2); _os.close(s2)

        # WeasyPrint emits "could not import external libraries" via its logger
        # AND via native fd-2 writes.  Silence both for the whole WeasyPrint scope.
        _wp_logger = _logging.getLogger("weasyprint")
        _prev_level = _wp_logger.level
        _wp_logger.setLevel(_logging.CRITICAL)
        _s1, _s2 = _silence_fds()
        try:
            from weasyprint import HTML, CSS  # type: ignore

            html_body = _md.markdown(
                md_clean,
                extensions=["tables", "fenced_code", "toc", "sane_lists"],
            )

            doc_date = datetime.now().strftime("%B %d, %Y")
            title_block = (
                f'<div class="doc-title" data-date="{doc_date}">'
                f"<h1>Peer Review -- Summary</h1>"
                f'<p class="subtitle">Generated {doc_date} · Multi-Agent Review System</p>'
                f"</div>\n"
            )

            full_html = (
                '<!DOCTYPE html>\n<html lang="en">\n'
                '<head><meta charset="utf-8"></head>\n'
                f"<body>{title_block}{html_body}</body>\n</html>"
            )

            HTML(string=full_html).write_pdf(
                pdf_path,
                stylesheets=[CSS(string=_PDF_CSS)],
            )
            return True
        finally:
            _restore_fds(_s1, _s2)
            _wp_logger.setLevel(_prev_level)

    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        err_str = str(exc)
        if any(lib in err_str for lib in (
            "libgobject", "libcairo", "libpango", "cannot load library",
        )):
            pass
        else:
            print(f"[PDF] weasyprint error: {exc}. Trying fpdf2 fallback…",
                  file=sys.stderr)

    # ── Attempt 2: fpdf2 ─────────────────────────────────────────────────────
    try:
        from fpdf import FPDF  # type: ignore

        NAVY = (26, 58, 92)
        BLUE = (37, 99, 168)
        LTBLUE = (238, 244, 251)
        DARK = (30, 33, 38)
        GRAY = (154, 163, 175)
        WHITE = (255, 255, 255)
        CODEBG = (244, 246, 248)
        BQBLUE = (240, 246, 255)

        _UNICODE_MAP = {
            "\u2014": "--",
            "\u2013": "-",
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2026": "...",
            "\u00b7": ".",
            "\u2022": "-",
            "\u00a0": " ",
            "\u2032": "'",
            "\u2033": '"',
            "\u2264": "<=",
            "\u2265": ">=",
            "\u00d7": "x",
            "\u2192": "->",
            "\u2190": "<-",
            "\u00b1": "+/-",
        }

        def _safe(text: str) -> str:
            """Replace Unicode chars unsupported by core PDF fonts."""
            for uc, repl in _UNICODE_MAP.items():
                text = text.replace(uc, repl)
            return text.encode("latin-1", errors="replace").decode("latin-1")

        class StylishPDF(FPDF):
            """Custom FPDF with header/footer and styled rendering."""

            def __init__(self) -> None:
                super().__init__(format="letter")
                self.set_auto_page_break(auto=True, margin=22)
                self.set_margins(left=21.7, top=23, right=21.7)
                self.alias_nb_pages()
                self._doc_date = datetime.now().strftime("%B %d, %Y")

            def header(self) -> None:
                if self.page_no() == 1:
                    return
                self.set_font("Helvetica", "", 7)
                self.set_text_color(*GRAY)
                self.cell(0, 4, "Peer Review", align="L")
                self.cell(0, 4, self._doc_date, align="R", new_x="LMARGIN", new_y="NEXT")
                self.set_draw_color(*NAVY)
                self.set_line_width(0.6)
                y = self.get_y() + 1
                self.line(self.l_margin, y, self.w - self.r_margin, y)
                self.set_y(y + 4)

            def footer(self) -> None:
                self.set_y(-15)
                self.set_font("Helvetica", "", 7)
                self.set_text_color(*GRAY)
                self.cell(
                    0, 10,
                    f"Page {self.page_no()}/{{nb}}",
                    align="C",
                )

            def title_banner(self) -> None:
                self.set_fill_color(*NAVY)
                bw = self.w - self.l_margin - self.r_margin
                bx, by = self.l_margin, self.get_y()
                bh = 26
                self.rect(bx, by, bw, bh, style="F")
                self.set_xy(bx + 6, by + 5)
                self.set_font("Helvetica", "B", 16)
                self.set_text_color(*WHITE)
                self.cell(0, 8, _safe("Peer Review -- Summary"))
                self.set_xy(bx + 6, by + 15)
                self.set_font("Helvetica", "", 8)
                self.set_text_color(168, 196, 224)
                self.cell(
                    0, 5,
                    _safe(f"Generated {self._doc_date}  |  Multi-Agent Review System"),
                )
                self.set_y(by + bh + 8)

            # ── inline markdown renderer ──────────────────────────────────
            def _render_rich(
                self,
                text: str,
                size: float = 9.5,
                line_h: float = 5.5,
            ) -> None:
                """Write text honouring **bold** and *italic* inline markers.

                Uses write() so line-wrapping respects the current l_margin.
                Caller is responsible for positioning (set_x / set_left_margin)
                before calling, and for any trailing ln() afterwards.
                """
                import re as _ire
                parts = _ire.split(r'(\*\*[^*]+?\*\*|\*[^*]+?\*)', text)
                self.set_text_color(*DARK)
                for part in parts:
                    if not part:
                        continue
                    if part.startswith('**') and part.endswith('**') and len(part) > 4:
                        self.set_font('Helvetica', 'B', size)
                        self.write(line_h, _safe(part[2:-2]))
                    elif part.startswith('*') and part.endswith('*') and len(part) > 2:
                        self.set_font('Helvetica', 'I', size)
                        self.write(line_h, _safe(part[1:-1]))
                    else:
                        self.set_font('Helvetica', '', size)
                        self.write(line_h, _safe(part))
                # Reset to plain after inline run
                self.set_font('Helvetica', '', size)

            def _strip_md(self, text: str) -> str:
                """Remove **bold** / *italic* markers, return plain safe string."""
                import re as _ire
                t = _ire.sub(r'\*\*([^*]+?)\*\*', r'\1', text)
                t = _ire.sub(r'\*([^*]+?)\*', r'\1', t)
                return _safe(t)

            def render_h1(self, text: str) -> None:
                self.ln(6)
                self.set_font("Helvetica", "B", 14)
                self.set_text_color(*NAVY)
                self.multi_cell(0, 7, self._strip_md(text))
                y = self.get_y() + 1
                self.set_draw_color(*BLUE)
                self.set_line_width(0.55)
                self.line(self.l_margin, y, self.w - self.r_margin, y)
                self.set_y(y + 4)

            def render_h2(self, text: str) -> None:
                self.ln(5)
                bw = self.w - self.l_margin - self.r_margin
                bx = self.l_margin
                by = self.get_y()
                self.set_fill_color(*LTBLUE)
                self.rect(bx, by, bw, 8, style="F")
                self.set_fill_color(*BLUE)
                self.rect(bx, by, 1.2, 8, style="F")
                self.set_xy(bx + 4, by + 1)
                self.set_font("Helvetica", "B", 11)
                self.set_text_color(*BLUE)
                self.cell(0, 6, self._strip_md(text))
                self.set_y(by + 10)

            def render_h3(self, text: str) -> None:
                self.ln(5)
                self.set_font("Helvetica", "B", 10)
                self.set_text_color(*NAVY)
                self.multi_cell(0, 5.5, self._strip_md(text))
                self.ln(2)

            def render_h4(self, text: str) -> None:
                self.ln(4)
                self.set_font("Helvetica", "B", 9.5)
                self.set_text_color(58, 90, 124)
                self.multi_cell(0, 5.5, self._strip_md(text))
                self.ln(2)

            def render_paragraph(self, text: str) -> None:
                """Render body paragraph with inline **bold** / *italic* support."""
                import re as _ire
                # A line that is ENTIRELY **bold** reads like a sub-heading –
                # promote it to h4 for better visual hierarchy.
                if _ire.fullmatch(r'\*\*[^*]+\*\*', text.strip()):
                    self.render_h4(text.strip()[2:-2])
                    return
                # Otherwise render inline rich text.
                saved_lm = self.l_margin
                self.set_left_margin(self.l_margin)  # no-op; ensures write() wraps correctly
                self._render_rich(text, size=9.5, line_h=5.5)
                self.ln(2.5)
                self.set_left_margin(saved_lm)

            def render_bullet(self, text: str, indent: int = 0) -> None:
                x_base = self.l_margin + 4 + (indent * 5)
                self.set_x(x_base)
                bx = self.get_x() + 1
                by = self.get_y() + 2.2
                self.set_fill_color(*BLUE)
                self.ellipse(bx, by, 1.6, 1.6, style="F")
                text_x = x_base + 5
                # Temporarily shift left margin so write() wraps at the indent
                saved_lm = self.l_margin
                self.set_left_margin(text_x)
                self.set_x(text_x)
                self._render_rich(text.strip(), size=9.5, line_h=5)
                self.ln(1.5)
                self.set_left_margin(saved_lm)

            def render_numbered(self, num: str, text: str) -> None:
                x_base = self.l_margin + 3
                self.set_x(x_base)
                self.set_font("Helvetica", "B", 9.5)
                self.set_text_color(*BLUE)
                self.cell(7, 5, _safe(f"{num}"))
                text_x = self.get_x()
                saved_lm = self.l_margin
                self.set_left_margin(text_x)
                self._render_rich(text.strip(), size=9.5, line_h=5)
                self.ln(1.5)
                self.set_left_margin(saved_lm)

            def render_blockquote(self, text: str) -> None:
                safe_text = _safe(text)  # sanitise once; used for width calc AND rendering
                bx = self.l_margin + 2
                self.set_x(bx)
                bw = self.w - self.l_margin - self.r_margin - 4
                self.set_font("Helvetica", "I", 9)
                lines_n = max(1, self.get_string_width(safe_text) / bw + 1)
                bh = lines_n * 5 + 6
                by = self.get_y()
                self.set_fill_color(*BQBLUE)
                self.rect(bx, by, bw, bh, style="F")
                self.set_fill_color(*BLUE)
                self.rect(bx, by, 1.5, bh, style="F")
                self.set_xy(bx + 5, by + 3)
                self.set_text_color(*DARK)
                self.set_font("Helvetica", "I", 9)
                self.multi_cell(bw - 8, 5, safe_text)
                self.set_y(max(self.get_y(), by + bh) + 3)

            def render_hr(self) -> None:
                self.ln(4)
                self.set_draw_color(200, 216, 234)
                self.set_line_width(0.3)
                y = self.get_y()
                self.line(self.l_margin, y, self.w - self.r_margin, y)
                self.ln(5)

            def render_code_block(self, text: str) -> None:
                self.ln(2)
                bx = self.l_margin + 2
                bw = self.w - self.l_margin - self.r_margin - 4
                self.set_font("Courier", "", 7.5)
                code_lines = text.split("\n")
                bh = len(code_lines) * 4.2 + 6
                by = self.get_y()
                if by + bh > self.h - self.b_margin:
                    self.add_page()
                    by = self.get_y()
                self.set_fill_color(*CODEBG)
                self.rect(bx, by, bw, bh, style="F")
                self.set_fill_color(127, 179, 211)
                self.rect(bx, by, 1.2, bh, style="F")
                self.set_xy(bx + 4, by + 3)
                self.set_text_color(*DARK)
                for cl in code_lines:
                    self.cell(0, 4.2, _safe(cl), new_x="LMARGIN", new_y="NEXT")
                    self.set_x(bx + 4)
                self.set_y(max(self.get_y(), by + bh) + 3)

        pdf = StylishPDF()
        pdf.add_page()
        pdf.title_banner()

        lines = md_clean.split("\n")
        i = 0
        in_code_block = False
        code_buf: list[str] = []
        blockquote_buf: list[str] = []

        while i < len(lines):
            line = lines[i]

            if line.strip().startswith("```"):
                if in_code_block:
                    pdf.render_code_block("\n".join(code_buf))
                    code_buf = []
                    in_code_block = False
                else:
                    in_code_block = True
                i += 1
                continue
            if in_code_block:
                code_buf.append(line)
                i += 1
                continue

            stripped = line.strip()

            if not stripped:
                if blockquote_buf:
                    pdf.render_blockquote(" ".join(blockquote_buf))
                    blockquote_buf = []
                i += 1
                continue

            if stripped.startswith(">"):
                blockquote_buf.append(stripped.lstrip("> ").strip())
                i += 1
                continue
            elif blockquote_buf:
                pdf.render_blockquote(" ".join(blockquote_buf))
                blockquote_buf = []

            hm = re.match(r"^(#{1,6})\s+(.*)", stripped)
            if hm:
                level = len(hm.group(1))
                text = hm.group(2).strip()
                if level == 1:
                    pdf.render_h1(text)
                elif level == 2:
                    pdf.render_h2(text)
                elif level == 3:
                    pdf.render_h3(text)
                else:
                    pdf.render_h4(text)
                i += 1
                continue

            if re.match(r"^[-*_]{3,}\s*$", stripped):
                pdf.render_hr()
                i += 1
                continue

            bm = re.match(r"^(\s*)([-+])\s+(.*)", line)
            if bm:
                indent = len(bm.group(1)) // 2
                text = bm.group(3)
                while (
                    i + 1 < len(lines)
                    and lines[i + 1].strip()
                    and not re.match(r"^\s*[-+]\s+", lines[i + 1])
                    and not re.match(r"^\s*\d+[.)]\s+", lines[i + 1])
                    and not lines[i + 1].strip().startswith("#")
                    and not lines[i + 1].strip().startswith("```")
                    and not re.match(r"^[-*_]{3,}\s*$", lines[i + 1].strip())
                    and (lines[i + 1].startswith("  ") or lines[i + 1].startswith("\t"))
                ):
                    i += 1
                    text += " " + lines[i].strip()
                pdf.render_bullet(text, indent=indent)
                i += 1
                continue

            nm = re.match(r"^(\d+[.)])\s+(.*)", stripped)
            if nm:
                num = nm.group(1)
                text = nm.group(2)
                while (
                    i + 1 < len(lines)
                    and lines[i + 1].strip()
                    and not re.match(r"^\s*[-+]\s+", lines[i + 1])
                    and not re.match(r"^\s*\d+[.)]\s+", lines[i + 1])
                    and not lines[i + 1].strip().startswith("#")
                    and not lines[i + 1].strip().startswith("```")
                    and not re.match(r"^[-*_]{3,}\s*$", lines[i + 1].strip())
                    and (lines[i + 1].startswith("  ") or lines[i + 1].startswith("\t")
                         or not re.match(r"^[A-Z#>]", lines[i + 1].strip()))
                ):
                    i += 1
                    text += " " + lines[i].strip()
                pdf.render_numbered(num, text)
                i += 1
                continue

            para_lines = [stripped]
            while (
                i + 1 < len(lines)
                and lines[i + 1].strip()
                and not lines[i + 1].strip().startswith("#")
                and not lines[i + 1].strip().startswith("```")
                and not lines[i + 1].strip().startswith(">")
                and not re.match(r"^\s*[-+]\s+", lines[i + 1])
                and not re.match(r"^\s*\d+[.)]\s+", lines[i + 1])
                and not re.match(r"^[-*_]{3,}\s*$", lines[i + 1].strip())
            ):
                i += 1
                para_lines.append(lines[i].strip())
            pdf.render_paragraph(" ".join(para_lines))
            i += 1

        if blockquote_buf:
            pdf.render_blockquote(" ".join(blockquote_buf))
        if in_code_block and code_buf:
            pdf.render_code_block("\n".join(code_buf))

        pdf.output(pdf_path)
        return True

    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        print(f"[PDF] fpdf2 error: {exc}", file=sys.stderr)

    return False
