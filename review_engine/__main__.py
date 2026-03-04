"""
Unified CLI entry point for the Multi-Agent Peer Review Engine.

Usage:
    python -m review_engine nih proposal.pdf --backend claude
    python -m review_engine foundation proposal.pdf --backend gemini
    python -m review_engine journal manuscript.pdf --backend claude --model claude-opus-4-6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from review_engine.config import ensure_types_loaded, get_config, REVIEW_TYPES


def _pick_pdf(title: str = "Select PDF") -> str:
    """Open a native file picker and return the chosen path."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
    )
    root.destroy()
    if not path:
        sys.exit("No file selected. Exiting.")
    return path


def main() -> None:
    ensure_types_loaded()

    parser = argparse.ArgumentParser(
        description="Multi-Agent Peer Review Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s nih proposal.pdf --backend claude\n"
            "  %(prog)s foundation proposal.pdf --backend gemini\n"
            "  %(prog)s journal manuscript.pdf --backend claude --model claude-opus-4-6\n"
            "  %(prog)s nih proposal.pdf --max-rounds 3"
        ),
    )
    parser.add_argument(
        "type", choices=sorted(REVIEW_TYPES.keys()),
        help="Review type: nih, foundation, or journal",
    )
    parser.add_argument(
        "pdf", nargs="?",
        help="PDF file to review (omit to open file picker)",
    )
    parser.add_argument(
        "--backend", "-b", choices=["claude", "gemini"], default="gemini",
        help="AI backend (default: gemini)",
    )
    parser.add_argument(
        "--model", "-m",
        help="Specific model ID (default depends on backend)",
    )
    parser.add_argument(
        "--max-rounds", "-r", type=int, metavar="N",
        help="Maximum review rounds (default depends on review type)",
    )
    parser.add_argument(
        "--output", "-o", metavar="FILE",
        help="Output markdown file (default: <pdf_stem>_review.md)",
    )
    args = parser.parse_args()

    config = get_config(args.type)

    # Validate backend
    if args.backend == "claude":
        from review_engine.backends.claude import validate_startup, DEFAULT_MODEL
        validate_startup()
        default_model = config.claude_default_model
    else:
        from review_engine.backends.gemini import validate_startup, DEFAULT_MODEL
        validate_startup()
        default_model = config.gemini_default_model

    pdf_path = str(Path(args.pdf or _pick_pdf(f"Select PDF {config.document_noun}")).resolve())
    model = args.model or default_model
    max_rounds = args.max_rounds or config.iteration.default_max_rounds
    output_path = (
        str(Path(args.output).resolve()) if args.output
        else str(Path(pdf_path).parent / f"{Path(pdf_path).stem}_review.md")
    )

    from review_engine.engine import run_review
    run_review(config, args.backend, pdf_path, model, max_rounds, output_path)


if __name__ == "__main__":
    main()
