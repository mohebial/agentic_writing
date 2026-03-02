#!/usr/bin/env python3
"""
Entry point for ``python -m journal_review``.

Dispatches to the Gemini or Claude backend based on --backend.

Usage:
    python -m journal_review manuscript.pdf --backend gemini
    python -m journal_review manuscript.pdf --backend claude
    python -m journal_review manuscript.pdf --backend claude --model claude-opus-4-20250514
    python -m journal_review --backend gemini --help
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path


def main() -> None:
    # ── Pre-parse just --backend so we can delegate the rest to the backend ──
    # We use parse_known_args so that backend-specific flags (--model, --output,
    # pdf positional) pass through untouched to the backend's own parser.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--backend", "-b",
        choices=["gemini", "claude"],
        default="gemini",
        help="AI backend to use  (default: gemini)",
    )
    known, remaining = pre.parse_known_args()

    # ── Rebuild sys.argv for the backend's main() ─────────────────────────────
    # Strip --backend / -b and its value so the backend parser doesn't choke.
    argv = sys.argv[1:]
    for flag in ("--backend", "-b"):
        if flag in argv:
            idx = argv.index(flag)
            argv = argv[:idx] + argv[idx + 2:]

    sys.argv = [sys.argv[0]] + argv

    if known.backend == "gemini":
        from journal_review.gemini import main as _main
    else:
        from journal_review.claude import main as _main

    _main()


if __name__ == "__main__":
    main()
