#!/usr/bin/env python3
"""
Backward-compatible entry point for ``python -m journal_review``.

Delegates to the unified review_engine with type='journal'.

Usage:
    python -m journal_review manuscript.pdf --backend claude
    python -m journal_review manuscript.pdf --backend gemini
"""

import sys

sys.argv.insert(1, "journal")

from review_engine.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
