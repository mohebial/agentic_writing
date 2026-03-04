#!/usr/bin/env python3
"""
Backward-compatible entry point for ``python -m NIH_review``.

Delegates to the unified review_engine with type='nih'.

Usage:
    python -m NIH_review proposal.pdf --backend claude
    python -m NIH_review proposal.pdf --backend gemini
"""

import sys

# Inject the review type as the first positional arg
sys.argv.insert(1, "nih")

from review_engine.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
