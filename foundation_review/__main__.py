#!/usr/bin/env python3
"""
Backward-compatible entry point for ``python -m foundation_review``.

Delegates to the unified review_engine with type='foundation'.

Usage:
    python -m foundation_review proposal.pdf --backend claude
    python -m foundation_review proposal.pdf --backend gemini
"""

import sys

sys.argv.insert(1, "foundation")

from review_engine.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
