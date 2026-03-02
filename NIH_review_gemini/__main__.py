#!/usr/bin/env python3
"""
Entry point for running gemini_review as a module.

Usage:
    python -m gemini_review proposal.pdf
    python -m gemini_review --help
"""

if __name__ == "__main__":
    from . import main
    main.main()
