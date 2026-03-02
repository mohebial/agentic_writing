"""
Multi-Agent NIH Grant Peer Review System — Gemini Edition

A modular system for iterative peer review of NIH grant applications using
Google Gemini API, with multiple specialized reviewer agents.

Main Entry Point:
    from gemini_review import main
    main.run(pdf_path, model, max_rounds, output_path)

Or run as module:
    python -m gemini_review proposal.pdf

Key Modules:
    - helpers: Utility functions for Gemini API, text processing, PDF export
    - main: Main review orchestration and entry point

Required Environment:
    GEMINI_API_KEY environment variable must be set
"""

__version__ = "1.0.0"
__author__ = "Ali Mohebali"

# Make key functions/classes available at package level for convenience
try:
    from . import helpers
    from . import main
except ImportError:
    pass
