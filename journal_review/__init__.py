"""
journal_review — Multi-Agent Journal Peer Review System

Two backends share a single set of prompts and domain logic:

  python -m journal_review manuscript.pdf --backend gemini
  python -m journal_review manuscript.pdf --backend claude

Or run a backend directly:

  python journal_review/gemini.py manuscript.pdf
  python journal_review/claude.py manuscript.pdf
"""

__version__ = "0.1.0"
__author__  = "agentic_writing"
