"""
foundation_review — Multi-Agent Private Foundation Grant Review System

Two backends share a single set of prompts and domain logic:

  python -m foundation_review proposal.pdf --backend gemini
  python -m foundation_review proposal.pdf --backend claude

Or run a backend directly:

  python foundation_review/gemini.py proposal.pdf
  python foundation_review/claude.py proposal.pdf
"""

__version__ = "2.0.0"
__author__  = "Ali Mohebali"
