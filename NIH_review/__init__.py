"""
NIH_review — Multi-Agent NIH Grant Peer Review System

Two backends share a single set of prompts and domain logic:

  python -m NIH_review proposal.pdf --backend gemini
  python -m NIH_review proposal.pdf --backend claude

Or run a backend directly:

  python NIH_review/gemini.py proposal.pdf
  python NIH_review/claude.py proposal.pdf
"""

__version__ = "2.0.0"
__author__  = "Ali Mohebali"
