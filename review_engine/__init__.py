"""
Unified Multi-Agent Peer Review Engine.

Consolidates NIH grant review, foundation grant review, and journal peer
review into one config-driven package.

Usage (CLI):
    python -m review_engine nih proposal.pdf --backend claude
    python -m review_engine foundation proposal.pdf --backend gemini
    python -m review_engine journal manuscript.pdf --backend claude

Usage (Python):
    from review_engine.config import get_config
    from review_engine.engine import run_review
"""

__version__ = "1.0.0"
