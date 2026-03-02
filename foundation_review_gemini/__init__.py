"""
Multi-Agent Private Foundation Grant Review System — Gemini Edition

Single-pass review emphasising novelty and innovation, designed for
private foundation grant proposals.

Roles:
  - Scientific Reviewer    (domain expert, all four criteria)
  - Innovation Reviewer    (originality and transformative potential)
  - Program Advisor        (mission fit, budget, organizational readiness)
  - Independence Auditor   (echo bias and score-anchoring correction)
  - Panel Chair            (recommendation letter: Fund / Fund with Conditions / Decline)
  - Project Director       (cover letter + revised narrative)

Requires: GEMINI_API_KEY environment variable
"""

__version__ = "1.0.0"
__author__  = "Ali Mohebali"

try:
    from . import helpers
    from . import main
except ImportError:
    pass
