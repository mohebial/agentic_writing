"""
Shared, domain-agnostic utilities used by all review packages.

Re-exports from submodules:
  - _shared.text:  banner, extract_section, format_critiques, preprocess_markdown
  - _shared.pdf:   convert_to_pdf
"""

from _shared.text import (       # noqa: F401
    banner,
    extract_section,
    format_critiques,
    preprocess_markdown,
)
from _shared.pdf import (        # noqa: F401
    convert_to_pdf,
)
