"""
Backend implementations for Claude and Gemini APIs.

Each backend provides: validate_startup(), stream_agent(), and
content-building helpers.  The engine.py orchestrator calls these
through a uniform interface.
"""
