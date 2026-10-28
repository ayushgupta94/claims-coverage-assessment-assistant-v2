"""Shared pytest setup.

Tests must never send traces to a real Langfuse project, even when the
developer's .env holds live keys. Real environment variables take priority
over .env, and empty keys leave tracing a no-op.
"""

import os

os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
