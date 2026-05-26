"""Hermes Smart Router plugin.

This module exposes `register(ctx)`, the entrypoint Hermes calls for pip-based
plugins. Directory installs use the repository-root `__init__.py`, which imports
this same function.
"""

from .router import route_gateway_message, transform_llm_output

__version__ = "0.1.0"


def register(ctx):
    """Register Hermes lifecycle hooks."""
    ctx.register_hook("pre_gateway_dispatch", route_gateway_message)
    ctx.register_hook("transform_llm_output", transform_llm_output)
