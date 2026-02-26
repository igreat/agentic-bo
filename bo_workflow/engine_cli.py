"""Compatibility facade for engine CLI commands.

New code should import from `bo_workflow.interfaces.cli.engine`.
"""

from .interfaces.cli.engine import handle, register_commands

__all__ = ["register_commands", "handle"]
