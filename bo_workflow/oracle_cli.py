"""Compatibility facade for oracle CLI commands.

New code should import from `bo_workflow.interfaces.cli.oracle_commands`.
"""

from .interfaces.cli.oracle_commands import handle, register_commands

__all__ = ["register_commands", "handle"]
