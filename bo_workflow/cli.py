"""Compatibility facade for top-level CLI.

New code should import from `bo_workflow.interfaces.cli.root`.
"""

import sys

from .interfaces.cli.root import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
