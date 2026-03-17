"""Module entrypoint for evaluation-only commands."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
