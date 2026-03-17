# Claude Skills

Project-level instructions are in `CLAUDE.md` at the repository root.

Skills in `skills/` map to top-level BO CLI subcommands and converter entrypoints.

- BO CLI: `bo_workflow/cli.py` composes subparsers from `engine_cli.py` and `bo_workflow/evaluation/cli.py`
- Converters: module CLIs under `bo_workflow/converters/` (e.g., `reaction_drfp`, `molecule_descriptors`)

See `CLAUDE.md` for the full command reference.
