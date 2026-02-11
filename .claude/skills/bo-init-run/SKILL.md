---
name: bo-init-run
description: Initialize a BO run from dataset or JSON spec.
---

# BO Init Run

Use this skill when the user asks to start an optimization campaign.

## Path 1: Dataset with explicit fields

```bash
uv run python -m src.bo_workflow.cli init \
  --dataset <CSV_PATH> --target <TARGET_COL> --objective <min|max>
```

Optional flags: `--seed <N>` (default 0), `--init-random <N>` (default 10), `--batch-size <N>` (default 1), `--run-id <ID>`, `--intent-json <JSON_OR_PATH>`.

## Path 2: JSON spec

```bash
uv run python -m src.bo_workflow.cli init-from-spec --spec <SPEC_PATH_OR_JSON>
```

## Path 3: Natural-language prompt

When the user describes the task in plain language rather than giving explicit target/objective:

```bash
uv run python -m src.bo_workflow.cli init-from-prompt \
  --dataset <CSV_PATH> --prompt "<USER_PROMPT>"
```

Optional: `--target`, `--objective` (inferred from prompt if omitted), `--max-features <K>`, `--spec-out <PATH>`.

## Return

- `run_id`
- inferred `active_features`
- state stored at `runs/<run_id>/state.json`

## Notes

- Prefer explicit `--target` and `--objective`; only use `init-from-prompt` when the user asks for auto-inference.
- Pass `--intent-json` to preserve the user's original prompt for provenance.
