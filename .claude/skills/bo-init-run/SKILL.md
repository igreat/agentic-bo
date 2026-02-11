---
name: bo-init-run
description: Initialize a BO run from a dataset.
---

# BO Init Run

Use this skill when the user asks to start an optimization campaign.

## Command

```bash
uv run python -m src.bo_workflow.cli init \
  --dataset <CSV_PATH> --target <TARGET_COL> --objective <min|max>
```

Optional flags: `--engine <hebo|bo_lcb|random>` (default hebo), `--seed <N>` (default 7), `--init-random <N>` (default 10), `--batch-size <N>` (default 1), `--run-id <ID>`, `--intent-json <JSON_OR_PATH>`.

## Return

- `run_id`
- inferred `active_features`
- state stored at `runs/<run_id>/state.json`

## Notes

- Always use explicit `--target` and `--objective`.
- Pass `--intent-json` to preserve the user's original prompt for provenance.
