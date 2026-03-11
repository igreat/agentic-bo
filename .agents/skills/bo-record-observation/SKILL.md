---
name: bo-record-observation
description: Record observed objective values and update run history.
---

# BO Record Observation

Use this skill when outcomes for suggested experiments are available.

## Command

```bash
uv run python -m bo_workflow.cli observe --run-id <RUN_ID> --data <JSON_OR_FILE>
```

## Accepted `--data` formats

- **Inline JSON object:** `'{"x": {"feat1": 1.0, "feat2": 2.0}, "y": 5.2}'`
- **Inline JSON list:** `'[{"x": {...}, "y": 5.2}, {"x": {...}, "y": 3.1}]'`
- **Path to `.json` file:** list of `{"x": {...}, "y": ...}` objects
- **Path to `.csv` file:** must have a `y` column; all other columns become `x`

The `y` key can also be the target column name (e.g. `"Target"` instead of `"y"`).

## Return

- number of recorded observations
- updated best value

Quick status check: `uv run python -m bo_workflow.cli status --run-id <RUN_ID>`
