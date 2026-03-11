---
name: bo-record-observation
description: Record observed objective values and update run history.
---

# BO Record Observation

Use this skill when outcomes for suggested experiments are available.

## Before recording

Read `bo_runs/<RUN_ID>/state.json` to get `active_features` and `target_column`. If the human's result is informal (e.g. "got 312 mV with 60% Ru, 40% Ir"), map it to the correct feature names. For any feature not mentioned, use the value from the original suggestion. Confirm ambiguous mappings with the user before recording.

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
