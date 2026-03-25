---
name: bo-run-evaluator
description: Run an external evaluator loop for a BO run using a pre-provisioned backend id.
---

# BO Run Evaluator

Use this skill when the user or operator has already provisioned an external
evaluation backend and wants the BO loop automated.

This is a low-level execution helper. It does **not** build backends or reveal labeled datasets.

## Command

```bash
uv run python -m bo_workflow.cli run-evaluator \
  --run-id <RUN_ID> \
  --backend-id <BACKEND_ID> \
  --iterations <T> \
  [--batch-size <N>]
```

## Use this only when

- the run already exists
- the backend id is explicitly provided by the user, operator, or benchmark task bundle
- external evaluation is intended

## Do not use this for

- building an oracle/backend
- discovering the search space
- replacing the normal `suggest` / `observe` flow when observations are supposed to come from a human or lab system directly

## Return

- number of recorded observations
- best value / best iteration
- report path
- optional legacy figure path only if one already exists

The resulting observations are recorded back into `bo_runs/<run_id>/observations.jsonl` with source `benchmark-evaluator`.
