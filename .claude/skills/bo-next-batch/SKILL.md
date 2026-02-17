---
name: bo-next-batch
description: Generate next experimental suggestions from current run state.
---

# BO Next Batch

Use this skill to get candidate experiments.

## Command

```bash
uv run python -m bo_workflow.cli suggest --run-id <RUN_ID> --batch-size <N>
```

Accepts status `initialized`, `oracle_ready`, or `running`. No oracle needed — HEBO/BO/random work directly from the design space.

## Return

- list of suggestions with `suggestion_id`
- feature values for each candidate
- reminder to record outcomes later via `bo-record-observation`
