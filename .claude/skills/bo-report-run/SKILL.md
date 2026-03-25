---
name: bo-report-run
description: Generate a final BO report and summarize optimization status.
---

# BO Report Run

Use this skill to summarize run progress.

## Quick status check

```bash
uv run python -m bo_workflow.cli status --run-id <RUN_ID>
```

Returns: run status, best value, number of observations. No side effects.

## Full report

```bash
uv run python -m bo_workflow.cli report --run-id <RUN_ID>
```

Returns: best value, best candidate, oracle info, and generates:
- `bo_runs/<run_id>/report.json`

Use `status` for quick progress checks during a loop. Use `report` for final summaries. If the user wants figures, generate them ad hoc from the run artifacts.
