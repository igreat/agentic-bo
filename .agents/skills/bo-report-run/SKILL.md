---
name: bo-report-run
description: Generate convergence report and summarize optimization status.
---

# BO Report Run

Use this skill to summarize run progress.

## Quick status check

```bash
uv run python -m bo_workflow.cli status --run-id <RUN_ID>
```

Returns: run status, best value, number of observations. No side effects.

## Full report with plot

```bash
uv run python -m bo_workflow.cli report --run-id <RUN_ID>
```

Returns: best value, best candidate, oracle info, and generates:
- `runs/<run_id>/report.json`
- `runs/<run_id>/convergence.pdf`

Use `status` for quick progress checks during a loop. Use `report` for final summaries or when the user wants a convergence plot.
