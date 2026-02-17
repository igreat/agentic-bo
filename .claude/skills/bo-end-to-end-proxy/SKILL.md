---
name: bo-end-to-end-proxy
description: Run full proxy BO loop from dataset input to final report.
---

# BO End-to-End Proxy

Use this skill when the user asks for a full automated run using dataset-only evaluation.

## Workflow

Run these four commands in sequence:

```bash
# 1. Initialize
uv run python -m bo_workflow.cli init \
  --dataset <CSV_PATH> --target <TARGET_COL> --objective <min|max> --engine <hebo|bo_lcb|random> --seed <SEED>

# 2. Train oracle
uv run python -m bo_workflow.cli build-oracle --run-id <RUN_ID>

# 3. Run proxy BO loop
uv run python -m bo_workflow.cli run-proxy --run-id <RUN_ID> \
  --iterations <T> --batch-size <N>

# 4. Generate report
uv run python -m bo_workflow.cli report --run-id <RUN_ID>
```

Extract `<RUN_ID>` from the `init` output's `run_id` field.

## Resuming a completed run

If the user wants more iterations on an existing completed run, do NOT re-init or re-build the oracle. Instead:

1. Flip the status back to `running`:
```python
import json, pathlib
p = pathlib.Path("runs/<RUN_ID>/state.json")
state = json.loads(p.read_text())
state["status"] = "running"
p.write_text(json.dumps(state, indent=2))
```
2. Call `run-proxy` with the additional iterations. The engine appends to existing observations — no work is repeated.

## Guardrails

- Clearly label results as proxy-oracle simulation.
- Include oracle CV RMSE when presenting optimization results.
- If user asks for real lab mode, do not auto-evaluate suggestions with the proxy oracle.
- `bo_lcb` supports batch-size 1 only.
