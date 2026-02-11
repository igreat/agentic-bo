---
name: bo-end-to-end-proxy
description: Run full proxy BO loop from dataset input to final report.
---

# BO End-to-End Proxy

Use this skill when the user asks for a full automated run using dataset-only evaluation.

If the user provides a plain-language prompt, prefer `bo-auto-from-prompt` instead.

## Workflow

Run these four commands in sequence:

```bash
# 1. Initialize
uv run python -m src.bo_workflow.cli init \
  --dataset <CSV_PATH> --target <TARGET_COL> --objective <min|max> --engine <hebo|bo_lcb|random> --seed <SEED>

# 2. Train oracle
uv run python -m src.bo_workflow.cli build-oracle --run-id <RUN_ID>

# 3. Run proxy BO loop
uv run python -m src.bo_workflow.cli run-proxy --run-id <RUN_ID> \
  --iterations <T> --batch-size <N> --engine <hebo|bo_lcb|random>

# 4. Generate report
uv run python -m src.bo_workflow.cli report --run-id <RUN_ID>
```

Extract `<RUN_ID>` from the `init` output's `run_id` field.

## Guardrails

- Clearly label results as proxy-oracle simulation.
- Include oracle CV RMSE when presenting optimization results.
- If user asks for real lab mode, do not auto-evaluate suggestions with the proxy oracle.
- `bo_lcb` supports batch-size 1 only.
