---
name: bo-auto-from-prompt
description: Build spec from a plain prompt and run an end-to-end proxy BO workflow.
---

# BO Auto From Prompt

Use this skill when the user gives a natural-language optimization request and wants a full automated run.

## Command

```bash
uv run python -m src.bo_workflow.cli auto-proxy-from-prompt \
  --dataset <CSV_PATH> \
  --prompt "<USER_PROMPT>" \
  --iterations <T> \
  --batch-size <N> \
  --engine <hebo|bo_lcb|random>
```

Optional overrides:

- `--target <TARGET_COLUMN>`
- `--objective <min|max>`
- `--engine <hebo|bo_lcb|random>`
- `--max-features <K>`
- `--spec-out <PATH>`

## Return

- parsed spec JSON
- initialized run metadata (`run_id`)
- oracle selection and CV RMSE
- final report and artifact paths

## Guardrails

- Clearly mark results as proxy-oracle simulation.
- If prompt parsing is ambiguous, include inferred target/objective in response.
