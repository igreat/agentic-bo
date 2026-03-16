---
name: bo-build-proxy-oracle
description: Train and persist a proxy oracle for an initialized BO run.
---

# BO Build Proxy Oracle

Use this skill to train a surrogate model from the run dataset. This is only needed for proxy-mode workflows (`run-proxy`). Suggest/observe workflows do not require an oracle.

This only works for runs initialized from a labeled dataset. Search-space-only runs do not have training labels and cannot build a proxy oracle.

## Command

```bash
uv run python -m bo_workflow.cli build-oracle --run-id <RUN_ID>
```

Optional flags: `--backend-id <ID>` (defaults to `<RUN_ID>`), `--cv-folds <N>` (default 5), `--max-features <K>` (limit active features for high-dimensional datasets).

## Return

- selected model name (`random_forest` or `extra_trees`)
- CV RMSE for each candidate model
- active features used by the oracle
- artifacts: `evaluation_backends/<backend_id>/oracle.pkl`, `evaluation_backends/<backend_id>/oracle_meta.json`
