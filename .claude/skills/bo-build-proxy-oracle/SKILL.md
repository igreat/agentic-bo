---
name: bo-build-proxy-oracle
description: Train and persist a proxy oracle for an initialized BO run.
---

# BO Build Proxy Oracle

Use this skill to train a surrogate model from the run dataset. This is only needed for proxy-mode workflows (`run-proxy`). Human-in-the-loop workflows (manual `suggest`/`observe`) do not require an oracle.

Always run `prep-data` first for proxy workflows:

```bash
uv run python -m bo_workflow.cli prep-data --run-id <RUN_ID>
```

If `ready_to_build` is `false`, do not call `build-oracle` until the dataset issues are resolved.

## Command

```bash
uv run python -m bo_workflow.cli build-oracle --run-id <RUN_ID>
```

Optional flags: `--cv-folds <N>` (default 5), `--max-features <K>` (limit active features for high-dimensional datasets).

## Return

- selected model name (`random_forest`, `extra_trees`, or `hist_gradient_boosting`)
- CV RMSE for each candidate model
- active features used by the oracle
- artifacts: `runs/<run_id>/oracle.pkl`, `runs/<run_id>/oracle_meta.json`
