---
name: bo-init-run
description: Initialize a BO run from a dataset.
---

# BO Init Run

Use this skill when the user asks to start an optimization campaign.

## Command

```bash
uv run python -m bo_workflow.cli init \
  --dataset <CSV_PATH> --target <TARGET_COL> --objective <min|max>
```

Optional flags: `--engine <hebo|bo_lcb|random|botorch>` (default hebo), `--seed <N>` (default 7), `--init-random <N>` (default 10), `--batch-size <N>` (default 1), `--run-id <ID>`, `--intent-json <JSON_OR_PATH>`, `--drop-cols <col1,col2>`, `--simplex-groups <cols:total>` (repeatable).

**Engine constraints:**
- `bo_lcb`: batch-size 1 only
- `botorch`: numeric features only — will error if the dataset has any categorical columns; use `hebo` instead

**Simplex constraints:**

Use `--simplex-groups` when the problem has compositional variables that must sum to a fixed total. This is domain knowledge — infer it from the user's problem description, not from the data.

```bash
# OER: metal proportions must sum to 100
--simplex-groups 'Metal_1_Proportion,Metal_2_Proportion,Metal_3_Proportion:100'

# HEA: elemental fractions must sum to 1
--simplex-groups 'x_Co,x_Cu,x_Mn,x_Fe,x_V:1'

# Multiple independent simplex groups
--simplex-groups 'A,B,C:1' --simplex-groups 'D,E:100'
```

Constraints are stored in `state.json` under `"constraints"` and enforced at every `suggest` call by normalizing the group columns to sum to `total`.

## Return

- `run_id`
- inferred `active_features`
- `constraints` list (empty if none specified)
- state stored at `runs/<run_id>/state.json`

## Notes

- Always use explicit `--target` and `--objective`.
- Pass `--intent-json` to preserve the user's original prompt for provenance.
- Infer simplex groups from the user's problem description. Common signals: "proportion", "fraction", "composition", "sum to 1/100%".
