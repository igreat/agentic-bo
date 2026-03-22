---
name: bo-init-run
description: Initialize a BO run from a dataset or explicit search-space JSON.
---

# BO Init Run

Use this skill when the user asks to start an optimization campaign.

## Command

Dataset-backed init:

```bash
uv run python -m bo_workflow.cli init \
  --dataset <CSV_PATH> --target <TARGET_COL> --objective <min|max>
```

Search-space init:

```bash
uv run python -m bo_workflow.cli init \
  --search-space-json '<JSON_OR_PATH>' --target <TARGET_COL> --objective <min|max>
```

Optional flags: `--engine <hebo|bo_lcb|random|botorch>` (default hebo), `--hebo-model <gp|rf>` (HEBO only, default gp), `--seed <N>` (default 7), `--init-random <N>` (default 10), `--batch-size <N>` (default 1), `--run-id <ID>`, `--intent-json <JSON_OR_PATH>`, `--drop-cols <col1,col2>`, `--simplex-groups <cols:total>` (repeatable).

**Engine constraints:**
- `bo_lcb`: batch-size 1 only
- `botorch`: prefer this for small-to-medium search spaces with many categorical choices, especially when evaluations are expensive and sample efficiency matters
- `hebo`: still a good general default for broader mixed tabular spaces and many routine BO runs
- `hebo --hebo-model rf`: preferred first fallback when `hebo --hebo-model gp` shows repeated jitter / GP fitting failures on mixed spaces

**Reasonable engine-choice heuristic:**
- Prefer `botorch` when the search space is mostly or entirely categorical, the all-categorical candidate count is still modest enough to reason about (default threshold `<= 2000` combinations), and each evaluation is expensive enough that finding a strong basin early matters more than cheap optimizer overhead.
- Prefer `hebo` when the space is broader and more mixed numeric/categorical, when you want the repo's general-purpose default, or when there is no clear reason to bias toward BoTorch.
- If `hebo --hebo-model gp` looks numerically unstable on a mixed space, try `hebo --hebo-model rf` before abandoning HEBO entirely.

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
- `input_spec.json` persisted under the run directory
- state stored at `bo_runs/<run_id>/state.json`

## Notes

- Always use explicit `--target` and `--objective`.
- Pass `--intent-json` to preserve the user's original prompt for provenance.
- Prefer `--search-space-json` when an upstream agent has already resolved `design_parameters` and `fixed_features`.
- If an upstream phase already chose `experiment_spec.bo_engine`, pass it explicitly via `--engine` instead of relying on the repo default.
- Infer simplex groups from the user's problem description. Common signals: "proportion", "fraction", "composition", "sum to 1/100%".
