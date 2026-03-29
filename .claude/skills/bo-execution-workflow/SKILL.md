---
name: bo-execution-workflow
description: BO execution layer — initializes a run from a resolved experiment spec, records observations, and continues through suggest/observe/report.
---

# BO Execution Workflow

This skill is the **BO execution layer**. It assumes the problem has already been framed by the layer above (e.g. `research-agent`). Do not use this skill for problem discovery, dataset acquisition, or representation selection — those decisions belong upstream.

## Input / Output Contract

**Expects (resolved before this skill is invoked):**

- target column name
- objective direction (`min` / `max`)
- resolved `design_parameters`
- optional `fixed_features`
- structured constraints if applicable
- optional dataset path for validation/context
- optional prior observations
- representation plan if encoding is needed

When invoked by `research-agent`, the canonical init path is explicit search-space JSON derived from `experiment_spec`.

**Produces (run artifacts under `bo_runs/<RUN_ID>/`):**

| File | Created by |
|---|---|
| `state.json` | `init` |
| `input_spec.json` | `init` |
| `intent.json` | `init` (when `--intent-json` is provided) |
| `suggestions.jsonl` | `suggest` |
| `observations.jsonl` | `observe` |
| `report.json` | `report` |

When this skill is invoked by `research-agent`, it supports two handoff points:

- **Setup-only handoff (Phase 3):**
  - run through `init`
  - record any seed observations
  - stop once the BO run is ready and `bo_run_id` is known
- **Execution continuation (Phase 4):**
  - continue from the existing `bo_run_id`
  - do **not** repeat `init`

---

## Step 1 — Confirm Execution Config

Verify all required inputs are resolved before running anything. If any are missing, surface them to the layer above — do not attempt problem discovery here.

The resolved execution config should include a structured search space:
- `design_parameters`
- optional `fixed_features`
- target column
- objective
- constraints if any

If everything is resolved, proceed to Step 2.

---

## Step 2 — Validate the Dataset (if present)

If a dataset path is part of the execution config, inspect it before running anything:

```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('<CSV_PATH>')
print('Shape:', df.shape)
print('Columns:', list(df.columns))
print('Missing values:'); print(df.isnull().sum()[df.isnull().sum() > 0])
print('Dtypes:'); print(df.dtypes)
if '<TARGET_COL>' in df.columns:
    print('Target stats:'); print(df['<TARGET_COL>'].describe())
"
```

**🔴 Blocking — must fix before `init`:**
- If the dataset includes the target column, target rows used for dataset-backed init or oracle training must not contain missing values
- Categorical column with >64 unique values → engine will error; flag this to the user before proceeding
- The dataset clearly does not match the resolved experiment spec → stop and clarify rather than drifting the search space

**🟡 Action required — configure explicitly at `init` time:**
- Columns whose values appear to sum to a fixed total (proportions, fractions) → declare `--simplex-groups` (see Step 3)
- Non-feature columns present in a dataset-backed init CSV (e.g. `rxn_smiles`, IDs) → pass `--drop-cols col1,col2` when dataset inference is intentionally used

**🟢 Auto-handled — informational only:**
- Constant/zero-variance columns → engine drops them silently in dataset-backed inference mode

If no dataset is present, skip to Step 3.

---

## Step 3 — Declare Constraints (Simplex)

If the execution config specifies compositional constraints (columns whose values must sum to a fixed total), declare them at `init` time using `--simplex-groups`:

```bash
# Metal proportions summing to 100 (e.g. OER dataset)
--simplex-groups 'Metal_1_Proportion,Metal_2_Proportion,Metal_3_Proportion:100'

# Elemental fractions summing to 1 (e.g. HEA dataset)
--simplex-groups 'x_Co,x_Cu,x_Mn,x_Fe,x_V:1'

# Multiple independent simplex groups
--simplex-groups 'A,B,C:1' --simplex-groups 'D,E:100'
```

Format: `'col1,col2,...:total'` — comma-separated column names, colon, then the required sum.

Constraints are stored in `state.json["constraints"]` at `init` time and enforced automatically at every `suggest` call.

If no simplex constraints apply, skip to Step 4.

---

## Step 4 — Encode if Needed

If the representation plan specifies encoding:

- **Reaction SMILES columns:** use `bo-encode-drfp` skill before `init`; use `bo-decode-drfp` after BO to map suggestions back to reactions
- **Molecule SMILES columns:** use `bo-encode-molecule-descriptors` skill before `init`; use `bo-decode-molecule-descriptors` after BO
- **No encoding needed:** skip to Step 5

The choice of representation belongs to the layer above. If the representation plan is not specified, surface this question upstream — do not auto-decide here.

---

## Step 5 — Initialize the Run

Prefer explicit search-space init when the experiment spec is already resolved:

```bash
uv run python -m bo_workflow.cli init \
  --search-space-json '<JSON_OR_PATH>' \
  --target <TARGET_COL> \
  --objective <min|max> \
  [--simplex-groups 'col1,col2:total'] \
  [--engine <hebo|bo_lcb|random|botorch>] \
  [--intent-json '<JSON_OR_PATH>'] \
  --seed 42
```

Use dataset-backed init only when a labeled CSV is intentionally the chosen BO input source:

```bash
uv run python -m bo_workflow.cli init \
  --dataset <CSV_PATH> \
  --target <TARGET_COL> \
  --objective <min|max> \
  [--simplex-groups 'col1,col2:total'] \
  [--drop-cols col1,col2] \
  [--engine <hebo|bo_lcb|random|botorch>] \
  [--intent-json '<JSON_OR_PATH>'] \
  --seed 42
```

When this skill is used under `research-agent`, serialize `experiment_spec` into the `--search-space-json` input rather than relying on a labeled dataset.
If `experiment_spec.bo_engine` is already set upstream, pass it through as `--engine <ENGINE>` during `init` rather than falling back to the default engine.

Extract `run_id` from the JSON output.

---

## Step 6 — Seed Prior Observations (if any)

If prior observations are already part of the resolved execution config, record them immediately after `init` with `bo-record-observation` before the first `suggest` call.

If this skill is being used for a **Phase 3 setup-only handoff** from `research-agent`, stop after `init` plus any seed observations, return the existing `bo_run_id`, and do not start the iterative loop yet.

---

## Step 7 — Continue Through Suggest / Observe / Report

If this skill is being used for a **Phase 4 continuation** from `research-agent`, reuse the existing `bo_run_id` and continue directly with `suggest` / `observe` / `report` instead of re-initializing.

If a benchmark task bundle provides a prebuilt evaluator backend id in the
public workspace, it is acceptable to automate this phase directly with:

```bash
uv run python -m bo_workflow.cli run-evaluator \
  --run-id <RUN_ID> \
  --backend-id <BACKEND_ID> \
  --iterations <T> \
  [--batch-size <N>]
```

This assumes the public workspace already contains
`evaluation_backends/<BACKEND_ID>/`.

If the user or operator explicitly provides a raw `backend_id` for an external
evaluator, it is also acceptable to automate this phase with `bo-run-evaluator`
instead of manually alternating `suggest` and `observe`.

Then repeat this loop until the user or external controller is satisfied:

```bash
uv run python -m bo_workflow.cli suggest --run-id <RUN_ID> --batch-size <N>
```

Present the suggestion clearly:
- show feature values in plain language
- if representations were decoded, show reagent names rather than raw encodings
- explain what the BO engine expects next

Record observations only when values are supplied by the user or another external observer:

```bash
uv run python -m bo_workflow.cli observe \
  --run-id <RUN_ID> \
  --data '{"x": {<FEATURE_VALUES>}, "y": <RESULT>}'
```

After sufficient iterations, generate the report:

```bash
uv run python -m bo_workflow.cli report --run-id <RUN_ID>
```

Do **not** invoke `build-oracle` or `run-proxy` from this skill. Use
`bo-run-evaluator` only when a prebuilt backend id is available for external
evaluation.

---

## Step 8 — Present Results

Always include in your final summary:

1. **Best result found** — the value and which experiment produced it
2. **Best conditions** — the exact feature values to replicate it
3. **Convergence trajectory** — how the best value improved over iterations (from `report.json["trajectory"]` or from ad hoc figures generated from `observations.jsonl`)
4. **Oracle quality** — only if the run artifacts explicitly report oracle metadata
5. **Simulation label** — only if the run artifacts explicitly indicate proxy-backed evaluation

---

## Guardrails

- Never invent observation values.
- Never auto-commit observations without the user or external observer providing the result value.
- If a categorical column has >64 unique values, flag this before `init` — the engine will error.
- Prefer `--search-space-json` when an upstream agent has already resolved the search space.
- Pass `--intent-json` when the upstream agent has captured the original user intent — this preserves provenance in `bo_runs/<RUN_ID>/intent.json`.
