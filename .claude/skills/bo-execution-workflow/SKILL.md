---
name: bo-execution-workflow
description: BO execution layer — runs the full BO loop given a resolved execution config. Invoked by an upstream research agent (e.g. bo-research-agent) once problem framing, dataset acquisition, and representation decisions are complete.
---

# BO Execution Workflow

This skill is the **BO execution layer**. It assumes the problem has already been framed by the layer above (e.g. `bo-research-agent`). Do not use this skill for problem discovery, dataset acquisition, or representation selection — those decisions belong upstream.

## Input / Output Contract

**Expects (resolved before this skill is invoked):**

| Input | Proxy mode | Human-in-the-loop |
|---|---|---|
| Dataset path (full labeled CSV with target values) | ✅ Required | ❌ Not required |
| Search-space CSV (feature columns, types, ranges/categories — no target values needed) | Optional | ✅ Required if no prior observations |
| Target column name | ✅ | ✅ |
| Objective direction (`min` / `max`) | ✅ | ✅ |
| Simplex groups (if applicable) | If applicable | If applicable |
| Representation plan (encoding needed?) | If applicable | If applicable |

> **Note on `--dataset` in human-in-the-loop mode:** The `init` CLI always requires `--dataset`, but the CSV does not need to contain existing target values. A template CSV with only feature column headers is sufficient to define the search space.

**Produces (run artifacts under `runs/<RUN_ID>/`):**

| File | Created by |
|---|---|
| `state.json` | `init` |
| `intent.json` | `init` (when `--intent-json` is provided) |
| `oracle.pkl` + `oracle_meta.json` | `build-oracle` (proxy mode only) |
| `suggestions.jsonl` | `suggest` / `run-proxy` |
| `observations.jsonl` | `observe` / `run-proxy` |
| `convergence.pdf` | `report` / `run-proxy` |
| `report.json` | `report` / `run-proxy` |

---

## Step 1 — Confirm Execution Config

Verify all required inputs are resolved before running anything. If any are missing, surface them to the layer above — do not attempt problem discovery here.

If everything is resolved, proceed to Step 2.

---

## Step 2 — Validate the Dataset

> **Skip this step entirely in human-in-the-loop mode with no prior data.**

Inspect the dataset before running anything:

```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('<CSV_PATH>')
print('Shape:', df.shape)
print('Columns:', list(df.columns))
print('Missing values:'); print(df.isnull().sum()[df.isnull().sum() > 0])
print('Dtypes:'); print(df.dtypes)
print('Target stats:'); print(df['<TARGET_COL>'].describe())
"
```

**🔴 Blocking — must fix before `init`:**
- Missing values in the target column → drop rows or abort; `init` will fail or corrupt state if target has NaNs
- Categorical column with >64 unique values → engine will error; flag this to the user before proceeding

**🟡 Action required — configure explicitly at `init` time:**
- Columns whose values appear to sum to ~100% (proportions, fractions) → declare `--simplex-groups` (see Step 3)
- Non-feature columns present in the CSV (e.g. `rxn_smiles`, IDs) → pass `--drop-cols col1,col2` at `init`

**🟢 Auto-handled — informational only:**
- Constant/zero-variance columns → engine drops them silently; no action needed

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

> **Do NOT use `bo-transform-columns` for simplex.** That skill handles scale transforms (log, sqrt, standardize) only. Simplex is a search-space constraint enforced at suggest time, not a column transform.

Constraints are stored in `state.json["constraints"]` at `init` time and enforced automatically at every `suggest` call by renormalizing group columns to sum to `total`. No action is needed after `init`.

If no simplex constraints apply, skip to Step 4.

---

## Step 4 — Encode if Needed

If the representation plan specifies encoding:

- **Reaction SMILES columns:** use `bo-encode-drfp` skill before `init`; use `bo-decode-drfp` after BO to map suggestions back to reactions
- **Molecule SMILES columns:** use `bo-encode-molecule-descriptors` skill before `init`; use `bo-decode-molecule-descriptors` after BO
- **No encoding needed (categorical/numeric features already in the CSV):** skip to Step 5

The choice of representation belongs to the layer above. If the representation plan is not specified, surface this question upstream — do not auto-decide here.

---

## Step 5 — Proxy Mode Workflow (automated simulation)

Use when the operating mode is `proxy`. The oracle is trained on the existing dataset and evaluates BO suggestions automatically. Do not use this mode for real lab experiments.

```bash
# 1. Initialize run
uv run python -m bo_workflow.cli init \
  --dataset <CSV_PATH> \
  --target <TARGET_COL> \
  --objective <min|max> \
  [--simplex-groups 'col1,col2:total'] \
  [--drop-cols col1,col2] \
  [--engine <hebo|bo_lcb|random|botorch>] \
  [--intent-json '<JSON_OR_PATH>'] \
  --seed 42

# Extract run_id from the JSON output

# 2. Train proxy oracle
uv run python -m bo_workflow.cli build-oracle --run-id <RUN_ID>

# 3. Run full proxy loop
uv run python -m bo_workflow.cli run-proxy \
  --run-id <RUN_ID> \
  --iterations <N>

# 4. Generate report
uv run python -m bo_workflow.cli report --run-id <RUN_ID>
```

Recommended iterations: 20 for a quick demo, 50+ for thorough optimization.

After running, present to the user:
- Best value found and at which iteration
- Best experiment conditions (exact feature values)
- Oracle CV RMSE (surrogate quality indicator)
- **Always label results as proxy-oracle simulation, not real experimental results**

---

## Step 6 — Human-in-the-Loop Workflow (real experiments)

Use when the operating mode is `human-in-the-loop`. The chemist runs real experiments and reports results. **Do NOT invoke the proxy oracle in this mode.**

```bash
# 1. Initialize run
uv run python -m bo_workflow.cli init \
  --dataset <CSV_PATH> \
  --target <TARGET_COL> \
  --objective <min|max> \
  [--simplex-groups 'col1,col2:total'] \
  [--drop-cols col1,col2] \
  [--intent-json '<JSON_OR_PATH>'] \
  --seed 42
```

Then repeat this loop until the user is satisfied:

```bash
# Get next suggestion
uv run python -m bo_workflow.cli suggest --run-id <RUN_ID> --batch-size <N>
```

Present the suggestion clearly to the chemist:
- Show feature values in plain language (e.g. "Try ligand=BrettPhos, base=MTBD, additive=None")
- If representations were decoded, show reagent names not SMILES strings
- Explain what the BO engine expects from them next (run the experiment, report the result)

Wait for the chemist to report the result, then record it:

```bash
uv run python -m bo_workflow.cli observe \
  --run-id <RUN_ID> \
  --data '{"x": {<FEATURE_VALUES>}, "y": <RESULT>}'
```

After sufficient iterations, generate the report:

```bash
uv run python -m bo_workflow.cli report --run-id <RUN_ID>
```

---

## Step 7 — Present Results

Always include in your final summary:

1. **Best result found** — the value and which experiment produced it
2. **Best conditions** — the exact feature values to replicate it
3. **Convergence trajectory** — how the best value improved over iterations (from `convergence.pdf`)
4. **Oracle quality** (proxy mode only) — CV RMSE, so the user knows surrogate reliability
5. **Simulation label** (proxy mode only) — remind the user this is a surrogate, not real experimental data

---

## Guardrails

- Never run proxy oracle evaluation in human-in-the-loop mode
- Always include oracle CV RMSE when presenting proxy results
- If a categorical column has >64 unique values, flag this before `init` — the engine will error
- Never auto-commit observations without the user confirming the result value
- Pass `--intent-json` when the upstream agent has captured the original user intent — this preserves provenance in `runs/<RUN_ID>/intent.json`

---

## Quick Reference — Which Workflow?

| Situation | Workflow |
|---|---|
| Have a full labeled dataset, want automated benchmark | Proxy mode (Step 5) |
| Running real lab experiments | Human-in-the-loop (Step 6) |
| Buchwald dataset, maximize yield | Proxy mode, no encoding, no simplex |
| OER dataset, minimize overpotential | Proxy mode + `--simplex-groups` for metal proportions |
| SMILES columns in dataset | Encode first (Step 4), then proxy or human-in-the-loop |
