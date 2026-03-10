---
name: bo-agentic-workflow
description: Full end-to-end BO orchestration for a new chemistry problem from scratch. Use when a chemist describes their problem in plain language and needs the complete workflow handled automatically.
---

# BO Agentic Workflow

Use this skill when a user describes a chemistry optimization problem from scratch — dataset, target, and objective — and wants Claude to handle the full workflow without manual setup.

## Step 1 — Understand the Problem

Ask (or infer from context) the following before doing anything:

| Question | Why it matters |
|---|---|
| What is the dataset path (CSV)? | Required for init |
| Which column is the target? | Required for init |
| Maximize or minimize? | Required for init |
| Proxy mode or human-in-the-loop? | Determines workflow branch |
| Are there compositional constraints? (proportions summing to 100%) | Determines if simplex needed |
| Are there SMILES columns? | Determines if encoder needed |

If the user has already provided all of this, skip straight to Step 2.

## Step 2 — Validate the Dataset

Before running anything, inspect the dataset:

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

Flag to the user if any of these are found:
- Missing values in the target column → must be dropped before init
- Constant columns (zero variance) → will be dropped automatically by engine
- High-cardinality categoricals (>64 unique values) → engine will error, flag this
- Proportions that appear to sum to ~100% → simplex converter needed

## Step 3 — Convert if Needed

### No conversion needed (Buchwald-style categorical datasets, HER-style numeric datasets)
Go straight to Step 4. This is the case when:
- All feature columns are already categorical (which reagent) or numeric (concentration, temperature)
- No SMILES columns
- No compositional proportions summing to 100%

### Simplex conversion needed (OER, HEA — proportions summing to 100%)
Compositional datasets where Metal_1_Proportion + Metal_2_Proportion + ... = 100% require
the simplex converter before init. Use `bo-transform-columns` skill.
Note: simplex converter must be implemented before this branch works.

### SMILES molecule columns
Use `bo-encode-molecule-descriptors` skill to encode SMILES → RDKit descriptors.
After BO runs, use `bo-decode-molecule-descriptors` to translate suggestions back to molecule names.

### Reaction SMILES columns
Use `bo-encode-drfp` skill to encode reaction SMILES → DRFP fingerprints.
After BO runs, use `bo-decode-drfp` to translate suggestions back to reactions.

## Step 4 — Proxy Mode Workflow (no real chemist)

Use this when the user wants a fully automated run against a surrogate oracle.
The oracle is trained on the existing dataset and evaluates BO suggestions automatically.

```bash
# 1. Initialize run
uv run python -m bo_workflow.cli init \
  --dataset <CSV_PATH> \
  --target <TARGET_COL> \
  --objective <min|max> \
  --seed 42

# Extract run_id from JSON output

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
- Best experiment conditions (feature values)
- Oracle CV RMSE (surrogate quality indicator)
- **Always label results as proxy-oracle simulation, not real experimental results**

## Step 5 — Human-in-the-Loop Workflow (real chemist)

Use this when the user will run real experiments and report results manually.
Do NOT use the proxy oracle in this mode.

```bash
# 1. Initialize run
uv run python -m bo_workflow.cli init \
  --dataset <CSV_PATH> \
  --target <TARGET_COL> \
  --objective <min|max> \
  --seed 42
```

Then repeat this loop until the user is satisfied:

```bash
# Get next suggestion
uv run python -m bo_workflow.cli suggest --run-id <RUN_ID> --batch-size <N>
```

Present the suggestion clearly to the chemist:
- Show feature values in plain language (e.g., "Try ligand=BrettPhos, base=MTBD, additive=X")
- If SMILES were decoded, show reagent names not SMILES strings
- Explain what the BO engine expects from them next

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

## Step 6 — Present Results

Always include in your final summary:
1. **Best result found** — the value and which experiment produced it
2. **Best conditions** — the exact feature values to replicate it
3. **Convergence** — how quickly BO found the optimum vs random search
4. **Oracle quality** (proxy mode only) — CV RMSE, so user knows surrogate reliability
5. **Simulation label** (proxy mode only) — remind the user this is a surrogate, not real data

## Guardrails

- Never run proxy oracle evaluation in human-in-the-loop mode
- Always include oracle CV RMSE when presenting proxy results
- If proportions appear to sum to 100% but simplex is not yet implemented, warn the user that suggestions may be invalid
- If a categorical column has >64 unique values, warn before init — the engine will error
- Never auto-commit observations without the user confirming the result value

## Quick Reference — Which Workflow?

| Situation | Workflow |
|---|---|
| Have a dataset, want automated benchmark | Proxy mode (Step 4) |
| Running real lab experiments | Human-in-the-loop (Step 5) |
| Buchwald dataset, maximize yield | Proxy mode, no conversion |
| OER dataset, minimize overpotential | Proxy mode, simplex needed |
| SMILES columns in dataset | Encode first (Step 3), then proxy or human-in-the-loop |
