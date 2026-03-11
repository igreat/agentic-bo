---
name: bo-transform-columns
description: Profile dataset columns and apply data transformations (log, sqrt, standardize, etc.) before BO.
---

# BO Transform Columns

Use this skill when preparing a dataset for BO and the target or feature columns may benefit from a scale transformation. Always **profile first**, then **transform**.

---

## Step 1 — Profile

```bash
uv run python -m bo_workflow.converters.column_transform profile \
  --input <CSV_PATH> \
  --cols <COL1> [<COL2> ...]
```

Omit `--cols` to profile all columns. Output is JSON with stats and a `recommended_transform` per column.

**Key fields to interpret:**
- `log_range_decades` ≥ 3 → strong case for `log10` (data spans orders of magnitude)
- `skewness` > 2 → strong right skew, try `log10` or `sqrt`
- `fraction_zero` > 0 with skew → use `log1p` instead of `log10`
- `fraction_negative` > 0 → skip log; try `standardize` if skewed

---

## Step 2 — Transform

```bash
uv run python -m bo_workflow.converters.column_transform transform \
  --input <CSV_PATH> \
  --cols <COL> \
  --transform <TRANSFORM> \
  --output <OUTPUT_CSV>
```

The original column is dropped and replaced with a prefixed version:
`ic50_nM` → `log10_ic50_nM`. Update `--target` accordingly when calling `init`.

Add `--keep-original` to retain both columns.

---

## When to use each transform

| Transform | When to use | Requires | Output name |
|-----------|-------------|----------|-------------|
| `log10` | Strictly positive, ≥3 decades of range (IC50, concentrations, rates) | all values > 0 | `log10_<col>` |
| `log1p` | Non-negative with zeros, right-skewed counts or areas | all values ≥ 0 | `log1p_<col>` |
| `neglog10` | Strictly positive, want higher = better (IC50 → pIC50, Ki → pKi) | all values > 0 | `neg_log10_<col>` |
| `sqrt` | Non-negative, moderate right skew (skewness 1–2) | all values ≥ 0 | `sqrt_<col>` |
| `standardize` | Mixed-sign or roughly normal; centres and unit-scales | non-constant | `std_<col>` |
| `minmax` | Bounded range needed, e.g. mixing features with different units | non-constant | `mm_<col>` |

### Chemistry-specific guidance

- **IC50, Ki, Kd, EC50** in any unit: use `log10` before BO. These span many orders of magnitude and tree/GP models perform poorly on raw scale.
- **pIC50, pKi, pKd** (already negative-log scale): no transform needed.
- **Yield (0–100%)**: typically low skew, no transform needed unless heavily bimodal.
- **Reaction rates, turnover numbers**: often log-normal; use `log10`.
- **ADME properties (logP, TPSA)**: already on reasonable scales; profile first.
- **Morgan fingerprint bits (0/1)**: never transform — they are binary.
- **Counts (e.g. ring count)**: rarely need transforms; check skewness.

### When NOT to transform

- Binary or near-binary columns (Morgan bits, flags).
- Already log-scaled columns (logP, logD).
- When the objective is `max` with a linear-scale reward you want to compare directly.
- Features that are already bounded [0, 1] (FractionCSP3, AromaticProportion).

---

## Return (transform)

- `output` — path to the transformed CSV
- `transformed_columns` — list of new column names (use these for `--target` or as feature references)
- `rows` — row count (sanity check)

---

## Example: EGFR IC50 workflow

```bash
# 1. Profile the target
uv run python -m bo_workflow.converters.column_transform profile \
  --input data/egfr_ic50.csv --cols ic50_nM

# 2. Apply log10 (profile will recommend it: spans ~16 decades)
uv run python -m bo_workflow.converters.column_transform transform \
  --input data/egfr_ic50.csv \
  --cols ic50_nM --transform log10 \
  --output data/egfr_log.csv

# 3. Encode SMILES
uv run python -m bo_workflow.converters.molecule_descriptors encode \
  --input data/egfr_log.csv \
  --output-dir data/egfr_log_desc \
  --smiles-cols smiles

# 4. Init BO — note updated --target name
uv run python -m bo_workflow.cli init \
  --dataset data/egfr_log_desc/features.csv \
  --target log10_ic50_nM --objective min --engine hebo
```
