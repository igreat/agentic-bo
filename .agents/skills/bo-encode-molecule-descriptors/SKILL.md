---
name: bo-encode-molecule-descriptors
description: Encode molecule SMILES columns into RDKit descriptor features for BO.
---

# BO Encode Molecule Descriptors

Use this skill when the user has a CSV with one or more molecule SMILES columns and wants to convert it into numerical descriptor features for the BO engine.

## Command

```bash
uv run python -m bo_workflow.converters.molecule_descriptors encode \
  --input <CSV_PATH> --output-dir <DIR> --smiles-cols <COL1> [<COL2> ...]
```

Optional flags: `--morgan-bits <N>` (default 64).

## Return

- `features_csv` — path to descriptor features CSV (descriptor/fingerprint columns + passthrough columns)
- `catalog_csv` — path to catalog CSV (same descriptors + original SMILES columns for decode)
- `rows` — number of rows encoded
- `descriptor_columns` — number of descriptor columns generated
- `passthrough_columns` — non-SMILES columns carried through from the input

## Notes

- `features.csv` is ready for `init --dataset`.
- `catalog.csv` is required by `decode` to map BO suggestions back to real molecules.
- Supports multiple SMILES columns; each gets prefixed descriptor columns.
