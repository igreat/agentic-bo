---
name: bo-encode-drfp
description: Encode reaction SMILES into DRFP fingerprint features for BO.
---

# BO Encode DRFP

Use this skill when the user has a CSV with reaction SMILES and wants to convert it into numerical fingerprint features suitable for the BO engine.

## Command

```bash
uv run python -m bo_workflow.converters.reaction_drfp encode \
  --input <CSV_PATH> --output-dir <DIR>
```

Optional flags: `--rxn-col <COL>` (default `rxn_smiles`), `--n-bits <N>` (default 64).

## Return

- `features_csv` — path to the encoded features CSV (fingerprint bit columns + passthrough columns)
- `catalog_csv` — path to the catalog CSV (maps fingerprint rows back to original reaction SMILES)
- `reactions` — number of reactions encoded
- `fingerprint_bits` — number of bit columns generated
- `passthrough_columns` — non-SMILES columns carried through from the input

## Notes

- The output `features.csv` is ready to pass directly to `init --dataset`.
- The `catalog.csv` is needed later by `decode` to map BO suggestions back to real reactions.
- Non-SMILES columns (e.g. yield, temperature) are passed through unchanged into `features.csv`.
