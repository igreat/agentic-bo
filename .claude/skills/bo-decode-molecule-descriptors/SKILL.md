---
name: bo-decode-molecule-descriptors
description: Decode BO descriptor suggestions back to nearest real molecules.
---

# BO Decode Molecule Descriptors

Use this skill when the BO engine has suggested descriptor vectors and the user wants the closest real molecules from a descriptor catalog.

## Command

```bash
uv run python -m bo_workflow.converters.molecule_descriptors decode \
  --catalog <CATALOG_CSV> --query '<JSON>'
```

Optional flags: `--k <N>` (number of nearest neighbors, default 3).

## Return

- JSON list of nearest-neighbor molecules, each with `rank`, `distance`, and all non-descriptor metadata columns from the catalog (including original SMILES columns).

## Notes

- `--query` accepts a JSON dict of descriptor values (inline JSON), matching engine suggestion `x` keys.
- `--catalog` is typically the `catalog.csv` from encode, but can be any external CSV with matching descriptor columns.
- Results are sorted by ascending Euclidean distance after min-max normalization on catalog descriptor columns.
