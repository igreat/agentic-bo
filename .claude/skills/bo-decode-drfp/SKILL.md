---
name: bo-decode-drfp
description: Decode BO suggestions back to nearest real reactions via DRFP similarity.
---

# BO Decode DRFP

Use this skill when the BO engine has suggested fingerprint vectors and the user wants to find the closest real reactions from the original catalog.

## Command

```bash
uv run python -m bo_workflow.converters.reaction_drfp decode \
  --catalog <CATALOG_CSV> --query '<JSON>'
```

Optional flags: `--k <N>` (number of nearest neighbors, default 3).

## Return

- JSON list of nearest-neighbor reactions, each with `rank`, `similarity`, and all non-fingerprint columns from the catalog (including the original reaction SMILES).

## Notes

- `--query` accepts a JSON dict of fingerprint bit values, matching the format produced by `suggest`.
- `--catalog` should point to the `catalog.csv` produced by the `encode` step.
- Results are sorted by descending Tanimoto similarity.
