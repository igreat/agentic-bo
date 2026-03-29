---
name: bo-encode-dft
description: Encode SMILES columns into DFT quantum-chemical descriptors (PySCF) for BO.
---

# BO Encode DFT Descriptors

Use this skill when the user has a CSV with SMILES columns and wants quantum-chemical (DFT) descriptors — orbital energies, Mulliken charges, NMR shieldings, dipole moments, buried volume — rather than classical RDKit fingerprints.

This converter is heavier than `bo-encode-molecule-descriptors` (minutes per molecule vs milliseconds) but produces physically grounded features that can capture electronic and steric effects invisible to topology-based descriptors.

## Prerequisites

```bash
uv pip install pyscf rdkit
uv pip install pyscf-properties   # optional: enables NMR shielding
uv pip install morfeus-ml         # optional: enables %VBur
```

## Command

```bash
uv run python -m bo_workflow.converters.smiles_to_dft encode \
  --input <CSV_PATH> --output-dir <DIR> --verbose
```

Optional flags:
- `--preset buchwald-hartwig` — domain-specific column→role mapping (base/ligand/additive)
- `--basis <BASIS>` (default `6-31g*`) — Gaussian basis set
- `--xc <FUNCTIONAL>` (default `b3lyp`) — DFT functional
- `--target-col <COL>` — name of the target/objective column (passthrough)

Without `--preset`, SMILES columns are auto-detected (columns where ≥80% of values parse as valid SMILES via RDKit).

## Return

JSON with:
- `features_csv` — path to output CSV (DFT descriptors + all passthrough columns)
- `rows` — number of rows
- `smiles_columns_detected` — which columns were treated as SMILES
- `passthrough_columns_kept` — non-SMILES columns passed through unchanged
- `descriptor_columns` — number of DFT descriptor columns generated

## Output schema

Each SMILES column produces prefixed descriptor columns:

| Descriptor | Source | Notes |
|-----------|--------|-------|
| `{prefix}_homo_energy` | PySCF RKS | Hartree |
| `{prefix}_lumo_energy` | PySCF RKS | Hartree |
| `{prefix}_dipole` | PySCF RKS | Debye |
| `{prefix}_E_scf` | PySCF RKS | total SCF energy |
| `{prefix}_electronegativity` | derived | -(HOMO+LUMO)/2 |
| `{prefix}_hardness` | derived | (LUMO-HOMO)/2 |
| `{prefix}_molar_volume` | RDKit | per-conformer |
| `{prefix}_molar_mass` | RDKit | |
| `{prefix}_number_of_atoms` | RDKit | heavy atoms |
| `{prefix}_atom{1-4}_Mulliken_charge` | PySCF | at selected atoms |
| `{prefix}_atom{1-4}_NMR_shift` | PySCF GIAO | if pyscf-properties installed |
| `{prefix}_atom{1-4}_%VBur` | Morfeus | if morfeus-ml installed |
| `{prefix}_c_{min,max,...}_Mulliken_charge` | PySCF | charge-ranked carbons |

For multi-conformer components (e.g. ligands in BH preset), each property gets `_MING`, `_MAXG`, `_STDEV`, `_MEAN` suffixes.

## Caching

Computed descriptors are cached in `<output-dir>/dft_cache.json`. Interrupted runs resume automatically.

## Notes

- `features.csv` is ready for `init --dataset`.
- Non-SMILES columns (yield, temperature, concentration) pass through unchanged.
- Large molecules (>30 heavy atoms) can take minutes per conformer with 6-31G*. Use `--basis sto-3g` for faster prototyping.
- NMR and %VBur are optional: if their packages are not installed, those columns are NaN (the rest of the pipeline still works).
- The DFT converter does NOT have a `decode` subcommand. For mapping BO suggestions back to real molecules, use `bo-decode-molecule-descriptors` with the features CSV as catalog.
