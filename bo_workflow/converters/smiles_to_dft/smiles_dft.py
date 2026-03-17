"""General-purpose SMILES → DFT descriptor converter.

Takes any CSV that contains SMILES columns (auto-detected or specified),
computes DFT descriptors for each unique molecule, and merges them with
all non-SMILES columns (temperature, concentration, yield, etc.) into
a single output CSV.

Auto-detection
--------------
Columns are classified as SMILES if ≥80 % of their non-null values
parse as valid molecules via RDKit.  Non-SMILES columns are passed
through as-is.

Presets
-------
For known reaction types a preset can supply domain-specific settings
(atom-selection roles, NMR on/off, multi-conformer stats).  Without a
preset every detected SMILES column gets sensible defaults.

Usage
-----
# Auto-detect SMILES columns
python -m bo_workflow.converters.smiles_to_dft encode \
    --input data/buchwald_hartwig_rxns.csv \
    --output-dir data/bh_dft --verbose

# With Buchwald-Hartwig preset (column→prefix mapping & roles)
python -m bo_workflow.converters.smiles_to_dft encode \
    --input data/buchwald_hartwig_rxns.csv \
    --output-dir data/bh_dft --preset buchwald-hartwig --verbose
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def _check_deps() -> None:
    missing = []
    try:
        import rdkit  # noqa: F401
    except ImportError:
        missing.append("rdkit")
    try:
        import pyscf  # noqa: F401
    except ImportError:
        missing.append("pyscf")
    if missing:
        raise ImportError(
            f"DFT converter requires: {', '.join(missing)}.  "
            f"Install with:  uv pip install {' '.join(missing)}"
        )


# ---------------------------------------------------------------------------
# SMILES auto-detection
# ---------------------------------------------------------------------------

def detect_smiles_columns(
    df: pd.DataFrame,
    *,
    threshold: float = 0.80,
    min_unique: int = 1,
    exclude: set[str] | None = None,
) -> list[str]:
    """Return column names whose values are parseable SMILES.

    A column qualifies if at least *threshold* fraction of its non-null
    values parse as valid SMILES via ``Chem.MolFromSmiles``.

    Parameters
    ----------
    df : DataFrame to inspect.
    threshold : minimum fraction of valid SMILES values (0-1).
    min_unique : ignore columns with fewer unique values.
    exclude : column names to never classify as SMILES.
    """
    from rdkit import Chem

    exclude = exclude or set()
    smiles_cols: list[str] = []

    for col in df.columns:
        if col in exclude:
            continue
        # Only consider object/string columns
        if df[col].dtype != object and not pd.api.types.is_string_dtype(df[col]):
            continue
        vals = df[col].dropna().unique()
        if len(vals) < min_unique:
            continue
        n_valid = sum(1 for v in vals if Chem.MolFromSmiles(str(v)) is not None)
        if n_valid / max(len(vals), 1) >= threshold:
            smiles_cols.append(col)

    return smiles_cols


# ---------------------------------------------------------------------------
# Component specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComponentSpec:
    csv_column: str        # column name in input CSV (e.g. "additive")
    prefix: str            # descriptor column prefix (e.g. "solvent")
    role: str              # atom-selection role
    compute_nmr: bool      # per-atom NMR descriptors
    compute_vbur: bool     # %VBur descriptors
    num_conformers: int    # conformers to generate
    multi_conformer: bool  # True → MING/MAXG/STDEV suffixes


BH_PRESET: list[ComponentSpec] = [
    ComponentSpec(
        csv_column="base", prefix="base", role="base",
        compute_nmr=True, compute_vbur=False,
        num_conformers=3, multi_conformer=False,
    ),
    ComponentSpec(
        csv_column="ligand", prefix="ligand", role="ligand",
        compute_nmr=True, compute_vbur=True,
        num_conformers=10, multi_conformer=True,
    ),
    ComponentSpec(
        csv_column="additive", prefix="solvent", role="additive",
        compute_nmr=False, compute_vbur=False,
        num_conformers=3, multi_conformer=False,
    ),
]


def _default_spec_for_column(col: str) -> ComponentSpec:
    """Build a sensible default ComponentSpec for an auto-detected column."""
    return ComponentSpec(
        csv_column=col,
        prefix=col,
        role="ligand",           # generic heteroatom selection
        compute_nmr=True,
        compute_vbur=False,
        num_conformers=3,
        multi_conformer=False,   # scalar descriptors by default
    )


# ---------------------------------------------------------------------------
# Per-conformer descriptor extraction
# ---------------------------------------------------------------------------

def _extract_conformer_descriptors(
    dft_result,
    atom_sel,
    conformer,
    spec: ComponentSpec,
) -> dict[str, float]:
    """Extract a flat dict of raw (non-aggregated) descriptors for one conformer."""
    from ._atom_selection import atom_type_flags, select_charge_ranked_carbons
    from ._vbur import compute_vbur

    d: dict[str, float] = {}

    # Molar volume per conformer (depends on 3D geometry)
    try:
        from rdkit.Chem import AllChem

        mol_copy = conformer.rdkit_mol if hasattr(conformer, "rdkit_mol") else None
        if mol_copy is not None:
            d["molar_volume"] = float(AllChem.ComputeMolVolume(mol_copy))
        else:
            d["molar_volume"] = float("nan")
    except Exception:
        d["molar_volume"] = float("nan")

    # Global electronic properties
    d["homo_energy"] = dft_result.homo_energy
    d["lumo_energy"] = dft_result.lumo_energy
    d["dipole"] = dft_result.dipole_moment
    d["E_scf"] = dft_result.scf_energy

    # Derived
    homo = dft_result.homo_energy
    lumo = dft_result.lumo_energy
    if not (math.isnan(homo) or math.isnan(lumo)):
        d["electronegativity"] = -(homo + lumo) / 2.0
        d["hardness"] = (lumo - homo) / 2.0
    else:
        d["electronegativity"] = float("nan")
        d["hardness"] = float("nan")

    # Placeholders (Tier 3)
    d["electronic_spatial_extent"] = float("nan")
    d["zero_point_correction"] = float("nan")
    d["G_thermal_correction"] = float("nan")

    # Per-atom descriptors at atom1-4
    for k, atom_idx in enumerate(atom_sel.indices):
        label = f"atom{k + 1}"
        # Mulliken charge
        if atom_idx < len(dft_result.mulliken_charges):
            d[f"{label}_Mulliken_charge"] = float(
                dft_result.mulliken_charges[atom_idx]
            )
        else:
            d[f"{label}_Mulliken_charge"] = float("nan")

        # NMR
        if spec.compute_nmr:
            if atom_idx < len(dft_result.nmr_shieldings):
                d[f"{label}_NMR_shift"] = float(
                    dft_result.nmr_shieldings[atom_idx]
                )
                d[f"{label}_NMR_anisotropy"] = float(
                    dft_result.nmr_anisotropies[atom_idx]
                )
            else:
                d[f"{label}_NMR_shift"] = float("nan")
                d[f"{label}_NMR_anisotropy"] = float("nan")

        # %VBur
        if spec.compute_vbur:
            d[f"{label}_%VBur"] = compute_vbur(
                conformer.coords, conformer.atom_symbols, atom_idx
            )

    # Charge-ranked carbons (c_min, c_min+1, c_max, c_max-1)
    c_ranked = select_charge_ranked_carbons(
        dft_result.mulliken_charges, conformer.atom_symbols
    )
    for pos in ("c_min", "c_min+1", "c_max", "c_max-1"):
        c_idx = c_ranked.get(pos)
        if c_idx is not None:
            d[f"{pos}_atom_number"] = float(c_idx)
            flags = atom_type_flags(c_idx, conformer.atom_symbols)
            for flag_key, flag_val in flags.items():
                d[f"{pos}_{flag_key}"] = flag_val
            d[f"{pos}_Mulliken_charge"] = float(
                dft_result.mulliken_charges[c_idx]
            )
            if spec.compute_nmr:
                if c_idx < len(dft_result.nmr_shieldings):
                    d[f"{pos}_NMR_shift"] = float(
                        dft_result.nmr_shieldings[c_idx]
                    )
                    d[f"{pos}_NMR_anisotropy"] = float(
                        dft_result.nmr_anisotropies[c_idx]
                    )
                else:
                    d[f"{pos}_NMR_shift"] = float("nan")
                    d[f"{pos}_NMR_anisotropy"] = float("nan")
            if spec.compute_vbur:
                d[f"{pos}_%VBur"] = compute_vbur(
                    conformer.coords, conformer.atom_symbols, c_idx
                )
        else:
            d[f"{pos}_atom_number"] = float("nan")
            for element in ("C", "N", "O", "P"):
                d[f"{pos}_atom={element}"] = float("nan")
            d[f"{pos}_Mulliken_charge"] = float("nan")
            if spec.compute_nmr:
                d[f"{pos}_NMR_shift"] = float("nan")
                d[f"{pos}_NMR_anisotropy"] = float("nan")
            if spec.compute_vbur:
                d[f"{pos}_%VBur"] = float("nan")

    return d


# ---------------------------------------------------------------------------
# Conformer aggregation
# ---------------------------------------------------------------------------

def _aggregate(
    per_conformer: list[dict[str, float]],
    prefix: str,
    multi_conformer: bool,
) -> dict[str, float]:
    """Aggregate per-conformer descriptors into final prefixed columns.

    For multi_conformer=True: produce {prefix}_{prop}_MING/MAXG/STDEV/MEAN.
    For multi_conformer=False: produce {prefix}_{prop} (single value from
    the lowest-energy conformer).
    """
    if not per_conformer:
        return {}

    result: dict[str, float] = {}

    if not multi_conformer:
        best = per_conformer[0]
        for prop, val in best.items():
            result[f"{prefix}_{prop}"] = val
        return result

    # Multi-conformer: compute stats
    all_keys = set()
    for d in per_conformer:
        all_keys.update(d.keys())

    for prop in sorted(all_keys):
        vals = [d[prop] for d in per_conformer if prop in d]
        valid = [v for v in vals if not math.isnan(v)]
        if not valid:
            result[f"{prefix}_{prop}_MING"] = float("nan")
            result[f"{prefix}_{prop}_MAXG"] = float("nan")
            result[f"{prefix}_{prop}_STDEV"] = float("nan")
            result[f"{prefix}_{prop}_MEAN"] = float("nan")
            continue

        arr = np.array(valid, dtype=np.float64)
        result[f"{prefix}_{prop}_MING"] = float(np.min(arr))
        result[f"{prefix}_{prop}_MAXG"] = float(np.max(arr))
        result[f"{prefix}_{prop}_STDEV"] = float(np.std(arr, ddof=0))
        result[f"{prefix}_{prop}_MEAN"] = float(np.mean(arr))

    return result


# ---------------------------------------------------------------------------
# Single molecule: full descriptor computation
# ---------------------------------------------------------------------------

def compute_molecule_descriptors(
    smiles: str,
    spec: ComponentSpec,
    *,
    basis: str = "6-31g*",
    xc: str = "b3lyp",
    verbose: bool = False,
) -> dict[str, float]:
    """Compute all Tier-1 DFT descriptors for a single molecule.

    Returns a flat dict of ``{prefix}_{property}[_{stat}]`` → float.
    """
    from ._atom_selection import select_atoms
    from ._conformers import (
        ConformerError,
        generate_conformers,
        mol_from_smiles,
        mol_properties,
    )
    from ._dft_engine import run_dft

    prefix = spec.prefix

    # 1. RDKit basic properties
    mol_props = mol_properties(smiles)
    basic = {
        f"{prefix}_number_of_atoms": mol_props["number_of_atoms"],
        f"{prefix}_molar_mass": mol_props["molar_mass"],
        f"{prefix}_molar_volume": mol_props["molar_volume"],
    }

    # 2. Generate conformers
    try:
        conformers = generate_conformers(
            smiles,
            num_conformers=spec.num_conformers,
            random_seed=42,
        )
    except ConformerError as exc:
        if verbose:
            print(f"[dft] Conformer generation failed for {smiles}: {exc}",
                  file=sys.stderr)
        return basic

    # 3. Atom selection (topology-based, same for all conformers)
    mol = mol_from_smiles(smiles)
    from rdkit import Chem
    mol_h = Chem.AddHs(mol)
    atom_sel = select_atoms(mol_h, role=spec.role)

    # 4. DFT on each conformer
    per_conformer: list[dict[str, float]] = []
    for i, conf in enumerate(conformers):
        if verbose:
            print(f"  conformer {i + 1}/{len(conformers)} ...",
                  file=sys.stderr, end="", flush=True)
        dft_res = run_dft(
            conf.atom_symbols, conf.coords,
            basis=basis, xc=xc,
            compute_nmr=spec.compute_nmr,
            verbose=2 if verbose else 0,
        )
        if verbose:
            status = "OK" if dft_res.converged else "FAILED"
            print(f" {status}", file=sys.stderr)

        if not dft_res.converged:
            continue

        desc = _extract_conformer_descriptors(dft_res, atom_sel, conf, spec)
        per_conformer.append(desc)

    if not per_conformer:
        if verbose:
            print(f"[dft] All conformers failed for {smiles}", file=sys.stderr)
        return basic

    # 5. Aggregate across conformers
    aggregated = _aggregate(per_conformer, prefix, spec.multi_conformer)

    # Merge basic + aggregated
    if spec.multi_conformer:
        # Constant properties get _MING suffix for consistency
        for key in ("number_of_atoms", "molar_mass"):
            full_key = f"{prefix}_{key}"
            if full_key in basic:
                aggregated[f"{full_key}_MING"] = basic[full_key]
        # molar_volume varies per conformer → already in aggregated
    else:
        aggregated[f"{prefix}_number_of_atoms"] = basic[f"{prefix}_number_of_atoms"]
        aggregated[f"{prefix}_molar_mass"] = basic[f"{prefix}_molar_mass"]
        # molar_volume comes from per-conformer extraction

    return aggregated


# ---------------------------------------------------------------------------
# Dataset-level encoding
# ---------------------------------------------------------------------------

def encode_dataset_dft(
    input_path: Path,
    output_dir: Path,
    *,
    target_col: str | None = None,
    components: list[ComponentSpec] | None = None,
    basis: str = "6-31g*",
    xc: str = "b3lyp",
    verbose: bool = False,
) -> pd.DataFrame:
    """Transform any CSV with SMILES columns into a DFT feature CSV.

    1. Auto-detect (or use preset) SMILES columns.
    2. Compute DFT descriptors for each unique SMILES (with caching).
    3. Merge descriptors + all non-SMILES columns into one DataFrame.

    Parameters
    ----------
    input_path : path to input CSV.
    output_dir : directory for cache and output files.
    target_col : optional target column name (kept in output as-is).
    components : explicit component specs; if None, auto-detect.
    basis : DFT basis set.
    xc : DFT exchange-correlation functional.
    verbose : print progress to stderr.

    Returns
    -------
    DataFrame with SMILES columns replaced by DFT descriptors, plus
    all non-SMILES columns preserved.
    """
    from ._descriptor_cache import DescriptorCache

    _check_deps()

    df = pd.read_csv(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Determine SMILES vs passthrough columns --------------------------
    if components is not None:
        smiles_cols = [s.csv_column for s in components if s.csv_column in df.columns]
    else:
        smiles_cols = detect_smiles_columns(df)
        components = [_default_spec_for_column(c) for c in smiles_cols]
        if verbose:
            print(f"[auto] Detected SMILES columns: {smiles_cols}", file=sys.stderr)

    passthrough_cols = [c for c in df.columns if c not in smiles_cols]
    if verbose:
        print(f"[auto] Passthrough columns: {passthrough_cols}", file=sys.stderr)

    # --- Compute descriptors per unique SMILES ----------------------------
    cache = DescriptorCache(output_dir)
    stats = {"cache_hits": 0, "computed": 0, "failed": []}

    # {csv_column: {smiles_str: {col: val}}}
    component_descriptors: dict[str, dict[str, dict[str, float]]] = {}

    for spec in components:
        col = spec.csv_column
        if col not in df.columns:
            if verbose:
                print(f"[warn] Column '{col}' not in CSV, skipping",
                      file=sys.stderr)
            continue

        unique_smiles = df[col].dropna().unique()
        if verbose:
            print(f"\n=== {col} → {spec.prefix}_ "
                  f"({len(unique_smiles)} unique molecules) ===",
                  file=sys.stderr)

        comp_cache: dict[str, dict[str, float]] = {}
        for idx, smi in enumerate(unique_smiles):
            smi_str = str(smi)
            # Cache key includes prefix so same SMILES in different roles
            # gets separate entries.
            cache_key = f"{spec.prefix}::{smi_str}"

            cached = cache.get(cache_key)
            if cached is not None:
                comp_cache[smi_str] = cached
                stats["cache_hits"] += 1
                if verbose:
                    print(f"  [{idx + 1}/{len(unique_smiles)}] "
                          f"{smi_str[:50]} (cached)", file=sys.stderr)
                continue

            if verbose:
                print(f"  [{idx + 1}/{len(unique_smiles)}] "
                      f"{smi_str[:50]} computing", file=sys.stderr)

            t0 = time.time()
            try:
                desc = compute_molecule_descriptors(
                    smi_str, spec, basis=basis, xc=xc, verbose=verbose,
                )
                comp_cache[smi_str] = desc
                cache.put(cache_key, desc)
                stats["computed"] += 1
                if verbose:
                    elapsed = time.time() - t0
                    print(f"    → {len(desc)} descriptors in {elapsed:.1f}s",
                          file=sys.stderr)
            except Exception as exc:
                if verbose:
                    print(f"    → FAILED: {exc}", file=sys.stderr)
                stats["failed"].append(smi_str)
                comp_cache[smi_str] = {}

        component_descriptors[col] = comp_cache

    # --- Assemble output DataFrame ----------------------------------------
    rows: list[dict] = []
    for row_idx, rxn_row in df.iterrows():
        row: dict = {}

        # Passthrough columns (temperature, yield, concentration, etc.)
        for pc in passthrough_cols:
            row[pc] = rxn_row[pc]

        # DFT descriptor columns for each SMILES component
        for spec in components:
            col = spec.csv_column
            if col not in component_descriptors:
                continue
            smi = str(rxn_row.get(col, ""))
            desc = component_descriptors[col].get(smi, {})
            row.update(desc)

        rows.append(row)

    result_df = pd.DataFrame(rows)

    # Summary
    if verbose:
        n_desc = len(result_df.columns) - len(passthrough_cols)
        print(f"\n=== Done: {len(result_df)} rows, "
              f"{len(passthrough_cols)} passthrough + {n_desc} DFT cols ===",
              file=sys.stderr)
        print(f"    computed: {stats['computed']}, "
              f"cached: {stats['cache_hits']}, "
              f"failed: {len(stats['failed'])}", file=sys.stderr)

    return result_df


# Keep backward-compatible alias
def encode_reactions_dft(
    input_path: Path,
    output_dir: Path,
    *,
    yield_col: str = "yield",
    components: list[ComponentSpec] | None = None,
    basis: str = "6-31g*",
    xc: str = "b3lyp",
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Legacy wrapper — calls encode_dataset_dft and returns (features, catalog)."""
    input_df = pd.read_csv(input_path)
    result = encode_dataset_dft(
        input_path, output_dir,
        target_col=yield_col,
        components=components,
        basis=basis, xc=xc,
        verbose=verbose,
    )
    # Split into features (no SMILES) and catalog (with SMILES)
    if components is None:
        smiles_cols = detect_smiles_columns(input_df)
    else:
        smiles_cols = [s.csv_column for s in components if s.csv_column in input_df.columns]
    features_df = result.drop(columns=[c for c in smiles_cols if c in result.columns],
                              errors="ignore")

    catalog_df = input_df.copy()
    for col in result.columns:
        if col not in catalog_df.columns:
            catalog_df[col] = result[col]

    return features_df, catalog_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_encode(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    components = None
    if args.preset == "buchwald-hartwig":
        components = BH_PRESET

    result_df = encode_dataset_dft(
        Path(args.input),
        out_dir,
        target_col=args.target_col,
        components=components,
        basis=args.basis,
        xc=args.xc,
        verbose=args.verbose,
    )

    features_path = out_dir / "features.csv"
    result_df.to_csv(features_path, index=False)

    # Identify descriptor columns (everything except passthrough)
    input_df = pd.read_csv(args.input, nrows=0)
    if components is not None:
        smiles_cols = [s.csv_column for s in components if s.csv_column in input_df.columns]
    else:
        smiles_cols = detect_smiles_columns(pd.read_csv(args.input))
    passthrough = [c for c in input_df.columns if c not in smiles_cols]
    descriptor_cols = [c for c in result_df.columns if c not in passthrough]

    print(json.dumps({
        "status": "ok",
        "input": str(args.input),
        "features_csv": str(features_path),
        "rows": len(result_df),
        "passthrough_columns": len(passthrough),
        "descriptor_columns": len(descriptor_cols),
        "total_columns": len(result_df.columns),
        "smiles_columns_detected": smiles_cols,
        "passthrough_columns_kept": passthrough,
    }, indent=2))


def _cmd_status(args: argparse.Namespace) -> None:
    from ._descriptor_cache import DescriptorCache

    cache = DescriptorCache(Path(args.cache_dir))
    print(json.dumps({
        "cache_dir": str(args.cache_dir),
        "cached_molecules": cache.size,
    }, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SMILES → DFT descriptor converter",
    )
    sub = parser.add_subparsers(dest="command")

    enc = sub.add_parser("encode", help="SMILES → DFT descriptor CSV")
    enc.add_argument("--input", required=True,
                     help="Input CSV (any columns; SMILES auto-detected)")
    enc.add_argument("--output-dir", required=True, help="Output directory")
    enc.add_argument("--target-col", default=None,
                     help="Target/objective column name (passthrough)")
    enc.add_argument("--preset", default=None, choices=["buchwald-hartwig"],
                     help="Predefined component mapping (optional)")
    enc.add_argument("--basis", default="6-31g*", help="DFT basis set")
    enc.add_argument("--xc", default="b3lyp", help="DFT functional")
    enc.add_argument("--verbose", action="store_true")

    st = sub.add_parser("status", help="Show cache status")
    st.add_argument("--cache-dir", required=True)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "encode":
        _cmd_encode(args)
    elif args.command == "status":
        _cmd_status(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
