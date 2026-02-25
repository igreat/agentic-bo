"""SMILES column -> RDKit molecular descriptor converter.

Column-focused design matching ``reaction_drfp.py``: encodes one or more
SMILES columns into molecular descriptor columns and passes all other
columns through unchanged.

Features per SMILES column:
  - 11 basic descriptors (MolWt, LogP, TPSA, etc.)
  - 4 Gasteiger charge stats (mean, std, min, max)
  - Morgan fingerprint bits (configurable, default 64)

Encode: takes a CSV, replaces each SMILES column with prefixed descriptor
columns, and writes two files:
  - ``features.csv`` — descriptor columns + all non-SMILES columns (feed to engine)
  - ``catalog.csv`` — same as features.csv but also keeps the original
    SMILES columns (used for decode lookup)

Decode: given descriptor values (e.g. from a HEBO suggestion), finds the
nearest molecules in the catalog by Euclidean distance on normalised descriptors.

Usage
-----
# Encode: SMILES columns -> descriptor feature CSV + lookup catalog
python -m bo_workflow.converters.molecule_descriptors encode \
    --input data/buchwald_hartwig_rxns.csv \
    --output-dir data/bh_desc \
    --smiles-cols aryl_halide ligand base additive

# Decode: descriptors -> nearest molecules (k-NN)
python -m bo_workflow.converters.molecule_descriptors decode \
    --catalog data/bh_desc/catalog.csv \
    --query '{"ah_MolWt": 200, "ah_LogP": 1.5}' \
    --k 3
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from collections.abc import Callable

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

# ---------------------------------------------------------------------------
# Descriptor calculation
# ---------------------------------------------------------------------------

# RDKit populates Descriptors.* dynamically, so Pylance can't resolve them.
# Use getattr to avoid false "not a known attribute" errors.
_DESCRIPTOR_FNS: dict[str, Callable] = {
    "MolWt": getattr(Descriptors, "MolWt"),
    "LogP": getattr(Descriptors, "MolLogP"),
    "TPSA": getattr(Descriptors, "TPSA"),
    "NumHDonors": getattr(Descriptors, "NumHDonors"),
    "NumHAcceptors": getattr(Descriptors, "NumHAcceptors"),
    "NumRotatableBonds": getattr(Descriptors, "NumRotatableBonds"),
    "RingCount": getattr(Descriptors, "RingCount"),
    "NumAromaticRings": getattr(rdMolDescriptors, "CalcNumAromaticRings"),
    "FractionCSP3": getattr(Descriptors, "FractionCSP3"),
    "HeavyAtomCount": getattr(Descriptors, "HeavyAtomCount"),
}

# Names of the Gasteiger charge features (for column detection in decode)
_GASTEIGER_NAMES = ("GasteigerMean", "GasteigerStd", "GasteigerMin", "GasteigerMax")

# Prefix for Morgan FP bit columns
_MORGAN_PREFIX = "mfp"


def _aromatic_proportion(mol: Chem.Mol) -> float:
    """Fraction of heavy atoms that are aromatic."""
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy == 0:
        return 0.0
    n_aromatic = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    return n_aromatic / n_heavy


def compute_gasteiger_charges(mol: Chem.Mol) -> dict[str, float]:
    """Compute Gasteiger partial charge statistics for a molecule.

    Returns mean, std, min, max of partial charges across all atoms.
    Filters out NaN/inf values before computing stats.
    """
    nan_result = {name: float("nan") for name in _GASTEIGER_NAMES}

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        return nan_result

    charges = []
    for i in range(mol.GetNumAtoms()):
        charge = mol.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge")
        if math.isfinite(charge):
            charges.append(charge)

    if not charges:
        return nan_result

    arr = np.array(charges, dtype=np.float64)
    return {
        "GasteigerMean": float(arr.mean()),
        "GasteigerStd": float(arr.std()),
        "GasteigerMin": float(arr.min()),
        "GasteigerMax": float(arr.max()),
    }


def compute_morgan_fp(mol: Chem.Mol, n_bits: int = 64) -> dict[str, int]:
    """Compute Morgan (ECFP) fingerprint as individual bit columns.

    Returns a dict mapping ``mfp_0`` .. ``mfp_{n_bits-1}`` to 0/1 values.
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return {f"{_MORGAN_PREFIX}_{i}": int(fp[i]) for i in range(n_bits)}


def compute_descriptors(smiles: str, morgan_bits: int = 64) -> dict[str, float]:
    """Compute all molecular descriptors for a single SMILES.

    Returns a dict with:
      - 11 basic descriptors (including AromaticProportion)
      - 4 Gasteiger charge stats
      - ``morgan_bits`` Morgan FP bits

    If the SMILES cannot be parsed, all values are NaN (descriptors/charges)
    or 0 (Morgan FP bits).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        result: dict[str, float] = {name: float("nan") for name in _DESCRIPTOR_FNS}
        result["AromaticProportion"] = float("nan")
        result.update({name: float("nan") for name in _GASTEIGER_NAMES})
        result.update({f"{_MORGAN_PREFIX}_{i}": 0 for i in range(morgan_bits)})
        return result

    # Basic descriptors
    result = {name: float(fn(mol)) for name, fn in _DESCRIPTOR_FNS.items()}
    result["AromaticProportion"] = _aromatic_proportion(mol)

    # Gasteiger charges
    result.update(compute_gasteiger_charges(mol))

    # Morgan fingerprint
    result.update(compute_morgan_fp(mol, n_bits=morgan_bits))

    return result


def _abbreviate(col_name: str, max_len: int = 4) -> str:
    """Create a short prefix from a column name.

    'aryl_halide' -> 'ah', 'ligand' -> 'lig', 'base' -> 'base'
    """
    parts = col_name.split("_")
    if len(parts) > 1:
        return "".join(p[0] for p in parts)
    return col_name[:max_len]


# ---------------------------------------------------------------------------
# Column detection helpers (used by decode and combined.py)
# ---------------------------------------------------------------------------

# All known scalar descriptor names (basic + Gasteiger + AromaticProportion)
ALL_SCALAR_DESCRIPTOR_NAMES: set[str] = (
    set(_DESCRIPTOR_FNS) | set(_GASTEIGER_NAMES) | {"AromaticProportion"}
)


def is_descriptor_col(col: str) -> bool:
    """Check if a column name is a descriptor or Morgan FP column."""
    # Scalar descriptors: prefix_DescriptorName
    if any(col.endswith(f"_{d}") for d in ALL_SCALAR_DESCRIPTOR_NAMES):
        return True
    # Morgan FP: prefix_mfp_N
    if f"_{_MORGAN_PREFIX}_" in col:
        return True
    return False


# ---------------------------------------------------------------------------
# Encode: SMILES columns -> descriptor columns
# ---------------------------------------------------------------------------

def encode_molecules(
    input_path: Path,
    smiles_cols: list[str],
    morgan_bits: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode SMILES columns into molecular descriptor columns.

    For each SMILES column, computes descriptors and prefixes with an
    abbreviation of the column name::

        aryl_halide -> ah_MolWt, ah_LogP, ..., ah_GasteigerMean, ..., ah_mfp_0, ...
        ligand      -> lig_MolWt, lig_LogP, ..., lig_GasteigerMean, ..., lig_mfp_0, ...

    Returns
    -------
    (features_df, catalog_df)
        features_df: descriptor columns + all non-SMILES columns
        catalog_df:  descriptor columns + all original columns (including SMILES)
    """
    df = pd.read_csv(input_path)
    missing = [c for c in smiles_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"SMILES columns not found: {missing}. Available: {list(df.columns)}"
        )

    descriptor_frames: list[pd.DataFrame] = []

    for col in smiles_cols:
        prefix = _abbreviate(col)
        desc_rows = [compute_descriptors(s, morgan_bits=morgan_bits) for s in df[col]]
        desc_df = pd.DataFrame(desc_rows)
        desc_df.columns = [f"{prefix}_{name}" for name in desc_df.columns]
        descriptor_frames.append(desc_df)

    # Combine: descriptor columns + passthrough (non-SMILES) columns
    passthrough_cols = [c for c in df.columns if c not in smiles_cols]
    features_df = pd.concat(descriptor_frames, axis=1)
    for col in passthrough_cols:
        features_df[col] = df[col].values

    # Catalog: same as features + original SMILES columns
    catalog_df = features_df.copy()
    for col in smiles_cols:
        catalog_df[col] = df[col].values

    return features_df, catalog_df


# ---------------------------------------------------------------------------
# Decode: descriptors -> nearest molecules via Euclidean distance
# ---------------------------------------------------------------------------

def decode_nearest(
    query_descriptors: np.ndarray,
    catalog: pd.DataFrame,
    descriptor_cols: list[str],
    k: int = 3,
) -> list[dict]:
    """k-NN decode using Euclidean distance on normalised descriptors.

    Parameters
    ----------
    query_descriptors : 1-D array matching ``descriptor_cols`` order
    catalog : DataFrame with descriptor columns + metadata
    descriptor_cols : ordered list of descriptor column names
    k : number of nearest neighbours

    Returns
    -------
    List of dicts with rank, distance, and all non-descriptor columns.
    """
    cat_vals = catalog[descriptor_cols].values.astype(np.float64)

    # Min-max normalise using catalog range
    col_min = cat_vals.min(axis=0)
    col_max = cat_vals.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0  # avoid division by zero

    normed_cat = (cat_vals - col_min) / col_range
    normed_query = (query_descriptors.astype(np.float64) - col_min) / col_range

    distances = np.linalg.norm(normed_cat - normed_query, axis=1)
    top_k_idx = np.argsort(distances)[:k]

    meta_cols = [c for c in catalog.columns if c not in descriptor_cols]
    results = []
    for rank, idx in enumerate(top_k_idx):
        entry = {
            "rank": rank + 1,
            "distance": round(float(distances[idx]), 4),
        }
        for col in meta_cols:
            entry[col] = catalog.iloc[idx][col]
        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_encode(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_df, catalog_df = encode_molecules(
        Path(args.input),
        smiles_cols=args.smiles_cols,
        morgan_bits=args.morgan_bits,
    )

    features_path = out_dir / "features.csv"
    catalog_path = out_dir / "catalog.csv"

    features_df.to_csv(features_path, index=False)
    catalog_df.to_csv(catalog_path, index=False)

    desc_cols = [c for c in features_df.columns if is_descriptor_col(c)]
    passthrough = [c for c in features_df.columns if c not in desc_cols]
    print(json.dumps({
        "status": "ok",
        "input": str(args.input),
        "features_csv": str(features_path),
        "catalog_csv": str(catalog_path),
        "rows": len(features_df),
        "descriptor_columns": len(desc_cols),
        "passthrough_columns": passthrough,
        "smiles_cols": args.smiles_cols,
    }, indent=2))


def _cmd_decode(args: argparse.Namespace) -> None:
    catalog = pd.read_csv(args.catalog)
    query = json.loads(args.query)

    descriptor_cols = sorted([c for c in catalog.columns if is_descriptor_col(c)])

    query_vec = np.array([float(query.get(c, 0)) for c in descriptor_cols])
    results = decode_nearest(query_vec, catalog, descriptor_cols, k=args.k)
    print(json.dumps(results, indent=2, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SMILES columns <-> RDKit molecular descriptor converter",
    )
    sub = parser.add_subparsers(dest="command")

    # encode
    enc = sub.add_parser("encode", help="SMILES columns -> descriptor feature CSV")
    enc.add_argument("--input", required=True, help="Input CSV with SMILES columns")
    enc.add_argument("--output-dir", required=True, help="Output directory")
    enc.add_argument(
        "--smiles-cols", nargs="+", required=True,
        help="Column names containing SMILES strings",
    )
    enc.add_argument(
        "--morgan-bits", type=int, default=64,
        help="Morgan fingerprint length per SMILES column (default: 64)",
    )

    # decode
    dec = sub.add_parser("decode", help="Descriptors -> nearest molecules (k-NN)")
    dec.add_argument("--catalog", required=True, help="catalog.csv from encode step")
    dec.add_argument("--query", required=True, help="JSON dict of descriptor values")
    dec.add_argument("--k", type=int, default=3, help="Number of nearest neighbours")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "encode":
        _cmd_encode(args)
    elif args.command == "decode":
        _cmd_decode(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
