"""Combined DRFP + molecular descriptor converter.

Composes both converters: DRFP encodes a reaction SMILES column into
fingerprint bits, and molecular descriptors encode per-component SMILES
columns into physicochemical features.  The result is a single feature
CSV with both representations, and a combined k-NN decode that handles
the mixed binary + continuous feature space.

Encode: takes a CSV with a reaction SMILES column and individual component
SMILES columns, produces combined features and a lookup catalog.

Decode: given a mixed query (fingerprint bits + descriptor values), finds
the nearest reactions using a weighted Tanimoto + normalised Euclidean
similarity metric.

Usage
-----
# Encode both representations
python -m bo_workflow.converters.combined encode \
    --input data/buchwald_hartwig_rxns.csv \
    --output-dir data/bh_combined \
    --rxn-col rxn_smiles \
    --smiles-cols aryl_halide ligand base additive \
    --n-bits 256

# Decode with combined metric
python -m bo_workflow.converters.combined decode \
    --catalog data/bh_combined/catalog.csv \
    --query '{"fp_0": 1, "ah_MolWt": 200}' \
    --k 3 --fp-weight 0.5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bo_workflow.converters.reaction_drfp import encode_reactions, _tanimoto
from bo_workflow.converters.molecule_descriptors import (
    encode_molecules,
    is_descriptor_col,
)


# ---------------------------------------------------------------------------
# Encode: chain both converters
# ---------------------------------------------------------------------------

def encode_combined(
    input_path: Path,
    rxn_col: str = "rxn_smiles",
    smiles_cols: list[str] | None = None,
    n_bits: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run both DRFP and molecular descriptor encoding.

    1. DRFP encodes ``rxn_col`` -> fp_0..fp_{n_bits-1}
    2. Molecular descriptors encode each ``smiles_col`` -> prefixed descriptors
    3. Merge: fp columns + descriptor columns + passthrough (yield, etc.)

    Parameters
    ----------
    input_path : Path to CSV with reaction SMILES + component SMILES columns
    rxn_col : Column containing reaction SMILES
    smiles_cols : Columns containing individual component SMILES.
        If None, auto-detects by looking for non-rxn, non-numeric columns
        that parse as SMILES.
    n_bits : DRFP fingerprint length

    Returns
    -------
    (features_df, catalog_df)
        features_df: fp columns + descriptor columns + passthrough columns
        catalog_df: same + original SMILES and rxn_smiles columns
    """
    df = pd.read_csv(input_path)

    if smiles_cols is None:
        smiles_cols = _auto_detect_smiles_cols(df, rxn_col)
        if not smiles_cols:
            raise ValueError(
                "No SMILES columns detected. Provide --smiles-cols explicitly."
            )

    # Step 1: DRFP encoding
    drfp_features, drfp_catalog = encode_reactions(input_path, n_bits=n_bits, rxn_col=rxn_col)
    fp_cols = [c for c in drfp_features.columns if c.startswith("fp_")]

    # Step 2: Molecular descriptor encoding
    desc_features, _desc_catalog = encode_molecules(input_path, smiles_cols=smiles_cols)
    desc_cols = [c for c in desc_features.columns if is_descriptor_col(c)]

    # Step 3: Merge
    # Passthrough = columns that are neither fp, descriptors, rxn_col, nor smiles_cols
    passthrough_cols = [
        c for c in df.columns
        if c != rxn_col and c not in smiles_cols
    ]

    features_df = pd.DataFrame()
    # Fingerprint columns
    for col in fp_cols:
        features_df[col] = drfp_features[col]
    # Descriptor columns
    for col in desc_cols:
        features_df[col] = desc_features[col]
    # Passthrough columns (yield, etc.)
    for col in passthrough_cols:
        features_df[col] = df[col].values

    # Catalog: features + original SMILES columns for lookup
    catalog_df = features_df.copy()
    catalog_df[rxn_col] = df[rxn_col].values
    for col in smiles_cols:
        catalog_df[col] = df[col].values

    return features_df, catalog_df


def _auto_detect_smiles_cols(df: pd.DataFrame, rxn_col: str) -> list[str]:
    """Heuristic: non-numeric, non-rxn object columns likely hold SMILES."""
    from rdkit import Chem

    candidates = []
    for col in df.columns:
        if col == rxn_col:
            continue
        if df[col].dtype != object:
            continue
        # Check if a sample of values parse as SMILES
        sample = df[col].dropna().head(5)
        if len(sample) == 0:
            continue
        parsed = sum(1 for s in sample if Chem.MolFromSmiles(str(s)) is not None)
        if parsed >= len(sample) * 0.6:
            candidates.append(col)
    return candidates


# ---------------------------------------------------------------------------
# Decode: mixed-metric k-NN (Tanimoto + normalised Euclidean)
# ---------------------------------------------------------------------------

def decode_nearest_combined(
    query: dict[str, float],
    catalog: pd.DataFrame,
    k: int = 3,
    fp_weight: float = 0.5,
) -> list[dict]:
    """Combined k-NN using weighted Tanimoto + normalised Euclidean.

    1. Identify fp_* columns -> compute Tanimoto similarity (0-1)
    2. Identify descriptor columns -> compute Euclidean distance,
       min-max normalise to 0-1, convert to similarity (1 - normalised_dist)
    3. Combined similarity = fp_weight * tanimoto + (1 - fp_weight) * desc_similarity
    4. Return top-k by combined similarity

    Parameters
    ----------
    query : dict mapping column names to values (fp_0, fp_1, ..., ah_MolWt, ...)
    catalog : DataFrame from encode_combined (features + original columns)
    k : number of nearest neighbours
    fp_weight : weight for fingerprint similarity (0-1).
        0.5 = equal weight. 1.0 = fingerprint only. 0.0 = descriptors only.

    Returns
    -------
    List of dicts with rank, combined_similarity, tanimoto_similarity,
    descriptor_similarity, and all non-feature columns.
    """
    # Partition columns
    fp_cols = sorted(
        [c for c in catalog.columns if c.startswith("fp_")],
        key=lambda c: int(c.split("_")[1]),
    )
    desc_cols = sorted([c for c in catalog.columns if is_descriptor_col(c)])
    meta_cols = [c for c in catalog.columns if c not in fp_cols and c not in desc_cols]

    n = len(catalog)

    # --- Fingerprint similarity (Tanimoto) ---
    if fp_cols:
        query_fp = np.array(
            [1 if float(query.get(c, 0)) > 0.5 else 0 for c in fp_cols],
            dtype=np.uint8,
        )
        cat_fps = catalog[fp_cols].values.astype(np.uint8)
        tanimoto_sims = np.array([_tanimoto(query_fp, row) for row in cat_fps])
    else:
        tanimoto_sims = np.ones(n)

    # --- Descriptor similarity (normalised Euclidean) ---
    if desc_cols:
        query_desc = np.array([float(query.get(c, 0)) for c in desc_cols])
        cat_desc = catalog[desc_cols].values.astype(np.float64)

        # Min-max normalise using catalog range
        col_min = cat_desc.min(axis=0)
        col_max = cat_desc.max(axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1.0

        normed_cat = (cat_desc - col_min) / col_range
        normed_query = (query_desc - col_min) / col_range

        distances = np.linalg.norm(normed_cat - normed_query, axis=1)
        # Normalise distances to 0-1 range, then convert to similarity
        max_dist = distances.max() if distances.max() > 0 else 1.0
        desc_sims = 1.0 - (distances / max_dist)
    else:
        desc_sims = np.ones(n)

    # --- Combined similarity ---
    combined = fp_weight * tanimoto_sims + (1 - fp_weight) * desc_sims
    top_k_idx = np.argsort(combined)[::-1][:k]

    results = []
    for rank, idx in enumerate(top_k_idx):
        entry = {
            "rank": rank + 1,
            "combined_similarity": round(float(combined[idx]), 4),
            "tanimoto_similarity": round(float(tanimoto_sims[idx]), 4),
            "descriptor_similarity": round(float(desc_sims[idx]), 4),
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

    features_df, catalog_df = encode_combined(
        Path(args.input),
        rxn_col=args.rxn_col,
        smiles_cols=args.smiles_cols,
        n_bits=args.n_bits,
    )

    features_path = out_dir / "features.csv"
    catalog_path = out_dir / "catalog.csv"

    features_df.to_csv(features_path, index=False)
    catalog_df.to_csv(catalog_path, index=False)

    fp_cols = [c for c in features_df.columns if c.startswith("fp_")]
    desc_cols = [c for c in features_df.columns if is_descriptor_col(c)]
    passthrough = [c for c in features_df.columns if c not in fp_cols and c not in desc_cols]

    print(json.dumps({
        "status": "ok",
        "input": str(args.input),
        "features_csv": str(features_path),
        "catalog_csv": str(catalog_path),
        "rows": len(features_df),
        "fingerprint_columns": len(fp_cols),
        "descriptor_columns": len(desc_cols),
        "passthrough_columns": passthrough,
        "rxn_col": args.rxn_col,
        "smiles_cols": args.smiles_cols,
    }, indent=2))


def _cmd_decode(args: argparse.Namespace) -> None:
    catalog = pd.read_csv(args.catalog)
    query = json.loads(args.query)

    results = decode_nearest_combined(
        query, catalog, k=args.k, fp_weight=args.fp_weight,
    )
    print(json.dumps(results, indent=2, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combined DRFP + molecular descriptor converter",
    )
    sub = parser.add_subparsers(dest="command")

    # encode
    enc = sub.add_parser("encode", help="Reaction + SMILES -> combined feature CSV")
    enc.add_argument("--input", required=True, help="Input CSV")
    enc.add_argument("--output-dir", required=True, help="Output directory")
    enc.add_argument("--rxn-col", default="rxn_smiles", help="Reaction SMILES column")
    enc.add_argument(
        "--smiles-cols", nargs="+", default=None,
        help="Component SMILES columns (auto-detect if omitted)",
    )
    enc.add_argument("--n-bits", type=int, default=256, help="DRFP fingerprint length")

    # decode
    dec = sub.add_parser("decode", help="Combined features -> nearest reactions (k-NN)")
    dec.add_argument("--catalog", required=True, help="catalog.csv from encode step")
    dec.add_argument("--query", required=True, help="JSON dict of feature values")
    dec.add_argument("--k", type=int, default=3, help="Number of nearest neighbours")
    dec.add_argument(
        "--fp-weight", type=float, default=0.5,
        help="Weight for fingerprint similarity (0-1). Default: 0.5",
    )

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
