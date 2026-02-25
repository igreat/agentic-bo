"""Reaction SMILES -> DRFP fingerprint converter.

Column-focused design: the converter encodes a single reaction SMILES column
into DRFP fingerprint bits and passes all other columns through unchanged.
The engine decides which column is the target at ``init`` time.

Encode: takes a CSV, replaces the reaction SMILES column with fingerprint bits,
and writes two files:
  - ``features.csv`` — fingerprint bits + all other columns (feed to engine)
  - ``catalog.csv`` — same as features.csv but also keeps the original
    reaction SMILES column (used for decode lookup)

Decode: given a fingerprint vector (e.g. from a HEBO suggestion), finds the
nearest reactions in the catalog by Tanimoto similarity.

Usage
-----
# Encode: raw reactions -> feature CSV + lookup catalog
python -m bo_workflow.converters.reaction_drfp encode \
    --input data/buchwald_hartwig_rxns.csv \
    --output-dir data/bh_drfp \
    --rxn-col rxn_smiles --n-bits 256

# Decode: find nearest reaction to a query fingerprint
python -m bo_workflow.converters.reaction_drfp decode \
    --catalog data/bh_drfp/catalog.csv \
    --query '{"fp_0": 1, "fp_1": 0, ...}' \
    --k 3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from drfp import DrfpEncoder


# ---------------------------------------------------------------------------
# Encode: reaction SMILES column -> DRFP fingerprint columns
# ---------------------------------------------------------------------------

def encode_reactions(
    input_path: Path,
    n_bits: int = 256,
    rxn_col: str = "rxn_smiles",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a CSV with reaction SMILES, compute DRFP.

    The converter is column-focused: it encodes ``rxn_col`` into fingerprint
    bits and passes every other column through unchanged.  The engine decides
    which column is the optimisation target at ``init`` time.

    Returns
    -------
    (features_df, catalog_df)
        features_df: fp_0..fp_{n-1} + all original columns except rxn_col
        catalog_df:  fp_0..fp_{n-1} + all original columns (including rxn_col)
    """
    df = pd.read_csv(input_path)
    if rxn_col not in df.columns:
        raise ValueError(f"Column '{rxn_col}' not found. Available: {list(df.columns)}")

    rxn_list = df[rxn_col].tolist()
    fps = DrfpEncoder.encode(rxn_list, n_folded_length=n_bits)
    fp_array = np.array(fps, dtype=np.uint8)

    fp_cols = [f"fp_{i}" for i in range(n_bits)]
    fp_df = pd.DataFrame(fp_array, columns=fp_cols)

    # Pass through every non-rxn column unchanged
    for col in df.columns:
        if col != rxn_col:
            fp_df[col] = df[col].values

    # Catalog keeps the original rxn_col too (needed for decode lookup)
    catalog_df = fp_df.copy()
    catalog_df[rxn_col] = df[rxn_col].values

    return fp_df, catalog_df


# ---------------------------------------------------------------------------
# Decode: fingerprint -> nearest reaction via Tanimoto similarity
# ---------------------------------------------------------------------------

def _tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    """Tanimoto similarity between two binary vectors."""
    a = a.astype(bool)
    b = b.astype(bool)
    intersection = np.sum(a & b)
    union = np.sum(a | b)
    if union == 0:
        return 0.0
    return float(intersection / union)


def decode_nearest(
    query_fp: np.ndarray,
    catalog: pd.DataFrame,
    k: int = 3,
    rxn_col: str = "rxn_smiles",
) -> list[dict]:
    """Find k nearest reactions in the catalog by Tanimoto similarity.

    The catalog does not have to be the one produced by ``encode``.  Any CSV
    with matching ``fp_*`` columns works -- e.g. a large external database of
    commercially available reactions.  This is useful for human-in-the-loop
    workflows where BO suggests novel fingerprints outside the training set.

    Parameters
    ----------
    query_fp : 1-D array of fingerprint bits (rounded to 0/1)
    catalog : DataFrame with fp_0..fp_N columns plus rxn_smiles
    k : number of nearest neighbors to return

    Returns
    -------
    List of dicts with keys: rank, similarity, and all non-fingerprint columns
    """
    fp_cols = sorted(
        [c for c in catalog.columns if c.startswith("fp_")],
        key=lambda c: int(c.split("_")[1]),
    )
    catalog_fps = catalog[fp_cols].values.astype(np.uint8)

    similarities = np.array([_tanimoto(query_fp, row) for row in catalog_fps])
    top_k_idx = np.argsort(similarities)[::-1][:k]

    results = []
    meta_cols = [c for c in catalog.columns if c not in fp_cols]
    for rank, idx in enumerate(top_k_idx):
        entry = {
            "rank": rank + 1,
            "similarity": round(float(similarities[idx]), 4),
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

    features_df, catalog_df = encode_reactions(
        Path(args.input),
        n_bits=args.n_bits,
        rxn_col=args.rxn_col,
    )

    features_path = out_dir / "features.csv"
    catalog_path = out_dir / "catalog.csv"

    features_df.to_csv(features_path, index=False)
    catalog_df.to_csv(catalog_path, index=False)

    n_fp = sum(1 for c in features_df.columns if c.startswith("fp_"))
    non_fp = [c for c in features_df.columns if not c.startswith("fp_")]
    print(json.dumps({
        "status": "ok",
        "input": str(args.input),
        "features_csv": str(features_path),
        "catalog_csv": str(catalog_path),
        "reactions": len(features_df),
        "fingerprint_bits": n_fp,
        "passthrough_columns": non_fp,
        "rxn_col": args.rxn_col,
    }, indent=2))


def _cmd_decode(args: argparse.Namespace) -> None:
    catalog = pd.read_csv(args.catalog)
    query = json.loads(args.query)

    fp_cols = sorted(
        [c for c in catalog.columns if c.startswith("fp_")],
        key=lambda c: int(c.split("_")[1]),
    )
    # Round continuous HEBO suggestions to binary
    query_fp = np.array(
        [1 if float(query.get(c, 0)) > 0.5 else 0 for c in fp_cols],
        dtype=np.uint8,
    )

    results = decode_nearest(query_fp, catalog, k=args.k)
    print(json.dumps(results, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reaction SMILES <-> DRFP fingerprint converter",
    )
    sub = parser.add_subparsers(dest="command")

    # encode
    enc = sub.add_parser("encode", help="Reaction SMILES -> DRFP feature CSV")
    enc.add_argument("--input", required=True, help="Input CSV with rxn_smiles column")
    enc.add_argument("--output-dir", required=True, help="Output directory for features.csv and catalog.csv")
    enc.add_argument("--rxn-col", default="rxn_smiles", help="Column containing reaction SMILES")
    enc.add_argument("--n-bits", type=int, default=256, help="DRFP fingerprint length")

    # decode
    dec = sub.add_parser("decode", help="Fingerprint -> nearest reaction (k-NN)")
    dec.add_argument("--catalog", required=True, help="catalog.csv from encode step")
    dec.add_argument("--query", required=True, help="JSON dict of fingerprint values (from HEBO suggestion)")
    dec.add_argument("--k", type=int, default=3, help="Number of nearest neighbors")

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
