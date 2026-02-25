"""Tests for the DRFP reaction fingerprint converter."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bo_workflow.converters.reaction_drfp import decode_nearest, encode_reactions


@pytest.fixture()
def bh_rxns() -> Path:
    path = Path(__file__).resolve().parents[1] / "data" / "buchwald_hartwig_rxns.csv"
    if not path.exists():
        pytest.skip("buchwald_hartwig_rxns.csv not found")
    return path


def test_encode_produces_expected_columns(bh_rxns: Path, tmp_path: Path) -> None:
    """Encode should produce fp bit columns + passthrough columns, no rxn_smiles."""
    features_df, catalog_df = encode_reactions(bh_rxns, n_bits=64, rxn_col="rxn_smiles")

    fp_cols = [c for c in features_df.columns if c.startswith("fp_")]
    assert len(fp_cols) == 64

    # rxn_smiles should be removed from features but kept in catalog
    assert "rxn_smiles" not in features_df.columns
    assert "rxn_smiles" in catalog_df.columns

    # passthrough columns should be in both
    for col in ("aryl_halide", "ligand", "base", "additive", "yield"):
        assert col in features_df.columns
        assert col in catalog_df.columns

    assert len(features_df) == len(catalog_df)


def test_encode_fingerprints_are_binary(bh_rxns: Path) -> None:
    """All fingerprint values should be 0 or 1."""
    features_df, _ = encode_reactions(bh_rxns, n_bits=64, rxn_col="rxn_smiles")

    fp_cols = [c for c in features_df.columns if c.startswith("fp_")]
    fp_values = features_df[fp_cols].values
    assert set(np.unique(fp_values)).issubset({0, 1})


def test_decode_returns_nearest_reactions(bh_rxns: Path) -> None:
    """Decode should return k nearest reactions sorted by descending similarity."""
    _, catalog_df = encode_reactions(bh_rxns, n_bits=64, rxn_col="rxn_smiles")

    # Use the first row's fingerprint as query -- should match itself perfectly
    fp_cols = sorted(
        [c for c in catalog_df.columns if c.startswith("fp_")],
        key=lambda c: int(c.split("_")[1]),
    )
    query_fp = catalog_df[fp_cols].iloc[0].values.astype(np.uint8)

    results = decode_nearest(query_fp, catalog_df, k=3)

    assert len(results) == 3
    assert results[0]["rank"] == 1
    assert results[0]["similarity"] == 1.0  # exact match
    assert "rxn_smiles" in results[0]

    # similarities should be descending
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)
