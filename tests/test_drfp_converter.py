"""Tests for the DRFP reaction fingerprint converter."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bo_workflow.converters.reaction_drfp import (
    _cmd_decode,
    decode_nearest,
    encode_reactions,
)


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


def test_decode_rejects_invalid_fp_column_names() -> None:
    catalog = pd.DataFrame(
        {
            "fp_0": [1, 0],
            "fp_bad": [0, 1],
            "rxn_smiles": ["A>>B", "C>>D"],
        }
    )
    query_fp = np.array([1], dtype=np.uint8)

    with pytest.raises(ValueError, match="Invalid fingerprint column names"):
        decode_nearest(query_fp, catalog, k=1)


def test_decode_rejects_catalog_without_fp_columns() -> None:
    catalog = pd.DataFrame({"rxn_smiles": ["A>>B"]})
    query_fp = np.array([], dtype=np.uint8)

    with pytest.raises(ValueError, match="Catalog must contain fingerprint columns"):
        decode_nearest(query_fp, catalog, k=1)


def test_decode_rejects_non_positive_k() -> None:
    catalog = pd.DataFrame({"fp_0": [1], "rxn_smiles": ["A>>B"]})
    query_fp = np.array([1], dtype=np.uint8)

    with pytest.raises(ValueError, match="k must be >= 1"):
        decode_nearest(query_fp, catalog, k=0)


def test_encode_rejects_passthrough_fp_collisions(
    bh_rxns: Path, tmp_path: Path
) -> None:
    df = pd.read_csv(bh_rxns).head(4).copy()
    df["fp_0"] = [0, 0, 0, 0]
    input_path = tmp_path / "with_collision.csv"
    df.to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="collide with generated fingerprint columns"):
        encode_reactions(input_path, n_bits=64, rxn_col="rxn_smiles")


def test_cmd_decode_supports_nested_query_payload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog = pd.DataFrame(
        {
            "fp_0": [1, 0],
            "fp_1": [1, 0],
            "rxn_smiles": ["A>>B", "C>>D"],
        }
    )
    catalog_path = tmp_path / "catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    args = argparse.Namespace(
        catalog=str(catalog_path),
        query=json.dumps({"x": {"fp_0": 1, "fp_1": 1}}),
        k=1,
    )
    _cmd_decode(args)

    output = capsys.readouterr().out
    result = json.loads(output)
    assert result[0]["rxn_smiles"] == "A>>B"


def test_decode_results_are_json_serializable_python_scalars() -> None:
    catalog = pd.DataFrame(
        {
            "fp_0": [1, 0],
            "fp_1": [1, 0],
            "yield": np.array([88, 12], dtype=np.int64),
            "rxn_smiles": ["A>>B", "C>>D"],
        }
    )
    query_fp = np.array([1, 1], dtype=np.uint8)

    results = decode_nearest(query_fp, catalog, k=1)
    payload = json.dumps(results)
    assert isinstance(payload, str)
    assert isinstance(results[0]["yield"], int)
