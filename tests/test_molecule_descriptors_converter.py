"""Tests for the molecule descriptors converter.

Covers encode and decode for single-SMILES inputs (EGFR-style)
and multi-SMILES inputs, plus edge cases.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bo_workflow.converters.molecule_descriptors import (
    canonicalize_smiles,
    compute_descriptors,
    decode_nearest,
    encode_molecules,
    is_descriptor_col,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A handful of real EGFR-active SMILES for realistic tests.
_EGFR_SMILES = [
    "Cc1cc(C)c(/C=C2\\C(=O)Nc3ncnc(Nc4ccc(F)c(Cl)c4)c32)[nH]1",  # pIC50 ~10.4
    "CN1CCN(c2ccc(Nc3ncnc4cc(OC)c(OC)cc34)cc2)CC1",               # gefitinib-like
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",                             # polycyclic
    "CCOc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCC",                      # erlotinib-like
    "O=C(Nc1cccc(Nc2ncnc3ccccc23)c1)c1ccccc1",                     # simple scaffold
]

_EGFR_IC50 = [41.0, 170.0, 9300.0, 2.0, 500.0]


@pytest.fixture()
def mini_egfr_csv(tmp_path: Path) -> Path:
    """Small EGFR-style CSV with smiles and ic50_nM columns."""
    df = pd.DataFrame({"smiles": _EGFR_SMILES, "ic50_nM": _EGFR_IC50})
    path = tmp_path / "egfr_mini.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def encoded(mini_egfr_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pre-encoded features and catalog for the mini EGFR set."""
    return encode_molecules(mini_egfr_csv, smiles_cols=["smiles"], morgan_bits=64)


# ---------------------------------------------------------------------------
# canonicalize_smiles
# ---------------------------------------------------------------------------


def test_canonicalize_valid_smiles() -> None:
    result = canonicalize_smiles("c1ccccc1")
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


def test_canonicalize_invalid_smiles_returns_none() -> None:
    assert canonicalize_smiles("not_a_smiles!!!") is None


def test_canonicalize_is_idempotent() -> None:
    smi = "Cc1ccccc1"
    assert canonicalize_smiles(canonicalize_smiles(smi)) == canonicalize_smiles(smi)


# ---------------------------------------------------------------------------
# compute_descriptors
# ---------------------------------------------------------------------------


def test_compute_descriptors_returns_expected_keys() -> None:
    desc = compute_descriptors("c1ccccc1", morgan_bits=32)
    assert "MolWt" in desc
    assert "LogP" in desc
    assert "GasteigerMean" in desc
    assert "AromaticProportion" in desc
    assert "mfp_0" in desc
    assert "mfp_31" in desc
    assert len([k for k in desc if k.startswith("mfp_")]) == 32


def test_compute_descriptors_benzene_values() -> None:
    desc = compute_descriptors("c1ccccc1")
    assert math.isfinite(desc["MolWt"])
    assert desc["MolWt"] == pytest.approx(78.046, abs=0.1)
    assert desc["AromaticProportion"] == pytest.approx(1.0)
    assert desc["NumHDonors"] == 0


def test_compute_descriptors_invalid_smiles_returns_nan() -> None:
    desc = compute_descriptors("INVALID")
    assert math.isnan(desc["MolWt"])
    assert math.isnan(desc["GasteigerMean"])
    # Morgan bits should be 0, not NaN
    assert desc["mfp_0"] == 0


# ---------------------------------------------------------------------------
# encode_molecules — column structure
# ---------------------------------------------------------------------------


def test_encode_produces_descriptor_columns(encoded: tuple) -> None:
    features_df, catalog_df = encoded
    desc_cols = [c for c in features_df.columns if is_descriptor_col(c)]
    # 10 basic + 1 AromaticProportion + 4 Gasteiger + 64 Morgan = 79
    assert len(desc_cols) == 79


def test_encode_descriptor_columns_are_prefixed(encoded: tuple) -> None:
    features_df, _ = encoded
    desc_cols = [c for c in features_df.columns if is_descriptor_col(c)]
    assert all(c.startswith("smil_") for c in desc_cols)


def test_encode_smiles_removed_from_features(encoded: tuple) -> None:
    features_df, catalog_df = encoded
    assert "smiles" not in features_df.columns
    assert "smiles" in catalog_df.columns


def test_encode_passthrough_columns_preserved(encoded: tuple) -> None:
    features_df, catalog_df = encoded
    assert "ic50_nM" in features_df.columns
    assert "ic50_nM" in catalog_df.columns


def test_encode_row_count_preserved(encoded: tuple) -> None:
    features_df, catalog_df = encoded
    assert len(features_df) == len(_EGFR_SMILES)
    assert len(catalog_df) == len(_EGFR_SMILES)


def test_encode_features_and_catalog_same_length(encoded: tuple) -> None:
    features_df, catalog_df = encoded
    assert len(features_df) == len(catalog_df)


def test_encode_descriptor_values_are_finite(encoded: tuple) -> None:
    features_df, _ = encoded
    desc_cols = [c for c in features_df.columns if is_descriptor_col(c)]
    # Morgan bits are 0/1 integers — all finite
    # Scalar descriptors should be finite for valid SMILES
    for col in desc_cols:
        vals = pd.to_numeric(features_df[col], errors="coerce")
        assert vals.notna().all(), f"NaN in descriptor column {col}"
        assert np.isfinite(vals.values).all(), f"Inf in descriptor column {col}"


def test_encode_morgan_bits_are_binary(encoded: tuple) -> None:
    features_df, _ = encoded
    mfp_cols = [c for c in features_df.columns if "_mfp_" in c]
    assert len(mfp_cols) == 64
    unique_vals = set(features_df[mfp_cols].values.ravel().tolist())
    assert unique_vals.issubset({0, 1})


def test_encode_raises_on_missing_smiles_col(mini_egfr_csv: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        encode_molecules(mini_egfr_csv, smiles_cols=["nonexistent_col"])


# ---------------------------------------------------------------------------
# encode_molecules — multi-SMILES column
# ---------------------------------------------------------------------------


def test_encode_two_smiles_columns(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "mol_a": ["c1ccccc1", "CCO"],
        "mol_b": ["c1cccnc1", "CC(=O)O"],
        "target": [1.0, 2.0],
    })
    csv_path = tmp_path / "two_cols.csv"
    df.to_csv(csv_path, index=False)

    features_df, catalog_df = encode_molecules(
        csv_path, smiles_cols=["mol_a", "mol_b"], morgan_bits=16
    )

    # Both SMILES columns removed from features, present in catalog
    assert "mol_a" not in features_df.columns
    assert "mol_b" not in features_df.columns
    assert "mol_a" in catalog_df.columns
    assert "mol_b" in catalog_df.columns

    # Each SMILES column gets its own prefix
    ma_cols = [c for c in features_df.columns if c.startswith("ma_")]
    mb_cols = [c for c in features_df.columns if c.startswith("mb_")]
    assert len(ma_cols) > 0
    assert len(mb_cols) > 0

    # Target passthrough
    assert "target" in features_df.columns


# ---------------------------------------------------------------------------
# decode_nearest — round-trip
# ---------------------------------------------------------------------------


def test_decode_self_is_nearest(encoded: tuple) -> None:
    """Encoding a molecule and decoding its own descriptor vector should return itself."""
    features_df, catalog_df = encoded
    desc_cols = [c for c in features_df.columns if is_descriptor_col(c)]

    # Use the first row's descriptor vector as query
    query_vec = features_df[desc_cols].iloc[0].values.astype(np.float64)
    results = decode_nearest(query_vec, catalog_df, desc_cols, k=1)

    assert len(results) == 1
    assert results[0]["rank"] == 1
    assert results[0]["distance"] == pytest.approx(0.0, abs=1e-6)
    assert results[0]["smiles"] == catalog_df["smiles"].iloc[0]


def test_decode_returns_k_results(encoded: tuple) -> None:
    features_df, catalog_df = encoded
    desc_cols = [c for c in features_df.columns if is_descriptor_col(c)]
    query_vec = features_df[desc_cols].iloc[0].values.astype(np.float64)

    for k in (1, 3, 5):
        results = decode_nearest(query_vec, catalog_df, desc_cols, k=k)
        assert len(results) == k


def test_decode_distances_are_non_decreasing(encoded: tuple) -> None:
    features_df, catalog_df = encoded
    desc_cols = [c for c in features_df.columns if is_descriptor_col(c)]
    query_vec = features_df[desc_cols].iloc[2].values.astype(np.float64)

    results = decode_nearest(query_vec, catalog_df, desc_cols, k=5)
    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)


def test_decode_rank_field_is_sequential(encoded: tuple) -> None:
    features_df, catalog_df = encoded
    desc_cols = [c for c in features_df.columns if is_descriptor_col(c)]
    query_vec = features_df[desc_cols].iloc[0].values.astype(np.float64)

    results = decode_nearest(query_vec, catalog_df, desc_cols, k=3)
    assert [r["rank"] for r in results] == [1, 2, 3]


def test_decode_includes_passthrough_metadata(encoded: tuple) -> None:
    features_df, catalog_df = encoded
    desc_cols = [c for c in features_df.columns if is_descriptor_col(c)]
    query_vec = features_df[desc_cols].iloc[0].values.astype(np.float64)

    results = decode_nearest(query_vec, catalog_df, desc_cols, k=1)
    assert "smiles" in results[0]
    assert "ic50_nM" in results[0]


def test_decode_results_are_json_serializable(encoded: tuple) -> None:
    features_df, catalog_df = encoded
    desc_cols = [c for c in features_df.columns if is_descriptor_col(c)]
    query_vec = features_df[desc_cols].iloc[0].values.astype(np.float64)

    results = decode_nearest(query_vec, catalog_df, desc_cols, k=3)
    payload = json.dumps(results)  # must not require default=str
    assert isinstance(payload, str)


def test_decode_k_capped_at_catalog_size(encoded: tuple) -> None:
    features_df, catalog_df = encoded
    desc_cols = [c for c in features_df.columns if is_descriptor_col(c)]
    query_vec = features_df[desc_cols].iloc[0].values.astype(np.float64)

    results = decode_nearest(query_vec, catalog_df, desc_cols, k=1000)
    assert len(results) == len(catalog_df)


# ---------------------------------------------------------------------------
# encode → decode round-trip: all molecules recover themselves
# ---------------------------------------------------------------------------


def test_all_molecules_recover_on_self_decode(encoded: tuple) -> None:
    """Every molecule's own descriptor vector should decode back to itself as rank-1."""
    features_df, catalog_df = encoded
    desc_cols = [c for c in features_df.columns if is_descriptor_col(c)]

    for i in range(len(features_df)):
        query_vec = features_df[desc_cols].iloc[i].values.astype(np.float64)
        results = decode_nearest(query_vec, catalog_df, desc_cols, k=1)
        assert results[0]["smiles"] == catalog_df["smiles"].iloc[i], (
            f"Row {i}: expected {catalog_df['smiles'].iloc[i]}, "
            f"got {results[0]['smiles']}"
        )


# ---------------------------------------------------------------------------
# is_descriptor_col
# ---------------------------------------------------------------------------


def test_is_descriptor_col_recognises_prefixed_columns() -> None:
    assert is_descriptor_col("smil_MolWt")
    assert is_descriptor_col("smil_mfp_0")
    assert is_descriptor_col("ah_GasteigerMean")
    assert is_descriptor_col("lig_AromaticProportion")


def test_is_descriptor_col_rejects_non_descriptor_columns() -> None:
    assert not is_descriptor_col("smiles")
    assert not is_descriptor_col("ic50_nM")
    assert not is_descriptor_col("target")
    assert not is_descriptor_col("canonical_smiles")


# ---------------------------------------------------------------------------
# Full EGFR dataset tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def egfr_encoded(egfr_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return encode_molecules(egfr_csv, smiles_cols=["smiles"], morgan_bits=64)


def test_egfr_encode_row_count(egfr_csv: Path, egfr_encoded: tuple) -> None:
    features_df, catalog_df = egfr_encoded
    raw = pd.read_csv(egfr_csv)
    assert len(features_df) == len(raw)
    assert len(catalog_df) == len(raw)


def test_egfr_encode_descriptor_count(egfr_encoded: tuple) -> None:
    features_df, _ = egfr_encoded
    desc_cols = [c for c in features_df.columns if is_descriptor_col(c)]
    assert len(desc_cols) == 79


def test_egfr_encode_nan_only_in_gasteiger(egfr_encoded: tuple) -> None:
    """NaNs should only appear in Gasteiger charge columns (failed charge calc)
    and only for a tiny fraction of molecules (<0.2%)."""
    features_df, _ = egfr_encoded
    desc_cols = [c for c in features_df.columns if is_descriptor_col(c)]
    nan_counts = features_df[desc_cols].isnull().sum()
    nan_cols = nan_counts[nan_counts > 0].index.tolist()

    # Only Gasteiger columns should have NaNs
    assert all("Gasteiger" in c for c in nan_cols), (
        f"Unexpected NaN in non-Gasteiger columns: {[c for c in nan_cols if 'Gasteiger' not in c]}"
    )
    # Fewer than 0.2% of rows affected
    nan_rows = features_df[desc_cols].isnull().any(axis=1).sum()
    assert nan_rows / len(features_df) < 0.002, (
        f"{nan_rows} rows with NaN ({nan_rows/len(features_df):.1%})"
    )


def test_egfr_encode_target_passthrough(egfr_encoded: tuple) -> None:
    features_df, catalog_df = egfr_encoded
    assert "ic50_nM" in features_df.columns
    assert "smiles" not in features_df.columns
    assert "smiles" in catalog_df.columns


def test_egfr_decode_self_round_trip(egfr_encoded: tuple) -> None:
    """Sample 20 molecules (with finite descriptors) and verify each decodes to itself."""
    features_df, catalog_df = egfr_encoded
    desc_cols = [c for c in features_df.columns if is_descriptor_col(c)]

    # Drop NaN rows from both so decode_nearest normalization stays finite
    finite_mask = features_df[desc_cols].notnull().all(axis=1)
    features_clean = features_df[finite_mask].reset_index(drop=True)
    catalog_clean = catalog_df[finite_mask].reset_index(drop=True)

    rng = np.random.default_rng(42)
    indices = rng.choice(len(features_clean), size=20, replace=False)

    for i in indices:
        query_vec = features_clean[desc_cols].iloc[i].values.astype(np.float64)
        results = decode_nearest(query_vec, catalog_clean, desc_cols, k=1)
        # Distance 0.0 is sufficient — two stereoisomers may share identical
        # descriptor vectors (e.g. [C@@H] vs [C@H]), so SMILES equality is
        # too strict.
        assert results[0]["distance"] == pytest.approx(0.0, abs=1e-6), (
            f"Row {i}: self-decode distance {results[0]['distance']} != 0"
        )


def test_egfr_decode_known_potent_molecule(egfr_encoded: tuple, egfr_csv: Path) -> None:
    """The most potent molecule with finite descriptors should decode to itself."""
    features_df, catalog_df = egfr_encoded
    desc_cols = [c for c in features_df.columns if is_descriptor_col(c)]

    finite_mask = features_df[desc_cols].notnull().all(axis=1)
    features_clean = features_df[finite_mask].reset_index(drop=True)
    catalog_clean = catalog_df[finite_mask].reset_index(drop=True)

    best_idx = int(catalog_clean["ic50_nM"].idxmin())
    query_vec = features_clean[desc_cols].iloc[best_idx].values.astype(np.float64)
    results = decode_nearest(query_vec, catalog_clean, desc_cols, k=1)

    assert results[0]["distance"] == pytest.approx(0.0, abs=1e-6)
    assert results[0]["ic50_nM"] == pytest.approx(catalog_clean["ic50_nM"].iloc[best_idx])
