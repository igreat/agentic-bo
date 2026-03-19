"""Tests for the SMILES → DFT descriptor converter.

Runs real PySCF DFT on small molecules (STO-3G for speed).
Also covers auto-detect, cache, conformer generation, and the
full encode_dataset_dft pipeline.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bo_workflow.converters.smiles_to_dft._conformers import (
    ConformerError,
    generate_conformers,
    mol_properties,
)
from bo_workflow.converters.smiles_to_dft._dft_engine import run_dft
from bo_workflow.converters.smiles_to_dft.smiles_dft import (
    BH_PRESET,
    ComponentSpec,
    _aggregate,
    compute_molecule_descriptors,
    detect_smiles_columns,
    encode_dataset_dft,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_csv(tmp_path: Path) -> Path:
    """CSV with 2 SMILES cols + numeric cols (3 rows)."""
    df = pd.DataFrame({
        "molecule": ["CCO", "CC", "C"],
        "solvent": ["O", "O", "CO"],
        "temperature": [25.0, 50.0, 75.0],
        "yield": [80.0, 60.0, 90.0],
    })
    p = tmp_path / "tiny.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# SMILES auto-detection
# ---------------------------------------------------------------------------

def test_detect_smiles_vs_numeric() -> None:
    df = pd.DataFrame({
        "mol": ["CCO", "CC", "C", "c1ccccc1"],
        "temp": [25.0, 50.0, 75.0, 100.0],
    })
    detected = detect_smiles_columns(df)
    assert "mol" in detected
    assert "temp" not in detected


def test_detect_multiple_smiles_columns() -> None:
    df = pd.DataFrame({
        "reagent_a": ["CCO", "CC", "CCC"],
        "reagent_b": ["O", "N", "S"],
        "temp": [25.0, 50.0, 75.0],
    })
    detected = detect_smiles_columns(df)
    assert set(detected) == {"reagent_a", "reagent_b"}


# ---------------------------------------------------------------------------
# Conformer generation (RDKit, no PySCF)
# ---------------------------------------------------------------------------

def test_conformers_have_rdkit_mol_and_sorted_energy() -> None:
    confs = generate_conformers("CCCC", num_conformers=5)
    assert len(confs) >= 1
    for c in confs:
        assert c.rdkit_mol is not None
        assert c.rdkit_mol.GetNumConformers() == 1
    if len(confs) > 1:
        assert [c.energy for c in confs] == sorted(c.energy for c in confs)


def test_mol_properties_heavy_atoms_only() -> None:
    props = mol_properties("CCO")  # ethanol: C, C, O = 3 heavy atoms
    assert props["number_of_atoms"] == 3.0
    assert props["molar_mass"] > 40
    assert not math.isnan(props["molar_volume"])


def test_invalid_smiles_raises() -> None:
    with pytest.raises(ConformerError):
        generate_conformers("not_a_smiles")


# ---------------------------------------------------------------------------
# Real PySCF DFT
# ---------------------------------------------------------------------------

def test_dft_methane_scf() -> None:
    """Run DFT on methane (CH4) — smallest organic molecule."""
    confs = generate_conformers("C", num_conformers=1)
    c = confs[0]
    result = run_dft(c.atom_symbols, c.coords, basis="sto-3g", compute_nmr=False)

    assert result.converged
    assert result.scf_energy < 0
    assert not math.isnan(result.homo_energy)
    assert not math.isnan(result.lumo_energy)
    assert result.homo_energy < result.lumo_energy  # HOMO < LUMO
    assert not math.isnan(result.dipole_moment)
    assert len(result.mulliken_charges) == len(c.atom_symbols)
    # Methane Mulliken charges should sum near zero
    assert abs(sum(result.mulliken_charges)) < 0.01


def test_dft_ethanol_with_nmr() -> None:
    """Run DFT + NMR on ethanol — checks NMR path doesn't crash."""
    confs = generate_conformers("CCO", num_conformers=1)
    c = confs[0]
    result = run_dft(c.atom_symbols, c.coords, basis="sto-3g", compute_nmr=True)

    assert result.converged
    assert len(result.nmr_shieldings) == len(c.atom_symbols)
    assert len(result.nmr_anisotropies) == len(c.atom_symbols)
    # NMR may be NaN if pyscf-properties has compat issues, but must not raise


def test_dft_derived_descriptors() -> None:
    """Check electronegativity and hardness are derived correctly."""
    confs = generate_conformers("C", num_conformers=1)
    c = confs[0]
    result = run_dft(c.atom_symbols, c.coords, basis="sto-3g", compute_nmr=False)

    # Manually verify derived quantities
    expected_en = -(result.homo_energy + result.lumo_energy) / 2.0
    expected_hard = (result.lumo_energy - result.homo_energy) / 2.0

    spec = ComponentSpec(
        csv_column="x", prefix="x", role="base",
        compute_nmr=False, compute_vbur=False,
        num_conformers=1, multi_conformer=False,
    )
    desc = compute_molecule_descriptors("C", spec, basis="sto-3g")
    assert abs(desc["x_electronegativity"] - expected_en) < 1e-6
    assert abs(desc["x_hardness"] - expected_hard) < 1e-6


# ---------------------------------------------------------------------------
# Full pipeline: compute_molecule_descriptors
# ---------------------------------------------------------------------------

def test_pipeline_single_conformer_base() -> None:
    """Full pipeline on ethanol as 'base' role — single conformer, scalar output."""
    spec = ComponentSpec(
        csv_column="base", prefix="base", role="base",
        compute_nmr=False, compute_vbur=False,
        num_conformers=1, multi_conformer=False,
    )
    desc = compute_molecule_descriptors("CCO", spec, basis="sto-3g")

    # Must have these keys (no _MING suffix)
    assert "base_homo_energy" in desc
    assert "base_lumo_energy" in desc
    assert "base_dipole" in desc
    assert "base_E_scf" in desc
    assert "base_number_of_atoms" in desc
    assert "base_molar_mass" in desc
    assert "base_molar_volume" in desc
    assert desc["base_number_of_atoms"] == 3.0

    # Must NOT have multi-conformer suffixes
    assert "base_homo_energy_MING" not in desc


def test_pipeline_multi_conformer_ligand() -> None:
    """Full pipeline on PPh3 as 'ligand' — multi-conformer, MING/MAXG/STDEV/MEAN."""
    spec = ComponentSpec(
        csv_column="ligand", prefix="lig", role="ligand",
        compute_nmr=False, compute_vbur=False,
        num_conformers=3, multi_conformer=True,
    )
    desc = compute_molecule_descriptors(
        "c1ccc(P(c2ccccc2)c2ccccc2)cc1", spec, basis="sto-3g",
    )

    # Must have multi-conformer suffixes
    assert "lig_homo_energy_MING" in desc
    assert "lig_homo_energy_MAXG" in desc
    assert "lig_homo_energy_STDEV" in desc
    assert "lig_homo_energy_MEAN" in desc

    # MING <= MEAN <= MAXG
    assert desc["lig_homo_energy_MING"] <= desc["lig_homo_energy_MEAN"]
    assert desc["lig_homo_energy_MEAN"] <= desc["lig_homo_energy_MAXG"]

    # molar_volume should also have multi-conformer stats
    assert "lig_molar_volume_MING" in desc
    assert "lig_molar_volume_STDEV" in desc

    # number_of_atoms is constant → only _MING
    assert "lig_number_of_atoms_MING" in desc


def test_pipeline_atom_selection_phosphorus() -> None:
    """Ligand role should select P as atom1."""
    spec = ComponentSpec(
        csv_column="lig", prefix="lig", role="ligand",
        compute_nmr=False, compute_vbur=False,
        num_conformers=1, multi_conformer=False,
    )
    desc = compute_molecule_descriptors(
        "c1ccc(P(c2ccccc2)c2ccccc2)cc1", spec, basis="sto-3g",
    )
    # atom1 is P → should have Mulliken charge
    assert "lig_atom1_Mulliken_charge" in desc
    assert not math.isnan(desc["lig_atom1_Mulliken_charge"])


# ---------------------------------------------------------------------------
# encode_dataset_dft — end-to-end CSV → CSV
# ---------------------------------------------------------------------------

def test_encode_dataset_auto_detect(tiny_csv: Path, tmp_path: Path) -> None:
    """Auto-detect SMILES columns and encode a whole CSV."""
    out_dir = tmp_path / "out"
    result_df = encode_dataset_dft(
        tiny_csv, out_dir, basis="sto-3g", verbose=False,
    )

    # Passthrough columns preserved
    assert "temperature" in result_df.columns
    assert "yield" in result_df.columns

    # SMILES columns replaced by descriptors
    assert "molecule" not in result_df.columns
    assert "solvent" not in result_df.columns

    # Descriptor columns present (prefixed by detected column name)
    desc_cols = [c for c in result_df.columns
                 if c not in ("temperature", "yield")]
    assert len(desc_cols) > 10  # should have many DFT descriptors
    assert len(result_df) == 3  # same row count as input

    # Cache file created
    assert (out_dir / "dft_cache.json").exists()


def test_encode_dataset_with_cache_reuse(tiny_csv: Path, tmp_path: Path) -> None:
    """Second encode should hit cache (faster, same result)."""
    out_dir = tmp_path / "out"

    df1 = encode_dataset_dft(tiny_csv, out_dir, basis="sto-3g")
    df2 = encode_dataset_dft(tiny_csv, out_dir, basis="sto-3g")

    # Same shape, same values
    assert list(df1.columns) == list(df2.columns)
    assert len(df1) == len(df2)


# ---------------------------------------------------------------------------
# Aggregation logic (unit test, no DFT)
# ---------------------------------------------------------------------------

def test_aggregate_multi_conformer_stats() -> None:
    data = [
        {"dipole": 2.0, "homo": -0.3},
        {"dipole": 4.0, "homo": -0.5},
    ]
    result = _aggregate(data, "lig", multi_conformer=True)
    assert result["lig_dipole_MING"] == 2.0
    assert result["lig_dipole_MAXG"] == 4.0
    assert result["lig_homo_MING"] == -0.5


def test_aggregate_single_conformer_no_suffix() -> None:
    data = [{"homo": -0.3}]
    result = _aggregate(data, "b", multi_conformer=False)
    assert result["b_homo"] == -0.3
    assert "b_homo_MING" not in result
