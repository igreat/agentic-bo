"""Molecular descriptor extraction and molecule assembly.

This module converts molecular structures (SMILES) into numeric feature
vectors that the BO oracle can consume.  It also handles assembling a
complete molecule from a scaffold + substituent choices.

All public functions return plain Python dicts so they can be serialized
to JSON without special handling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
import numpy as np

from .types import (
    DescriptorConfig,
    MolecularDesignSpec,
    Scaffold,
    Substituent,
)

# ---------------------------------------------------------------------------
# Lookup tables for electronic / steric descriptors
# ---------------------------------------------------------------------------

# Hammett sigma-para constants for common substituents.
# Keys are canonical substituent *names* as they appear in SubstituentLibrary.
_HAMMETT_SIGMA: dict[str, float] = {
    "H": 0.00,
    "F": 0.06,
    "Cl": 0.23,
    "Br": 0.23,
    "I": 0.18,
    "methyl": -0.17,
    "CH3": -0.17,
    "ethyl": -0.15,
    "isopropyl": -0.15,
    "tert-butyl": -0.20,
    "phenyl": -0.01,
    "methoxy": -0.27,
    "OCH3": -0.27,
    "OH": -0.37,
    "NH2": -0.66,
    "amino": -0.66,
    "NO2": 0.78,
    "nitro": 0.78,
    "CN": 0.66,
    "cyano": 0.66,
    "CF3": 0.54,
    "trifluoromethyl": 0.54,
    "acetyl": 0.50,
    "COCH3": 0.50,
    "COOH": 0.45,
    "CHO": 0.42,
    "SO2CH3": 0.72,
    "vinyl": -0.04,
    "allyl": -0.07,
    "benzyl": -0.09,
    "SH": 0.15,
    "SCH3": 0.00,
}

# Taft steric parameters (Es) — more negative = bulkier.
_TAFT_ES: dict[str, float] = {
    "H": 1.24,
    "F": 0.78,
    "Cl": 0.27,
    "Br": 0.08,
    "I": -0.16,
    "methyl": 0.00,
    "CH3": 0.00,
    "ethyl": -0.07,
    "isopropyl": -0.47,
    "tert-butyl": -1.54,
    "phenyl": -2.55,
    "methoxy": 0.69,
    "OCH3": 0.69,
    "OH": 0.69,
    "NH2": 0.69,
    "amino": 0.69,
    "NO2": -1.01,
    "nitro": -1.01,
    "CN": -0.51,
    "cyano": -0.51,
    "CF3": -1.16,
    "trifluoromethyl": -1.16,
    "vinyl": -0.21,
    "allyl": -0.38,
    "benzyl": -0.38,
}

# A-values (kcal/mol) for conformational analysis.
_A_VALUES: dict[str, float] = {
    "H": 0.0,
    "F": 0.15,
    "Cl": 0.43,
    "Br": 0.38,
    "I": 0.43,
    "methyl": 1.74,
    "CH3": 1.74,
    "ethyl": 1.75,
    "isopropyl": 2.21,
    "tert-butyl": 4.7,
    "phenyl": 2.8,
    "methoxy": 0.6,
    "OCH3": 0.6,
    "OH": 0.52,
    "NH2": 1.4,
    "amino": 1.4,
    "NO2": 1.1,
    "nitro": 1.1,
    "CN": 0.17,
    "cyano": 0.17,
    "CF3": 2.5,
    "trifluoromethyl": 2.5,
    "vinyl": 1.35,
    "allyl": 1.35,
    "benzyl": 1.75,
}


def _try_load_external_table(path: str | Path | None) -> dict[str, float]:
    """Load a JSON lookup table from disk, returning {} on failure."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Descriptor computation
# ---------------------------------------------------------------------------


def compute_basic_descriptors(mol: Chem.Mol) -> dict[str, float]:
    """Compute basic RDKit molecular descriptors.

    Returns a dict with keys: MolWt, LogP, TPSA, NumHDonors, NumHAcceptors,
    NumRotatableBonds, RingCount, NumAromaticRings, NumHeavyAtoms,
    FractionCSP3, AromaticProportion.
    """
    n_heavy = mol.GetNumHeavyAtoms()
    n_aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    aromatic_proportion = n_aromatic / n_heavy if n_heavy > 0 else 0.0

    return {
        "MolWt": float(Descriptors.MolWt(mol)),
        "LogP": float(Descriptors.MolLogP(mol)),
        "TPSA": float(Descriptors.TPSA(mol)),
        "NumHDonors": float(rdMolDescriptors.CalcNumHBD(mol)),
        "NumHAcceptors": float(rdMolDescriptors.CalcNumHBA(mol)),
        "NumRotatableBonds": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "RingCount": float(rdMolDescriptors.CalcNumRings(mol)),
        "NumAromaticRings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "NumHeavyAtoms": float(n_heavy),
        "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "AromaticProportion": float(aromatic_proportion),
    }


def compute_fingerprint(
    mol: Chem.Mol,
    n_bits: int = 128,
    radius: int = 2,
) -> dict[str, float]:
    """Compute Morgan (ECFP-like) fingerprint as individual bit features.

    Returns a dict with keys ``fp_bit_0`` through ``fp_bit_{n_bits-1}``,
    each valued 0.0 or 1.0.
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=float)
    for idx in fp.GetOnBits():
        arr[idx] = 1.0
    return {f"fp_bit_{i}": arr[i] for i in range(n_bits)}


def compute_electronic_descriptors(
    mol: Chem.Mol,
    substituent_choices: dict[str, Substituent] | None = None,
) -> dict[str, float]:
    """Compute electronic descriptors.

    Includes Gasteiger partial charges (mean, std, min, max over heavy atoms)
    and, when *substituent_choices* is provided, Hammett sigma values for
    each substituent position.
    """
    result: dict[str, float] = {}

    # Gasteiger charges
    AllChem.ComputeGasteigerCharges(mol)
    charges = []
    for atom in mol.GetAtoms():
        gc = atom.GetDoubleProp("_GasteigerCharge")
        if np.isfinite(gc):
            charges.append(gc)
    if charges:
        arr = np.array(charges)
        result["gasteiger_mean"] = float(arr.mean())
        result["gasteiger_std"] = float(arr.std())
        result["gasteiger_min"] = float(arr.min())
        result["gasteiger_max"] = float(arr.max())
    else:
        result["gasteiger_mean"] = 0.0
        result["gasteiger_std"] = 0.0
        result["gasteiger_min"] = 0.0
        result["gasteiger_max"] = 0.0

    # Per-position Hammett sigma
    if substituent_choices:
        for pos_name, sub in substituent_choices.items():
            sigma = sub.properties.get(
                "hammett_sigma", _HAMMETT_SIGMA.get(sub.name, 0.0)
            )
            result[f"hammett_sigma_{pos_name}"] = float(sigma)

    return result


def compute_steric_descriptors(
    substituent_choices: dict[str, Substituent] | None = None,
) -> dict[str, float]:
    """Compute steric descriptors from substituent lookup tables.

    Returns Taft Es and A-values for each variable position.
    """
    result: dict[str, float] = {}
    if not substituent_choices:
        return result
    for pos_name, sub in substituent_choices.items():
        es = sub.properties.get("taft_Es", _TAFT_ES.get(sub.name, 0.0))
        av = sub.properties.get("a_value", _A_VALUES.get(sub.name, 0.0))
        result[f"taft_Es_{pos_name}"] = float(es)
        result[f"a_value_{pos_name}"] = float(av)
    return result


def compute_descriptors(
    smiles: str,
    config: DescriptorConfig | None = None,
    substituent_choices: dict[str, Substituent] | None = None,
) -> dict[str, float]:
    """Compute all configured descriptors for a molecule.

    Args:
        smiles: SMILES string of the complete molecule.
        config: Which descriptor groups to compute.  Defaults to all basic +
            fingerprint + electronic + steric.
        substituent_choices: Map from position name to Substituent.  Needed for
            Hammett sigma and Taft/A-value lookups.

    Returns:
        Flat dict of descriptor name → float value.

    Raises:
        ValueError: If the SMILES cannot be parsed by RDKit.
    """
    if config is None:
        config = DescriptorConfig()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles}")

    features: dict[str, float] = {}

    if config.basic:
        features.update(compute_basic_descriptors(mol))

    if config.fingerprint_enabled:
        features.update(
            compute_fingerprint(
                mol,
                n_bits=config.fingerprint_n_bits,
                radius=config.fingerprint_radius,
            )
        )

    if config.electronic:
        features.update(
            compute_electronic_descriptors(mol, substituent_choices)
        )

    if config.steric:
        features.update(compute_steric_descriptors(substituent_choices))

    return features


# ---------------------------------------------------------------------------
# Molecule assembly
# ---------------------------------------------------------------------------


def assemble_molecule(
    scaffold: Scaffold,
    choices: dict[str, Substituent],
) -> str:
    """Combine a scaffold with substituent choices into a complete SMILES.

    The scaffold SMILES must use ``[*:N]`` dummy atoms to mark variable
    positions.  The mapping from position name to isotope label N is
    positional: the first entry in ``scaffold.variable_positions`` maps
    to ``[*:1]``, the second to ``[*:2]``, etc.

    Substituent SMILES conventions:

    * ``[H]`` — hydrogen (dummy is simply removed)
    * Single atoms: ``[F]``, ``[Cl]``, ``[Br]``, ``[I]`` — directly bonded
    * Fragments with ``[*]`` marker: the dummy atom in the substituent is
      the attachment point  (e.g. ``[*]OC`` for methoxy)
    * Fragments without marker: the **first atom** is the attachment point
      (e.g. ``OC`` for methoxy, ``C`` for methyl, ``CC`` for ethyl)

    Args:
        scaffold: The molecular scaffold specification.
        choices: Map from position name to the chosen Substituent.

    Returns:
        Canonical SMILES of the assembled molecule.

    Raises:
        ValueError: If the scaffold or any substituent SMILES is invalid,
            or if a required position is missing from *choices*.
    """
    for pos_name in scaffold.variable_positions:
        if pos_name not in choices:
            raise ValueError(
                f"Missing substituent choice for position '{pos_name}'."
            )

    # Strategy: replace each [*:N] with the substituent's attachment-point
    # SMILES using RDKit's ReplaceSubstructs or a manual RWMol approach.
    #
    # We use a straightforward method:
    # 1. Parse scaffold.
    # 2. For each dummy atom, find its label and neighbor.
    # 3. Remove the dummy, attach the substituent's first heavy atom
    #    (or the atom connected to the substituent's own dummy) to the
    #    scaffold neighbor.

    scaffold_mol = Chem.MolFromSmiles(scaffold.smiles)
    if scaffold_mol is None:
        raise ValueError(f"Cannot parse scaffold SMILES: {scaffold.smiles}")

    rw_mol = Chem.RWMol(scaffold_mol)

    # Collect dummy atom info: (dummy_idx, neighbor_idx, label)
    # Process in reverse index order to avoid index-shift issues.
    dummy_info: list[tuple[int, int, int]] = []
    for atom in rw_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            label = atom.GetIsotope()
            if label == 0:
                label = atom.GetAtomMapNum()
            neighbors = atom.GetNeighbors()
            if neighbors and label > 0:
                dummy_info.append((atom.GetIdx(), neighbors[0].GetIdx(), label))

    # Map label -> position name
    label_to_pos: dict[int, str] = {}
    for idx, pos_name in enumerate(scaffold.variable_positions, start=1):
        label_to_pos[idx] = pos_name

    # Sort dummies by index descending for safe removal
    dummy_info.sort(key=lambda x: x[0], reverse=True)

    for dummy_idx, neighbor_idx, label in dummy_info:
        pos_name = label_to_pos.get(label)
        if pos_name is None:
            continue
        sub = choices[pos_name]
        sub_smiles = sub.smiles

        # Special case: [H] means just remove the dummy
        sub_mol = Chem.MolFromSmiles(sub_smiles)
        if sub_mol is None:
            raise ValueError(
                f"Cannot parse substituent SMILES '{sub_smiles}' "
                f"for position '{pos_name}'."
            )

        if sub_mol.GetNumHeavyAtoms() == 0:
            # Hydrogen — just remove the dummy atom
            rw_mol.RemoveAtom(dummy_idx)
            continue

        # Determine the attachment atom index in the substituent
        sub_has_dummy = False
        sub_dummy_idx = -1
        sub_attach_idx = -1
        for a in sub_mol.GetAtoms():
            if a.GetAtomicNum() == 0:
                sub_has_dummy = True
                sub_dummy_idx = a.GetIdx()
                sub_neighbors = a.GetNeighbors()
                if sub_neighbors:
                    sub_attach_idx = sub_neighbors[0].GetIdx()
                break

        if not sub_has_dummy:
            # No dummy in substituent: first atom is attachment point
            sub_attach_idx = 0

        # Combine molecules
        combo = Chem.CombineMols(rw_mol, sub_mol)
        combo_rw = Chem.RWMol(combo)

        n_scaffold = rw_mol.GetNumAtoms()
        # Actual indices in combo
        combo_attach = n_scaffold + sub_attach_idx

        # Add bond: scaffold neighbor <-> substituent attachment
        combo_rw.AddBond(neighbor_idx, combo_attach, Chem.BondType.SINGLE)

        # Collect atoms to remove (dummy in scaffold + dummy in substituent if any)
        to_remove = [dummy_idx]
        if sub_has_dummy and sub_dummy_idx >= 0:
            to_remove.append(n_scaffold + sub_dummy_idx)

        # Remove atoms in descending order
        for ri in sorted(to_remove, reverse=True):
            combo_rw.RemoveAtom(ri)

        rw_mol = combo_rw

    try:
        Chem.SanitizeMol(rw_mol)
        result = Chem.MolToSmiles(rw_mol)
    except Exception as exc:
        raise ValueError(
            f"Failed to sanitize assembled molecule: {exc}"
        ) from exc

    return result


# ---------------------------------------------------------------------------
# Feature expansion for the oracle
# ---------------------------------------------------------------------------


def get_descriptor_feature_names(
    spec: MolecularDesignSpec,
) -> list[str]:
    """Return the ordered list of descriptor feature names that
    ``expand_features_for_oracle`` will produce for the given spec.

    Useful for setting up the oracle training pipeline.
    """
    config = spec.descriptor_config
    names: list[str] = []

    if config.basic:
        names.extend([
            "MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors",
            "NumRotatableBonds", "RingCount", "NumAromaticRings",
            "NumHeavyAtoms", "FractionCSP3", "AromaticProportion",
        ])

    if config.fingerprint_enabled:
        names.extend([f"fp_bit_{i}" for i in range(config.fingerprint_n_bits)])

    if config.electronic:
        names.extend([
            "gasteiger_mean", "gasteiger_std", "gasteiger_min", "gasteiger_max",
        ])
        for pos_name in spec.scaffold.variable_positions:
            names.append(f"hammett_sigma_{pos_name}")

    if config.steric:
        for pos_name in spec.scaffold.variable_positions:
            names.append(f"taft_Es_{pos_name}")
            names.append(f"a_value_{pos_name}")

    # Energy features (DFT / xTB / ML)
    if config.dft_enabled:
        names.extend([
            "dft_total_energy_hartree",
            "dft_homo_ev",
            "dft_lumo_ev",
            "dft_gap_ev",
            "dft_dipole_debye",
            "dft_strain_energy_kcal",
            "dft_delta_energy",
        ])

    return names


def expand_features_for_oracle(
    design_row: dict[str, Any],
    spec: MolecularDesignSpec,
    dft_cache: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Convert a HEBO suggestion row into the full oracle feature vector.

    This is the key translation function.  Given a design row like
    ``{"R1": "F", "R2": "methyl", "temperature_C": 80, "solvent": "DMF"}``,
    it:
    1. Looks up Substituent objects from the spec libraries.
    2. Assembles the full molecule SMILES.
    3. Computes all configured descriptors.
    4. Optionally merges DFT features from the cache.

    Args:
        design_row: A single HEBO suggestion as a dict.  Keys include
            position names (categorical choices) and possibly condition
            parameters (which are ignored here).
        spec: The molecular design specification.
        dft_cache: Optional map from canonical SMILES to DFT feature dicts.

    Returns:
        Flat dict of descriptor feature name → float value.
    """
    # 1. Look up substituents
    choices: dict[str, Substituent] = {}
    for pos_name, lib in spec.libraries.items():
        sub_name = design_row.get(pos_name)
        if sub_name is None:
            raise ValueError(
                f"Design row missing position '{pos_name}'. "
                f"Row keys: {list(design_row.keys())}"
            )
        choices[pos_name] = lib.by_name(str(sub_name))

    # 2. Assemble molecule
    full_smiles = assemble_molecule(spec.scaffold, choices)

    # 3. Compute descriptors
    features = compute_descriptors(
        full_smiles,
        config=spec.descriptor_config,
        substituent_choices=choices,
    )

    # 4. Merge DFT/energy features if available
    if dft_cache and spec.descriptor_config.dft_enabled:
        # Use canonical isomeric SMILES for robust cache lookup
        mol_obj = Chem.MolFromSmiles(full_smiles)
        if mol_obj is not None:
            canon = Chem.MolToSmiles(mol_obj, isomericSmiles=True, canonical=True)
        else:
            canon = full_smiles
        dft_feats = dft_cache.get(canon, dft_cache.get(full_smiles, {}))
        for k, v in dft_feats.items():
            features[f"dft_{k}"] = float(v)

        # Compute delta_energy: E_product - E_scaffold (thermodynamic driving force)
        # The scaffold energy is stored under the scaffold's own SMILES key
        scaffold_smiles = spec.scaffold.smiles
        scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
        if scaffold_mol is not None:
            scaffold_canon = Chem.MolToSmiles(
                scaffold_mol, isomericSmiles=True, canonical=True
            )
            scaffold_feats = dft_cache.get(
                scaffold_canon, dft_cache.get(scaffold_smiles, {})
            )
            product_energy = dft_feats.get("total_energy_hartree")
            scaffold_energy = scaffold_feats.get("total_energy_hartree")
            if product_energy is not None and scaffold_energy is not None:
                import math

                pe = float(product_energy)
                se = float(scaffold_energy)
                if not (math.isnan(pe) or math.isnan(se)):
                    features["dft_delta_energy"] = pe - se

        # Ensure delta_energy key exists even if we couldn't compute it
        if "dft_delta_energy" not in features:
            features["dft_delta_energy"] = float("nan")

    return features
