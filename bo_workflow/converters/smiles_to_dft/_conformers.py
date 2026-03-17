"""RDKit-based 3D conformer generation and force-field pre-optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ConformerError(Exception):
    """Raised when conformer generation fails entirely."""


@dataclass(frozen=True)
class Conformer:
    """A single optimized 3D conformer."""

    coords: np.ndarray  # shape (n_atoms, 3), Angstroms
    energy: float  # MMFF energy (kcal/mol), NaN if optimization skipped
    atom_symbols: list[str]  # element symbols in atom order
    atom_numbers: list[int]  # atomic numbers in atom order
    rdkit_mol: object = None  # RDKit Mol with this conformer's 3D coords


def generate_conformers(
    smiles: str,
    *,
    num_conformers: int = 10,
    max_attempts: int = 50,
    random_seed: int = 42,
    prune_rms_thresh: float = 0.5,
) -> list[Conformer]:
    """Generate and MMFF-optimize conformers from a SMILES string.

    Pipeline:
      1. ``Chem.MolFromSmiles`` → ``Chem.AddHs``
      2. ``AllChem.EmbedMultipleConfs`` (ETKDG v3, deterministic seed)
      3. ``AllChem.MMFFOptimizeMoleculeConfs`` (MMFF94)
      4. Sort by energy ascending, return :class:`Conformer` list

    Rigid molecules (e.g. small bases) naturally produce fewer conformers;
    the caller should handle 1-conformer results (STDEV = 0).

    Raises
    ------
    ConformerError
        If the SMILES cannot be parsed or 3D embedding fails entirely.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors3D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ConformerError(f"RDKit cannot parse SMILES: {smiles}")

    mol = Chem.AddHs(mol)

    # ETKDG v3 parameters
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        params.pruneRmsThresh = prune_rms_thresh
        params.numThreads = 1
        n_confs = AllChem.EmbedMultipleConfs(mol, num_conformers, params)
    except (TypeError, AttributeError):
        # Fallback for older/newer RDKit API
        n_confs = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=num_conformers,
            maxAttempts=max_attempts,
            randomSeed=random_seed,
            pruneRmsThresh=prune_rms_thresh,
        )

    if n_confs == 0:
        # Fallback: try single embed with random coordinates
        try:
            params2 = AllChem.ETKDGv3()
            params2.randomSeed = random_seed
            params2.useRandomCoords = True
            n_confs = AllChem.EmbedMultipleConfs(mol, 1, params2)
        except (TypeError, AttributeError):
            n_confs = AllChem.EmbedMultipleConfs(
                mol, numConfs=1, randomSeed=random_seed, useRandomCoords=True,
            )
        if n_confs == 0:
            raise ConformerError(
                f"Cannot generate 3D coordinates for: {smiles}"
            )

    # MMFF94 optimization
    try:
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
    except Exception:
        # If MMFF fails, keep unoptimized coordinates
        results = [(1, float("nan"))] * mol.GetNumConformers()

    # Extract conformers
    atom_symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    atom_numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]

    conformers: list[Conformer] = []
    for conf_id in range(mol.GetNumConformers()):
        conf = mol.GetConformer(conf_id)
        coords = np.array(conf.GetPositions(), dtype=np.float64)
        # results[conf_id] = (converged, energy)
        _converged, energy = results[conf_id]
        if np.isnan(energy):
            energy = float("inf")

        # Build a single-conformer RDKit mol for per-conformer volume etc.
        mol_copy = Chem.RWMol(mol)
        mol_copy.RemoveAllConformers()
        new_conf = Chem.Conformer(mol.GetNumAtoms())
        for ai in range(mol.GetNumAtoms()):
            new_conf.SetAtomPosition(ai, conf.GetAtomPosition(ai))
        mol_copy.AddConformer(new_conf, assignId=True)

        conformers.append(
            Conformer(
                coords=coords,
                energy=energy,
                atom_symbols=list(atom_symbols),
                atom_numbers=list(atom_numbers),
                rdkit_mol=mol_copy.GetMol(),
            )
        )

    # Sort by energy (best first)
    conformers.sort(key=lambda c: c.energy)
    return conformers


def mol_from_smiles(smiles: str):
    """Parse SMILES and return an RDKit Mol (with Hs, no 3D)."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ConformerError(f"RDKit cannot parse SMILES: {smiles}")
    return mol


def mol_properties(smiles: str) -> dict[str, float]:
    """Compute basic RDKit molecular properties (no DFT needed).

    Returns dict with: number_of_atoms, molar_mass, molar_volume.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "number_of_atoms": float("nan"),
            "molar_mass": float("nan"),
            "molar_volume": float("nan"),
        }

    mol_h = Chem.AddHs(mol)
    # Count heavy atoms only (matches Gaussian convention in BH_synthesis_data)
    n_atoms = mol.GetNumHeavyAtoms()
    molar_mass = Descriptors.MolWt(mol_h)

    # Molar volume approximation via AllChem
    try:
        AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol_h)
        vol = AllChem.ComputeMolVolume(mol_h)
    except Exception:
        vol = float("nan")

    return {
        "number_of_atoms": float(n_atoms),
        "molar_mass": float(molar_mass),
        "molar_volume": float(vol),
    }
