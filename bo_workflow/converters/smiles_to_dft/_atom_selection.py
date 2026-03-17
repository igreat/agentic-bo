"""Atom selection for per-atom descriptor extraction.

Selects the "atoms of interest" for ligands, bases, and additives:

- **atom1-4**: For phosphine ligands, atom1 = P, atom2-4 = atoms directly
  bonded to P (sorted by atomic number descending). For other molecules,
  atom1 = primary heteroatom (N > P > O > S by priority), atom2-4 = bonded
  neighbors.

- **c_min / c_min+1 / c_max / c_max-1**: Carbon atoms ranked by Mulliken
  charge. c_min = most negative charge, c_max = most positive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AtomSelection:
    """Indices and symbols for atoms of interest (up to 4)."""

    indices: list[int]  # 0-based atom indices, length 1-4
    symbols: list[str]  # element symbols at those indices


# Priority order for heteroatom selection (higher = preferred)
_HETEROATOM_PRIORITY = {"P": 4, "N": 3, "O": 2, "S": 1}


def select_atoms(
    mol,  # rdkit.Chem.Mol (with Hs)
    role: str = "ligand",
) -> AtomSelection:
    """Select up to 4 atoms of interest from a molecule.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule with explicit hydrogens.
    role : str
        Component role: "ligand", "base", "additive", "aryl_halide".

    Returns
    -------
    AtomSelection
        Up to 4 atom indices and their element symbols.
    """
    from rdkit import Chem

    if role == "ligand":
        return _select_ligand_atoms(mol)
    elif role == "aryl_halide":
        return _select_halide_atoms(mol)
    else:
        # base / additive / solvent — use primary heteroatom
        return _select_heteroatom_center(mol)


def _select_ligand_atoms(mol) -> AtomSelection:
    """For phosphine ligands: atom1=P, atom2-4=P-bonded heavy atoms."""
    from rdkit import Chem

    # Find phosphorus atom(s)
    p_atoms = [a for a in mol.GetAtoms() if a.GetSymbol() == "P"]
    if p_atoms:
        center = p_atoms[0]  # use first P
    else:
        # Fallback to generic heteroatom selection
        return _select_heteroatom_center(mol)

    center_idx = center.GetIdx()
    # Get heavy-atom neighbors of P, sorted by atomic number descending
    neighbors = []
    for nbr in center.GetNeighbors():
        if nbr.GetAtomicNum() > 1:  # skip H
            neighbors.append(nbr)
    neighbors.sort(key=lambda a: a.GetAtomicNum(), reverse=True)

    indices = [center_idx]
    symbols = [center.GetSymbol()]
    for nbr in neighbors[:3]:  # up to 3 more
        indices.append(nbr.GetIdx())
        symbols.append(nbr.GetSymbol())

    return AtomSelection(indices=indices, symbols=symbols)


def _select_halide_atoms(mol) -> AtomSelection:
    """For aryl halides: atom1 = halogen (Cl/Br/I)."""
    halogens = {"Cl", "Br", "I", "F"}
    halide_atoms = [a for a in mol.GetAtoms() if a.GetSymbol() in halogens]
    if not halide_atoms:
        return _select_heteroatom_center(mol)

    center = halide_atoms[0]
    center_idx = center.GetIdx()
    neighbors = [
        n for n in center.GetNeighbors() if n.GetAtomicNum() > 1
    ]
    neighbors.sort(key=lambda a: a.GetAtomicNum(), reverse=True)

    indices = [center_idx]
    symbols = [center.GetSymbol()]
    for nbr in neighbors[:3]:
        indices.append(nbr.GetIdx())
        symbols.append(nbr.GetSymbol())

    return AtomSelection(indices=indices, symbols=symbols)


def _select_heteroatom_center(mol) -> AtomSelection:
    """Select primary heteroatom by priority + connectivity."""
    best_atom = None
    best_score = -1

    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in _HETEROATOM_PRIORITY:
            priority = _HETEROATOM_PRIORITY[sym]
            connectivity = atom.GetDegree()
            score = priority * 100 + connectivity
            if score > best_score:
                best_score = score
                best_atom = atom

    if best_atom is None:
        # No heteroatom found — use most connected atom
        best_atom = max(mol.GetAtoms(), key=lambda a: a.GetDegree())

    center_idx = best_atom.GetIdx()
    neighbors = [
        n for n in best_atom.GetNeighbors() if n.GetAtomicNum() > 1
    ]
    neighbors.sort(key=lambda a: a.GetAtomicNum(), reverse=True)

    indices = [center_idx]
    symbols = [best_atom.GetSymbol()]
    for nbr in neighbors[:3]:
        indices.append(nbr.GetIdx())
        symbols.append(nbr.GetSymbol())

    return AtomSelection(indices=indices, symbols=symbols)


def select_charge_ranked_carbons(
    mulliken_charges: np.ndarray,
    atom_symbols: list[str],
) -> dict[str, int]:
    """Return atom indices for charge-ranked carbon positions.

    Parameters
    ----------
    mulliken_charges : array of shape (n_atoms,)
    atom_symbols : list of element symbols

    Returns
    -------
    dict with keys "c_min", "c_min+1", "c_max", "c_max-1" → 0-based indices.
    Missing keys if fewer than 4 carbon atoms.
    """
    # Collect (index, charge) for carbon atoms
    carbons = [
        (i, mulliken_charges[i])
        for i, sym in enumerate(atom_symbols)
        if sym == "C"
    ]
    if not carbons:
        return {}

    # Sort by charge ascending (most negative first)
    carbons.sort(key=lambda x: x[1])

    result: dict[str, int] = {}
    result["c_min"] = carbons[0][0]
    if len(carbons) >= 2:
        result["c_min+1"] = carbons[1][0]
    if len(carbons) >= 1:
        result["c_max"] = carbons[-1][0]
    if len(carbons) >= 2:
        result["c_max-1"] = carbons[-2][0]

    return result


def atom_type_flags(
    atom_idx: int,
    atom_symbols: list[str],
) -> dict[str, float]:
    """Return binary flags for the element type at atom_idx.

    Returns dict like {"atom=C": 1.0, "atom=N": 0.0, "atom=O": 0.0, "atom=P": 0.0}.
    """
    sym = atom_symbols[atom_idx] if atom_idx < len(atom_symbols) else "?"
    flags = {}
    for element in ("C", "N", "O", "P"):
        flags[f"atom={element}"] = 1.0 if sym == element else 0.0
    return flags
