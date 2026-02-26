"""Multi-tier energy computation for molecular Bayesian Optimization.

Provides physics-based energy features — total energy, HOMO/LUMO, dipole,
strain energy — as oracle descriptors and feasibility corrections.

Architecture (4-tier cascade):
  Tier 1: Pre-computed JSON cache (0 ms)
  Tier 2: GFN2-xTB via tblite native interface (~0.5–2 s)
  Tier 3: TorchANI ANI-2x ML potential on GPU (<1 ms)
  Tier 4: PySCF B3LYP/6-31G* DFT (~4 min, offline batch only)

Energy features produced (6 values per molecule):
  total_energy_hartree, homo_ev, lumo_ev, gap_ev, dipole_debye,
  strain_energy_kcal
"""

from __future__ import annotations

import json
import logging
import math
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prevent OpenMP thread contention when multiple xTB jobs run concurrently.
# Must be set *before* importing tblite / numpy-linked BLAS.
# ---------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENERGY_FEATURE_KEYS: list[str] = [
    "total_energy_hartree",
    "homo_ev",
    "lumo_ev",
    "gap_ev",
    "dipole_debye",
    "strain_energy_kcal",
]

_HARTREE_TO_KCAL = 627.509474
_HARTREE_TO_EV = 27.211386245988
_BOHR_TO_ANGSTROM = 0.529177249
_ANGSTROM_TO_BOHR = 1.0 / _BOHR_TO_ANGSTROM

# Sentinel value for hopelessly distorted molecules.
_STRAIN_SENTINEL = 9999.0

# MMFF energy threshold — above this or NaN means the molecule is too
# distorted for reliable 3-D embedding.
_MMFF_ENERGY_CUTOFF = 1e6

# ---------------------------------------------------------------------------
# Bond-order corrections to base ring strain (kcal/mol, additive).
# Base strain (3-ring: +27, 4-ring: +26) assumes all-sp3 saturated rings.
# Higher bond orders dramatically increase strain because the ideal bond
# angle diverges further from the ring-imposed angle.
#
# References (experimental):
#   cyclopropene (C1=CC1): ~55 kcal/mol total → correction ~+28
#   cyclopropyne (C1#CC1): physically impossible → correction +120
#   cyclobutene  (C1=CCC1): ~30 kcal/mol total → correction ~+7
#   cyclobutyne  (C1#CCC1): effectively impossible → correction +80
# ---------------------------------------------------------------------------
_BOND_ORDER_RING_CORRECTIONS: dict[tuple[int, float], float] = {
    # (ring_size, bond_order): additional_strain_kcal
    (3, 2.0): 28.0,   # double bond in 3-ring (cyclopropene)
    (3, 3.0): 120.0,  # triple bond in 3-ring (cyclopropyne — impossible)
    (4, 2.0): 7.0,    # double bond in 4-ring (cyclobutene)
    (4, 3.0): 80.0,   # triple bond in 4-ring (cyclobutyne — impossible)
    (5, 3.0): 15.0,   # triple bond in 5-ring (cyclopentyne — highly strained)
}

# Allene (cumulated C=C=C) in ring: central sp atom requires 180°,
# impossible in any ring geometry.
_ALLENE_IN_RING_PENALTY = 100.0  # kcal/mol

# Antiaromaticity destabilization for fully conjugated 4n π-electron rings.
_ANTIAROMATICITY_PENALTY = 45.0  # kcal/mol (cyclobutadiene ~45 kcal/mol)

# Anti-Bredt penalty: double/triple bond at bridgehead of bicyclic system.
_ANTI_BREDT_PENALTY_SMALL = 50.0   # bridgehead unsaturation, smallest ring ≤ 6
_ANTI_BREDT_PENALTY_MEDIUM = 25.0  # bridgehead unsaturation, smallest ring 7–8


# ---------------------------------------------------------------------------
# Canonical SMILES helpers
# ---------------------------------------------------------------------------


def _canonical_smiles(smiles: str) -> str:
    """Return the canonical isomeric SMILES for cache key consistency."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


# ═══════════════════════════════════════════════════════════════════════════
# EnergyCache
# ═══════════════════════════════════════════════════════════════════════════


class EnergyCache:
    """JSON-backed, canonical-SMILES-keyed energy feature cache.

    The JSON file contains a ``_meta`` key with method/version information
    and one key per canonical isomeric SMILES mapping to a dict of energy
    features.

    Parameters
    ----------
    path : Path | None
        Path to the JSON cache file.  If *None*, cache is in-memory only.
    method : str
        Energy method label (e.g. ``"GFN2-xTB"``, ``"B3LYP/6-31G*"``).
        Stored in ``_meta``.  On load, a mismatch triggers a warning but
        does **not** invalidate the cache (different-level data is still
        useful).
    """

    def __init__(self, path: Path | None = None, method: str = "unknown") -> None:
        self._path = path
        self._method = method
        self._data: dict[str, dict[str, float]] = {}
        self._meta: dict[str, Any] = {"method": method, "version": "1"}

        if path is not None and path.exists():
            self._load(path)

    # -- persistence --------------------------------------------------------

    def _load(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        stored_meta = raw.pop("_meta", {})
        if stored_meta.get("method", self._method) != self._method:
            logger.warning(
                "Energy cache method mismatch: file has '%s', current is '%s'. "
                "Data will still be used but may mix accuracy levels.",
                stored_meta.get("method"),
                self._method,
            )
        self._meta.update(stored_meta)
        self._data = raw

    def save(self) -> None:
        """Persist the cache to disk (no-op if path is *None*)."""
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"_meta": self._meta, **self._data}
        with self._path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

    # -- dict-like interface ------------------------------------------------

    def __contains__(self, smiles: str) -> bool:
        key = _canonical_smiles(smiles)
        return key in self._data

    def get(self, smiles: str) -> dict[str, float] | None:
        """Look up cached energy features.  Returns *None* on miss."""
        try:
            key = _canonical_smiles(smiles)
        except ValueError:
            return None
        return self._data.get(key)

    def put(self, smiles: str, features: dict[str, float]) -> None:
        """Store energy features and write-through to disk."""
        key = _canonical_smiles(smiles)
        self._data[key] = features
        self.save()

    def as_dict(self) -> dict[str, dict[str, float]]:
        """Return a plain dict copy (for passing to expand_features)."""
        return dict(self._data)

    def merge(self, other: dict[str, dict[str, float]]) -> int:
        """Merge entries from *other* that are missing locally.

        Returns the number of new entries added.
        """
        added = 0
        for smiles, features in other.items():
            if smiles == "_meta":
                continue
            try:
                key = _canonical_smiles(smiles)
            except ValueError:
                continue
            if key not in self._data:
                self._data[key] = features
                added += 1
        return added

    def __len__(self) -> int:
        return len(self._data)


# ═══════════════════════════════════════════════════════════════════════════
# 3-D Coordinate Generation
# ═══════════════════════════════════════════════════════════════════════════


def _smiles_to_3d(
    smiles: str,
    *,
    num_conformers: int = 4,
    max_iters: int = 500,
) -> tuple[list[int], np.ndarray]:
    """Convert SMILES to optimized 3-D coordinates.

    Uses ETKDG multi-conformer generation + MMFF94 optimization, picking
    the lowest-energy conformer.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule.
    num_conformers : int
        Number of ETKDG conformers to generate.
    max_iters : int
        Maximum MMFF optimization iterations per conformer.

    Returns
    -------
    atomic_numbers : list[int]
        Atomic numbers for each atom (including hydrogens).
    positions : np.ndarray, shape (N, 3)
        Cartesian coordinates in **Angstrom**.

    Raises
    ------
    ValueError
        If SMILES cannot be parsed or all conformer attempts fail.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy > 80:
        logger.warning(
            "Molecule has %d heavy atoms (>80); 3-D embedding may be slow "
            "or unreliable: %s",
            n_heavy,
            smiles,
        )

    # --- Attempt ETKDG embedding ---
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1  # single-threaded for reproducibility

    cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)

    # Fallback: try with random coordinates if ETKDG fails
    if len(cids) == 0:
        logger.warning("ETKDG failed for %s; retrying with random coordinates.", smiles)
        params.useRandomCoords = True
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)

    if len(cids) == 0:
        raise ValueError(f"Cannot generate 3-D coordinates for: {smiles}")

    # --- MMFF optimize each conformer, pick lowest energy ---
    best_energy = float("inf")
    best_conf_id = cids[0]

    for cid in cids:
        try:
            result = AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=max_iters)
            # result: 0=converged, 1=not converged, -1=setup failed
        except Exception:
            continue

        try:
            ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=cid)
            if ff is None:
                continue
            energy = ff.CalcEnergy()
        except Exception:
            continue

        if math.isnan(energy) or energy > _MMFF_ENERGY_CUTOFF:
            # Molecule is too distorted; skip this conformer
            continue

        if energy < best_energy:
            best_energy = energy
            best_conf_id = cid

    # Check if ALL conformers had distorted energy
    if best_energy > _MMFF_ENERGY_CUTOFF or math.isinf(best_energy):
        logger.warning(
            "All MMFF conformer energies are distorted for %s "
            "(best=%.1f); strain will be set to sentinel.",
            smiles,
            best_energy if not math.isinf(best_energy) else float("nan"),
        )
        # Still return the first conformer's geometry for downstream use,
        # but callers should check for the sentinel strain value.

    conf = mol.GetConformer(best_conf_id)
    positions = np.array(conf.GetPositions(), dtype=np.float64)
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    return atomic_numbers, positions


# ═══════════════════════════════════════════════════════════════════════════
# SSSR-Based Strain Energy Heuristic
# ═══════════════════════════════════════════════════════════════════════════


def _bond_order_ring_correction(mol: Chem.Mol, ring: tuple[int, ...]) -> float:
    """Compute additional strain from unsaturated bonds within a strained ring.

    Examines each bond in the ring.  For double or triple bonds, looks up
    the additional strain penalty from ``_BOND_ORDER_RING_CORRECTIONS``.
    Also detects *allene-in-ring* motifs (cumulated diene, C=C=C inside the
    ring) where the central sp-hybridised atom requires 180° — impossible
    in any ring.

    Parameters
    ----------
    mol : Chem.Mol
        The parsed molecule.
    ring : tuple[int, ...]
        Atom indices forming the ring (from SSSR).

    Returns
    -------
    float
        Additional strain in kcal/mol from bond-order effects in this ring.
    """
    ring_size = len(ring)
    ring_set = set(ring)
    correction = 0.0

    # Scan bonds in the ring for unsaturation
    for i in range(ring_size):
        a1 = ring[i]
        a2 = ring[(i + 1) % ring_size]
        bond = mol.GetBondBetweenAtoms(a1, a2)
        if bond is None:
            continue
        bond_order = bond.GetBondTypeAsDouble()

        if bond_order >= 2.0:
            key = (ring_size, bond_order)
            correction += _BOND_ORDER_RING_CORRECTIONS.get(key, 0.0)

    # Detect allene-in-ring: sp atom with two double bonds to ring members
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetHybridization() == Chem.HybridizationType.SP:
            double_bonds_in_ring = 0
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() in ring_set:
                    bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
                    if bond and bond.GetBondTypeAsDouble() == 2.0:
                        double_bonds_in_ring += 1
            if double_bonds_in_ring >= 2:
                correction += _ALLENE_IN_RING_PENALTY

    return correction


def _antiaromaticity_penalty(mol: Chem.Mol, ring: tuple[int, ...]) -> float:
    """Detect antiaromatic destabilisation in a fully conjugated ring.

    A ring is considered antiaromatic when:

    1. It is **not** already flagged as aromatic by RDKit's kekulisation.
    2. All ring atoms are sp2-hybridised (fully conjugated).
    3. The ring contains 4 *n* π electrons (Hückel antiaromaticity rule).

    Heteroatoms with an H (pyrrole-type) donate 2 π electrons; all other
    atoms donate 1.

    Parameters
    ----------
    mol : Chem.Mol
        The parsed molecule.
    ring : tuple[int, ...]
        Atom indices forming the ring.

    Returns
    -------
    float
        Antiaromaticity penalty in kcal/mol (0 if not antiaromatic).
    """
    # Skip rings already recognised as aromatic by RDKit
    if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
        return 0.0

    # All ring atoms must be sp2 (or aromatic — already guarded above)
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        hyb = atom.GetHybridization()
        if hyb != Chem.HybridizationType.SP2 and not atom.GetIsAromatic():
            return 0.0  # not fully conjugated

    # Count π electrons using Hückel accounting
    pi_electrons = 0
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        # Pyrrole-type heteroatom: lone pair donates 2 π electrons
        if atom.GetSymbol() in ("N", "O", "S") and atom.GetTotalNumHs() > 0:
            pi_electrons += 2
        else:
            pi_electrons += 1

    # 4n rule: antiaromatic if π electrons is a positive multiple of 4
    if pi_electrons > 0 and pi_electrons % 4 == 0:
        return _ANTIAROMATICITY_PENALTY

    return 0.0


def _anti_bredt_strain(mol: Chem.Mol) -> float:
    """Estimate strain from anti-Bredt double/triple bonds at bridgeheads.

    Bridgehead atoms (atoms belonging to 2+ SSSR rings) cannot easily
    accommodate double or triple bonds because the bridged geometry
    prevents the necessary orbital alignment.  The penalty magnitude
    depends on the smallest ring containing the bridgehead — smaller rings
    make the violation more severe.

    Parameters
    ----------
    mol : Chem.Mol
        The parsed molecule.

    Returns
    -------
    float
        Anti-Bredt strain penalty in kcal/mol (0 if no violations).
    """
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    if len(atom_rings) < 2:
        return 0.0

    # Map each atom to the sizes of the rings it belongs to
    atom_to_ring_sizes: dict[int, list[int]] = {}
    for ring in atom_rings:
        for idx in ring:
            atom_to_ring_sizes.setdefault(idx, []).append(len(ring))

    bridgehead_atoms = {
        idx for idx, sizes in atom_to_ring_sizes.items() if len(sizes) >= 2
    }

    if not bridgehead_atoms:
        return 0.0

    penalty = 0.0
    seen_bonds: set[tuple[int, int]] = set()

    for idx in bridgehead_atoms:
        atom = mol.GetAtomWithIdx(idx)
        for bond in atom.GetBonds():
            bond_order = bond.GetBondTypeAsDouble()
            if bond_order < 2.0:
                continue

            # Avoid double-counting the same bond
            bond_key = (
                min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            )
            if bond_key in seen_bonds:
                continue
            seen_bonds.add(bond_key)

            # Penalty depends on the smallest ring at this bridgehead
            smallest_ring = min(atom_to_ring_sizes[idx])
            if smallest_ring <= 6:
                penalty += _ANTI_BREDT_PENALTY_SMALL
            elif smallest_ring <= 8:
                penalty += _ANTI_BREDT_PENALTY_MEDIUM
            # Rings > 8: Bredt rule relaxes; no penalty

    return penalty


def _estimate_strain_energy(smiles: str) -> float:
    """Estimate ring strain energy from SSSR ring sizes and bond orders.

    Uses the Smallest Set of Smallest Rings (SSSR) from RDKit to identify
    strained small rings, then applies corrections for:

    * **Base ring strain**: 3-ring +27, 4-ring +26 kcal/mol (sp3 baseline).
    * **Bond-order correction**: double/triple bonds in small rings
      dramatically increase strain (sp2/sp ideal angles diverge from ring
      geometry).
    * **Allene-in-ring**: cumulated diene (C=C=C) inside a ring — the
      central sp atom requires 180°, impossible in any ring.
    * **Antiaromaticity**: fully conjugated rings with 4 *n* π electrons
      (Hückel rule) are thermodynamically destabilised.
    * **Anti-Bredt**: double/triple bonds at bridgehead atoms in bicyclic
      systems where orbital overlap is geometrically impossible.
    * **Shared-atom penalty**: +5 kcal/mol per atom shared between
      strained rings.

    Parameters
    ----------
    smiles : str
        SMILES string.

    Returns
    -------
    float
        Estimated strain energy in kcal/mol.  Returns 0.0 for molecules
        with no strained rings.  Returns ``_STRAIN_SENTINEL`` (9999) if
        the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _STRAIN_SENTINEL

    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    if not atom_rings:
        return 0.0

    strain = 0.0

    # --- Base ring-size strain + bond-order correction ---
    strained_ring_atoms: list[set[int]] = []
    for ring in atom_rings:
        ring_size = len(ring)
        base = 0.0
        if ring_size == 3:
            base = 27.0
        elif ring_size == 4:
            base = 26.0

        if base > 0.0:
            strain += base
            strain += _bond_order_ring_correction(mol, ring)
            strained_ring_atoms.append(set(ring))

        # Antiaromaticity: applies to any ring size
        strain += _antiaromaticity_penalty(mol, ring)

    # --- Shared-atom penalty: atoms in multiple strained rings ---
    if len(strained_ring_atoms) >= 2:
        from itertools import combinations

        shared_atoms: set[int] = set()
        for ring_a, ring_b in combinations(strained_ring_atoms, 2):
            shared_atoms |= ring_a & ring_b

        strain += len(shared_atoms) * 5.0

    # --- Anti-Bredt violations (bridgehead unsaturation) ---
    strain += _anti_bredt_strain(mol)

    return strain


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2: GFN2-xTB via tblite native interface
# ═══════════════════════════════════════════════════════════════════════════


def compute_xtb_energy(
    smiles: str,
    *,
    solvent: str | None = None,
) -> dict[str, float]:
    """Compute energy features using GFN2-xTB via tblite.

    Uses tblite's **native Python interface** (``tblite.interface.Calculator``)
    to access orbital energies for HOMO/LUMO extraction.  ASE is used only
    as a coordinate container for geometry optimization.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule.
    solvent : str | None
        Implicit solvent name for GBSA (e.g. ``"water"``, ``"thf"``).
        ``None`` means gas-phase computation.

    Returns
    -------
    dict with keys matching ``ENERGY_FEATURE_KEYS``.

    Raises
    ------
    ImportError
        If tblite is not installed.
    """
    try:
        from tblite.interface import Calculator
    except ImportError:
        raise ImportError(
            "tblite is not installed.  Install with: "
            "pip install 'tblite>=0.3.0'  "
            "(or use energy_backend='none' / 'ml')."
        )

    # --- 3-D coordinates ---
    try:
        atomic_numbers, positions_angstrom = _smiles_to_3d(smiles)
    except ValueError as exc:
        logger.warning("3-D embedding failed for %s: %s", smiles, exc)
        # Return heuristic strain + NaN energy (don't crash)
        strain = _estimate_strain_energy(smiles)
        return _nan_features(strain_energy=strain)

    numbers = np.array(atomic_numbers, dtype=np.int32)
    positions_bohr = positions_angstrom * _ANGSTROM_TO_BOHR

    # --- Geometry optimization via ASE + tblite ASE calculator ---
    try:
        from ase import Atoms
        from tblite.ase import TBLiteCalculator
        from ase.optimize import LBFGS

        atoms = Atoms(numbers=numbers, positions=positions_angstrom)
        calc = TBLiteCalculator(method="GFN2-xTB")
        if solvent is not None:
            # tblite ASE calculator accepts solvent keyword
            calc = TBLiteCalculator(method="GFN2-xTB", solvent=solvent)
        atoms.calc = calc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt = LBFGS(atoms, logfile=None)
            try:
                opt.run(fmax=0.05, steps=200)
            except Exception as exc:
                logger.warning(
                    "ASE LBFGS optimization failed for %s: %s. "
                    "Using initial geometry.",
                    smiles,
                    exc,
                )

        # Use optimized positions
        positions_bohr = atoms.get_positions() * _ANGSTROM_TO_BOHR

    except ImportError:
        logger.warning(
            "ASE not available; skipping geometry optimization for %s.",
            smiles,
        )
    except Exception as exc:
        logger.warning(
            "Geometry optimization error for %s: %s. Using initial geometry.",
            smiles,
            exc,
        )

    # --- Single-point with native tblite interface for orbital energies ---
    try:
        native_calc = Calculator("GFN2-xTB", numbers, positions_bohr)
        if solvent is not None:
            try:
                native_calc.set("solvent", solvent)
            except Exception:
                logger.warning(
                    "Could not set solvent '%s' in tblite native interface.",
                    solvent,
                )

        results = native_calc.singlepoint()
    except Exception as exc:
        logger.warning(
            "xTB single-point failed for %s: %s. Falling back to heuristic.",
            smiles,
            exc,
        )
        strain = _estimate_strain_energy(smiles)
        return _nan_features(strain_energy=strain)

    # --- Extract features ---
    total_energy = float(results.get("energy"))  # Hartree

    # HOMO/LUMO from orbital energies + occupations
    homo_ev = float("nan")
    lumo_ev = float("nan")
    gap_ev = float("nan")
    try:
        orb_energies = np.array(results.get("orbital-energies"))  # Hartree
        orb_occupations = np.array(results.get("orbital-occupations"))

        # HOMO: highest occupied, LUMO: lowest unoccupied
        occupied = orb_occupations > 0.5
        if occupied.any():
            homo_idx = np.where(occupied)[0][-1]
            homo_ev = float(orb_energies[homo_idx] * _HARTREE_TO_EV)

        unoccupied = orb_occupations < 0.5
        if unoccupied.any():
            lumo_idx = np.where(unoccupied)[0][0]
            lumo_ev = float(orb_energies[lumo_idx] * _HARTREE_TO_EV)

        if not math.isnan(homo_ev) and not math.isnan(lumo_ev):
            gap_ev = lumo_ev - homo_ev
    except Exception as exc:
        logger.warning("Orbital energy extraction failed for %s: %s", smiles, exc)

    # Dipole moment
    dipole_debye = float("nan")
    try:
        dipole = np.array(results.get("dipole"))
        # tblite returns dipole in e*Bohr; convert to Debye (1 D = 0.393456 e*a0)
        dipole_debye = float(np.linalg.norm(dipole) / 0.393456)
    except Exception as exc:
        logger.warning("Dipole extraction failed for %s: %s", smiles, exc)

    # Strain energy (heuristic from SSSR)
    strain = _estimate_strain_energy(smiles)

    return {
        "total_energy_hartree": total_energy,
        "homo_ev": homo_ev,
        "lumo_ev": lumo_ev,
        "gap_ev": gap_ev,
        "dipole_debye": dipole_debye,
        "strain_energy_kcal": strain,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2b: GFN2-xTB via command-line subprocess (fallback when tblite
#          Python bindings are unavailable but the `xtb` binary is installed)
# ═══════════════════════════════════════════════════════════════════════════


def _find_xtb_binary() -> str | None:
    """Locate the ``xtb`` command-line binary.

    Checks, in order:
    1. ``XTB_BINARY`` environment variable (explicit user override)
    2. Well-known install path ``~/xtb-dist/bin/xtb``
    3. ``shutil.which("xtb")`` (system PATH)

    Returns the absolute path, or ``None`` if not found.
    """
    import shutil

    # 1. Explicit env var
    env_path = os.environ.get("XTB_BINARY")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. Common local install
    home = Path.home()
    local_xtb = home / "xtb-dist" / "bin" / "xtb"
    if local_xtb.is_file():
        return str(local_xtb)

    # 3. System PATH
    which = shutil.which("xtb")
    if which is not None:
        return which

    return None


def _find_xtb_paramdir() -> str | None:
    """Locate the xTB parameter directory (XTBPATH).

    Checks:
    1. ``XTBPATH`` environment variable
    2. ``~/xtb-dist/share/xtb`` (paired with the binary)
    """
    env_path = os.environ.get("XTBPATH")
    if env_path and os.path.isdir(env_path):
        return env_path

    home = Path.home()
    local_param = home / "xtb-dist" / "share" / "xtb"
    if local_param.is_dir():
        return str(local_param)

    return None


def compute_xtb_cli_energy(
    smiles: str,
    *,
    solvent: str | None = None,
    optimize: bool = True,
) -> dict[str, float]:
    """Compute GFN2-xTB energy features via the ``xtb`` command-line binary.

    This is a fallback for when the ``tblite`` Python bindings cannot be
    installed (e.g. no Fortran compiler).  It provides identical features
    including HOMO/LUMO from the JSON output.

    Parameters
    ----------
    smiles : str
        SMILES string.
    solvent : str | None
        Implicit solvent for GBSA (e.g. ``"water"``).
    optimize : bool
        If ``True`` (default), run geometry optimization before single-point.

    Returns
    -------
    dict with keys matching ``ENERGY_FEATURE_KEYS``.

    Raises
    ------
    FileNotFoundError
        If the ``xtb`` binary is not found.
    """
    import subprocess
    import tempfile

    xtb_bin = _find_xtb_binary()
    if xtb_bin is None:
        raise FileNotFoundError(
            "xtb binary not found.  Install xtb from "
            "https://github.com/grimme-lab/xtb/releases "
            "and set XTB_BINARY or add to PATH."
        )

    # --- 3-D coordinates ---
    try:
        atomic_numbers, positions_angstrom = _smiles_to_3d(smiles)
    except ValueError as exc:
        logger.warning("3-D embedding failed for %s: %s", smiles, exc)
        strain = _estimate_strain_energy(smiles)
        return _nan_features(strain_energy=strain)

    # --- Write XYZ file ---
    n_atoms = len(atomic_numbers)

    # Map atomic numbers to element symbols
    from rdkit.Chem import PeriodicTable

    pt = Chem.GetPeriodicTable()
    lines = [f"{n_atoms}", ""]
    for z, pos in zip(atomic_numbers, positions_angstrom):
        sym = pt.GetElementSymbol(z)
        lines.append(f"{sym:2s}  {pos[0]:14.8f}  {pos[1]:14.8f}  {pos[2]:14.8f}")

    xyz_content = "\n".join(lines) + "\n"

    # --- Run xtb in temp directory ---
    with tempfile.TemporaryDirectory(prefix="xtb_") as tmpdir:
        xyz_path = os.path.join(tmpdir, "mol.xyz")
        with open(xyz_path, "w") as f:
            f.write(xyz_content)

        cmd = [xtb_bin, "mol.xyz", "--gfn", "2", "--json", "full"]
        if optimize:
            cmd.append("--opt")
        if solvent is not None:
            cmd.extend(["--gbsa", solvent])

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        # Set XTBPATH for parameter files
        paramdir = _find_xtb_paramdir()
        if paramdir:
            env["XTBPATH"] = paramdir
            env["XTBHOME"] = str(Path(paramdir).parent.parent)

        try:
            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minutes max per molecule
                env=env,
            )
        except subprocess.TimeoutExpired:
            logger.warning("xtb timed out for %s", smiles)
            strain = _estimate_strain_energy(smiles)
            return _nan_features(strain_energy=strain)
        except OSError as exc:
            logger.warning("Failed to run xtb for %s: %s", smiles, exc)
            strain = _estimate_strain_energy(smiles)
            return _nan_features(strain_energy=strain)

        if result.returncode != 0:
            logger.warning(
                "xtb returned exit code %d for %s: %s",
                result.returncode,
                smiles,
                result.stderr[:200] if result.stderr else "(no stderr)",
            )
            strain = _estimate_strain_energy(smiles)
            return _nan_features(strain_energy=strain)

        # --- Parse JSON output ---
        json_path = os.path.join(tmpdir, "xtbout.json")
        if not os.path.isfile(json_path):
            logger.warning("xtb did not produce JSON output for %s", smiles)
            strain = _estimate_strain_energy(smiles)
            return _nan_features(strain_energy=strain)

        with open(json_path) as f:
            data = json.load(f)

    # --- Extract features ---
    total_energy = float(data.get("total energy", float("nan")))

    # HOMO/LUMO from orbital energies (already in eV in the JSON)
    homo_ev = float("nan")
    lumo_ev = float("nan")
    gap_ev = float("nan")

    orb_energies_ev = data.get("orbital energies / eV", [])
    occupations = data.get("fractional occupation", [])

    if orb_energies_ev and occupations:
        orb_e = np.array(orb_energies_ev)
        occ = np.array(occupations)

        occupied = occ > 0.5
        if occupied.any():
            homo_idx = np.where(occupied)[0][-1]
            homo_ev = float(orb_e[homo_idx])

        unoccupied = occ < 0.5
        if unoccupied.any():
            lumo_idx = np.where(unoccupied)[0][0]
            lumo_ev = float(orb_e[lumo_idx])

        if not math.isnan(homo_ev) and not math.isnan(lumo_ev):
            gap_ev = lumo_ev - homo_ev

    # Validate against the directly reported gap
    reported_gap = data.get("HOMO-LUMO gap / eV")
    if reported_gap is not None and not math.isnan(gap_ev):
        if abs(gap_ev - reported_gap) > 0.01:
            logger.debug(
                "Computed gap (%.4f) differs from reported (%.4f) for %s",
                gap_ev,
                reported_gap,
                smiles,
            )

    # Dipole moment
    dipole_debye = float("nan")
    dipole_au = data.get("dipole / a.u.")
    if dipole_au is not None:
        # 1 a.u. of dipole = 2.5417463 Debye
        dipole_debye = float(np.linalg.norm(dipole_au) * 2.5417463)

    strain = _estimate_strain_energy(smiles)

    return {
        "total_energy_hartree": total_energy,
        "homo_ev": homo_ev,
        "lumo_ev": lumo_ev,
        "gap_ev": gap_ev,
        "dipole_debye": dipole_debye,
        "strain_energy_kcal": strain,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Tier 3: TorchANI ML Potential
# ═══════════════════════════════════════════════════════════════════════════


_TORCHANI_ELEMENTS = {"H", "C", "N", "O", "S", "F", "Cl"}


def compute_ml_energy(smiles: str) -> dict[str, float]:
    """Compute energy using TorchANI ANI-2x on GPU.

    ANI-2x supports: H, C, N, O, S, F, Cl only.  No orbital energies
    are available; HOMO/LUMO/gap/dipole are returned as NaN.

    Parameters
    ----------
    smiles : str
        SMILES string.

    Returns
    -------
    dict with keys matching ``ENERGY_FEATURE_KEYS``.

    Raises
    ------
    ImportError
        If torchani is not installed.
    ValueError
        If the molecule contains unsupported elements.
    """
    try:
        import torch
        import torchani
    except ImportError:
        raise ImportError(
            "torchani is not installed.  Install with: "
            "pip install 'torchani>=2.2'  "
            "(or use energy_backend='none' / 'xtb')."
        )

    # Get 3-D coordinates
    try:
        atomic_numbers, positions_angstrom = _smiles_to_3d(smiles)
    except ValueError as exc:
        logger.warning("3-D embedding failed for %s: %s", smiles, exc)
        strain = _estimate_strain_energy(smiles)
        return _nan_features(strain_energy=strain)

    # Validate element support
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mol_with_h = Chem.AddHs(mol)
        for atom in mol_with_h.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in _TORCHANI_ELEMENTS:
                raise ValueError(
                    f"TorchANI ANI-2x does not support element '{symbol}'. "
                    f"Supported: {sorted(_TORCHANI_ELEMENTS)}"
                )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchani.models.ANI2x(periodic_table_index=True).to(device)

    species = torch.tensor([atomic_numbers], dtype=torch.long, device=device)
    coords = torch.tensor(
        [positions_angstrom], dtype=torch.float32, device=device
    ).requires_grad_(True)

    energy_result = model((species, coords))
    total_energy_hartree = float(energy_result.energies.item())

    strain = _estimate_strain_energy(smiles)

    return {
        "total_energy_hartree": total_energy_hartree,
        "homo_ev": float("nan"),
        "lumo_ev": float("nan"),
        "gap_ev": float("nan"),
        "dipole_debye": float("nan"),
        "strain_energy_kcal": strain,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Tier 4: PySCF DFT
# ═══════════════════════════════════════════════════════════════════════════


def compute_dft_energy(smiles: str) -> dict[str, float]:
    """Compute energy using PySCF B3LYP/6-31G* DFT.

    This is expensive (~4 min for 20 heavy atoms) and should be used
    only for offline batch precomputation.

    Parameters
    ----------
    smiles : str
        SMILES string.

    Returns
    -------
    dict with keys matching ``ENERGY_FEATURE_KEYS``.

    Raises
    ------
    ImportError
        If pyscf is not installed.
    """
    try:
        from pyscf import gto, dft
    except ImportError:
        raise ImportError(
            "pyscf is not installed.  Install with: "
            "pip install 'pyscf>=2.4.0'  "
            "(or use energy_backend='none' / 'xtb')."
        )

    # Get 3-D coordinates
    try:
        atomic_numbers, positions_angstrom = _smiles_to_3d(smiles)
    except ValueError as exc:
        logger.warning("3-D embedding failed for %s: %s", smiles, exc)
        strain = _estimate_strain_energy(smiles)
        return _nan_features(strain_energy=strain)

    # Build PySCF molecule
    from ase.data import chemical_symbols

    atom_str = ""
    for z, pos in zip(atomic_numbers, positions_angstrom):
        sym = chemical_symbols[z]
        atom_str += f"{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}; "

    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = "6-31g*"
    mol.verbose = 0
    mol.build()

    # B3LYP DFT
    mf = dft.RKS(mol)
    mf.xc = "B3LYP"
    mf.verbose = 0

    try:
        total_energy_hartree = mf.kernel()
    except Exception as exc:
        logger.warning("PySCF DFT failed for %s: %s", smiles, exc)
        strain = _estimate_strain_energy(smiles)
        return _nan_features(strain_energy=strain)

    # HOMO/LUMO from MO energies + occupations
    homo_ev = float("nan")
    lumo_ev = float("nan")
    gap_ev = float("nan")
    try:
        mo_energies = mf.mo_energy  # Hartree
        mo_occ = mf.mo_occ

        occupied = mo_occ > 0.5
        if occupied.any():
            homo_idx = np.where(occupied)[0][-1]
            homo_ev = float(mo_energies[homo_idx] * _HARTREE_TO_EV)

        unoccupied = mo_occ < 0.5
        if unoccupied.any():
            lumo_idx = np.where(unoccupied)[0][0]
            lumo_ev = float(mo_energies[lumo_idx] * _HARTREE_TO_EV)

        if not math.isnan(homo_ev) and not math.isnan(lumo_ev):
            gap_ev = lumo_ev - homo_ev
    except Exception as exc:
        logger.warning("DFT orbital extraction failed for %s: %s", smiles, exc)

    # Dipole
    dipole_debye = float("nan")
    try:
        dipole = mf.dip_moment(verbose=0)
        dipole_debye = float(np.linalg.norm(dipole))
    except Exception as exc:
        logger.warning("DFT dipole extraction failed for %s: %s", smiles, exc)

    strain = _estimate_strain_energy(smiles)

    return {
        "total_energy_hartree": float(total_energy_hartree),
        "homo_ev": homo_ev,
        "lumo_ev": lumo_ev,
        "gap_ev": gap_ev,
        "dipole_debye": dipole_debye,
        "strain_energy_kcal": strain,
    }


# ═══════════════════════════════════════════════════════════════════════════
# NaN feature helper
# ═══════════════════════════════════════════════════════════════════════════


def _nan_features(*, strain_energy: float = _STRAIN_SENTINEL) -> dict[str, float]:
    """Return a feature dict with NaN energy values and given strain."""
    return {
        "total_energy_hartree": float("nan"),
        "homo_ev": float("nan"),
        "lumo_ev": float("nan"),
        "gap_ev": float("nan"),
        "dipole_debye": float("nan"),
        "strain_energy_kcal": strain_energy,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════


def get_energy_features(
    smiles: str,
    cache: EnergyCache | None,
    backend: str = "none",
    *,
    solvent: str | None = None,
) -> dict[str, float]:
    """Get energy features for a molecule using the tiered cascade.

    Parameters
    ----------
    smiles : str
        SMILES string.
    cache : EnergyCache | None
        Energy cache for Tier 1 lookups and write-through.
    backend : str
        Energy backend: ``"none"``, ``"xtb"``, ``"dft"``, ``"ml"``, ``"auto"``.
    solvent : str | None
        Solvent for GBSA (xTB) or PCM (DFT).

    Returns
    -------
    dict with keys matching ``ENERGY_FEATURE_KEYS``, or empty dict if
    ``backend == "none"``.
    """
    if backend == "none":
        return {}

    # Tier 1: cache lookup
    if cache is not None:
        cached = cache.get(smiles)
        if cached is not None:
            return cached

    # Tier 2-4: compute
    features: dict[str, float]

    if backend == "xtb":
        try:
            features = compute_xtb_energy(smiles, solvent=solvent)
        except ImportError:
            # tblite not available; fall back to xtb CLI binary
            features = compute_xtb_cli_energy(smiles, solvent=solvent)
    elif backend == "dft":
        features = compute_dft_energy(smiles)
    elif backend == "ml":
        features = compute_ml_energy(smiles)
    elif backend == "auto":
        features = _auto_cascade(smiles, solvent=solvent)
    else:
        raise ValueError(f"Unknown energy backend: {backend!r}")

    # Write-through to cache
    if cache is not None:
        cache.put(smiles, features)

    return features


def _auto_cascade(
    smiles: str,
    *,
    solvent: str | None = None,
) -> dict[str, float]:
    """Try backends in order: xTB (tblite) -> xTB (CLI) -> ML -> heuristic."""

    # Try xTB via tblite Python bindings
    try:
        return compute_xtb_energy(smiles, solvent=solvent)
    except ImportError:
        logger.info("tblite not available; trying xtb CLI for %s.", smiles)
    except Exception as exc:
        logger.warning("xTB (tblite) failed for %s: %s; trying xtb CLI.", smiles, exc)

    # Try xTB via CLI binary (provides full HOMO/LUMO)
    try:
        return compute_xtb_cli_energy(smiles, solvent=solvent)
    except FileNotFoundError:
        logger.info("xtb binary not found; trying TorchANI for %s.", smiles)
    except Exception as exc:
        logger.warning("xTB CLI failed for %s: %s; trying TorchANI.", smiles, exc)

    # Try ML (no HOMO/LUMO, only total energy)
    try:
        return compute_ml_energy(smiles)
    except ImportError:
        logger.info("torchani not available; using heuristic strain only for %s.", smiles)
    except ValueError as exc:
        logger.warning("TorchANI unsupported elements for %s: %s", smiles, exc)
    except Exception as exc:
        logger.warning("TorchANI failed for %s: %s", smiles, exc)

    # Fallback: heuristic strain only
    strain = _estimate_strain_energy(smiles)
    return _nan_features(strain_energy=strain)
