"""PySCF-based DFT computation: SCF energy, orbital energies, dipole, NMR.

All calculations use restricted Kohn-Sham (RKS) with B3LYP/6-31G* by
default, matching the level of theory in the original Doyle et al. study.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DFTResult:
    """Results from a single-conformer DFT calculation."""

    converged: bool
    scf_energy: float  # Hartree
    homo_energy: float  # Hartree
    lumo_energy: float  # Hartree
    dipole_moment: float  # Debye (vector magnitude)
    mulliken_charges: np.ndarray  # shape (n_atoms,)
    nmr_shieldings: np.ndarray  # shape (n_atoms,), ppm isotropic
    nmr_anisotropies: np.ndarray  # shape (n_atoms,), ppm


def _nan_result(n_atoms: int) -> DFTResult:
    """Return a DFTResult filled with NaN (for failed calculations)."""
    return DFTResult(
        converged=False,
        scf_energy=float("nan"),
        homo_energy=float("nan"),
        lumo_energy=float("nan"),
        dipole_moment=float("nan"),
        mulliken_charges=np.full(n_atoms, float("nan")),
        nmr_shieldings=np.full(n_atoms, float("nan")),
        nmr_anisotropies=np.full(n_atoms, float("nan")),
    )


def run_dft(
    atom_symbols: list[str],
    coords: np.ndarray,
    *,
    basis: str = "6-31g*",
    xc: str = "b3lyp",
    max_cycle: int = 200,
    verbose: int = 0,
    compute_nmr: bool = True,
) -> DFTResult:
    """Run a full DFT calculation on a single conformer.

    Pipeline
    --------
    1. Build ``pyscf.gto.Mole`` from atom list + coordinates
    2. Run RKS DFT with specified functional/basis
    3. Extract HOMO/LUMO from orbital energies
    4. Compute dipole moment magnitude
    5. Compute Mulliken population charges
    6. (optional) Run NMR shielding calculation

    Parameters
    ----------
    atom_symbols : list of element symbols (e.g. ["C", "H", "H", "H", "H"])
    coords : array of shape (n_atoms, 3), Angstroms
    basis : Gaussian basis set name
    xc : DFT exchange-correlation functional
    max_cycle : maximum SCF iterations
    verbose : PySCF verbosity level (0=silent, 4=debug)
    compute_nmr : whether to run NMR shielding calculation

    Returns
    -------
    DFTResult
        If SCF fails to converge, ``converged=False`` and values are NaN.
    """
    import pyscf
    from pyscf import dft, gto, scf

    n_atoms = len(atom_symbols)

    # Suppress PySCF output unless verbose
    if verbose < 2:
        pyscf.lib.logger.QUIET = True

    # 1. Build Mole
    mol = _build_mol(atom_symbols, coords, basis, verbose)

    # 2. Run DFT SCF
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.max_cycle = max_cycle
    mf.conv_tol = 1e-9
    mf.verbose = verbose

    try:
        mf.kernel()
    except Exception as exc:
        if verbose:
            print(f"[dft] SCF failed: {exc}", file=sys.stderr)
        return _nan_result(n_atoms)

    if not mf.converged:
        # Retry with tighter DIIS and more cycles
        if verbose:
            print("[dft] SCF not converged, retrying with ADIIS...", file=sys.stderr)
        mf2 = dft.RKS(mol)
        mf2.xc = xc
        mf2.max_cycle = 500
        mf2.diis = scf.ADIIS()
        mf2.verbose = verbose
        try:
            mf2.kernel()
        except Exception:
            return _nan_result(n_atoms)
        if not mf2.converged:
            return _nan_result(n_atoms)
        mf = mf2

    # 3. HOMO / LUMO
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    occ_mask = mo_occ > 0
    if occ_mask.any():
        homo_idx = np.where(occ_mask)[0][-1]
        homo_energy = float(mo_energy[homo_idx])
    else:
        homo_energy = float("nan")

    virt_mask = mo_occ == 0
    if virt_mask.any():
        lumo_idx = np.where(virt_mask)[0][0]
        lumo_energy = float(mo_energy[lumo_idx])
    else:
        lumo_energy = float("nan")

    # 4. Dipole moment (Debye)
    dip_vec = mf.dip_moment(verbose=0)
    dipole_moment = float(np.linalg.norm(dip_vec))

    # 5. Mulliken charges
    try:
        pop, chg = mf.mulliken_pop(verbose=0)
        mulliken_charges = np.array(chg, dtype=np.float64)
    except Exception:
        mulliken_charges = np.full(n_atoms, float("nan"))

    # 6. NMR shieldings
    if compute_nmr:
        nmr_iso, nmr_aniso = _compute_nmr(mf, verbose)
    else:
        nmr_iso = np.full(n_atoms, float("nan"))
        nmr_aniso = np.full(n_atoms, float("nan"))

    return DFTResult(
        converged=True,
        scf_energy=float(mf.e_tot),
        homo_energy=homo_energy,
        lumo_energy=lumo_energy,
        dipole_moment=dipole_moment,
        mulliken_charges=mulliken_charges,
        nmr_shieldings=nmr_iso,
        nmr_anisotropies=nmr_aniso,
    )


def _build_mol(
    atom_symbols: list[str],
    coords: np.ndarray,
    basis: str,
    verbose: int = 0,
):
    """Build a PySCF Mole object from atoms + coordinates."""
    from pyscf import gto

    # Build atom string: "C  0.0  0.0  0.0; H  1.0  0.0  0.0; ..."
    atom_str = "; ".join(
        f"{sym}  {coords[i, 0]:.8f}  {coords[i, 1]:.8f}  {coords[i, 2]:.8f}"
        for i, sym in enumerate(atom_symbols)
    )

    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = basis
    mol.verbose = verbose
    mol.build()
    return mol


def _compute_nmr(mf, verbose: int = 0):
    """Run NMR shielding calculation, return (isotropic, anisotropy) arrays.

    Uses PySCF's ``prop.nmr`` module with GIAO (gauge-including atomic
    orbitals) for gauge-invariant shielding tensors.

    Returns
    -------
    (isotropic, anisotropy) : tuple of arrays, each shape (n_atoms,)
    """
    n_atoms = mf.mol.natm

    try:
        import warnings

        warnings.filterwarnings("ignore", module="pyscf")

        from pyscf.prop import nmr as nmr_mod

        nmr_calc = nmr_mod.RKS(mf)
        nmr_calc.verbose = verbose

        # NOTE: Do NOT call mf.grids.build() here — it changes grid
        # parameters and triggers a blksize assertion error in
        # pyscf.dft.numint.block_loop.  The grids from the SCF step
        # are already compatible.

        # kernel() returns shielding tensors: shape (n_atoms, 3, 3)
        shielding_tensors = nmr_calc.kernel()

        isotropic = np.zeros(n_atoms, dtype=np.float64)
        anisotropy = np.zeros(n_atoms, dtype=np.float64)

        for i in range(n_atoms):
            tensor = np.array(shielding_tensors[i])  # 3x3
            # Isotropic = trace / 3
            iso = np.trace(tensor) / 3.0
            isotropic[i] = iso

            # Anisotropy from eigenvalues
            eigvals = np.sort(np.linalg.eigvalsh(tensor))
            # Anisotropy = largest eigenvalue - average of other two
            aniso = eigvals[2] - (eigvals[0] + eigvals[1]) / 2.0
            anisotropy[i] = aniso

        return isotropic, anisotropy

    except Exception as exc:
        if verbose:
            print(f"[dft] NMR calculation failed: {exc}", file=sys.stderr)
        return (
            np.full(n_atoms, float("nan")),
            np.full(n_atoms, float("nan")),
        )
