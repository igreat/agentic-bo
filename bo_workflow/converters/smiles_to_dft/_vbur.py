"""Percent buried volume (%VBur) calculation.

Uses the Morfeus library if installed, otherwise fills with NaN.
"""

from __future__ import annotations

import numpy as np


def vbur_available() -> bool:
    """Check if the Morfeus library is importable."""
    try:
        import morfeus  # noqa: F401

        return True
    except ImportError:
        return False


def compute_vbur(
    coords: np.ndarray,
    atom_symbols: list[str],
    center_atom_idx: int,
    *,
    radius: float = 3.5,
) -> float:
    """Compute percent buried volume at a given center atom.

    Parameters
    ----------
    coords : array of shape (n_atoms, 3), Angstroms
    atom_symbols : element symbols in atom order
    center_atom_idx : 0-based index of the center atom
    radius : sphere radius in Angstrom (default 3.5, standard SambVca)

    Returns
    -------
    float
        Fraction of buried volume in [0, 1]. Returns NaN if Morfeus
        is not installed or computation fails.
    """
    try:
        from morfeus import BuriedVolume

        # Morfeus uses 1-based indexing
        bv = BuriedVolume(
            atom_symbols,
            coords,
            center_atom_idx + 1,  # 1-based
            radius=radius,
        )
        return float(bv.fraction_buried_volume)

    except ImportError:
        return float("nan")

    except Exception:
        return float("nan")
