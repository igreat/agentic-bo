"""Disk-backed cache mapping canonical SMILES → computed descriptors.

The cache stores results as a JSON file so that interrupted DFT
computations can resume without recomputing molecules.  With only
~44 unique molecules in the Buchwald-Hartwig dataset, the cache
file is small.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DescriptorCache:
    """JSON-file cache: canonical SMILES → {descriptor_name: float}."""

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._cache_path = cache_dir / "dft_cache.json"
        self._data: dict[str, dict[str, float]] = {}
        if self._cache_path.exists():
            try:
                self._data = json.loads(self._cache_path.read_text())
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def has(self, smiles: str) -> bool:
        """Check if canonical SMILES is already cached."""
        return self._canonical(smiles) in self._data

    def get(self, smiles: str) -> dict[str, float] | None:
        """Return cached descriptors, or None."""
        return self._data.get(self._canonical(smiles))

    def put(self, smiles: str, descriptors: dict[str, float]) -> None:
        """Store descriptors for a canonical SMILES and flush to disk."""
        self._data[self._canonical(smiles)] = descriptors
        self.flush()

    def flush(self) -> None:
        """Write current cache state to disk."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # Use a custom encoder that handles NaN → null
        self._cache_path.write_text(
            json.dumps(self._data, indent=2, allow_nan=True)
        )

    @property
    def size(self) -> int:
        return len(self._data)

    @staticmethod
    def _canonical(smiles: str) -> str:
        """Canonicalize SMILES for cache key consistency.

        Supports either plain SMILES or ``prefix::SMILES`` keys so the same
        molecule can be cached separately for different component roles while
        still canonicalizing the molecular part.
        """
        prefix = ""
        smiles_part = smiles
        if "::" in smiles:
            prefix, smiles_part = smiles.split("::", 1)
            prefix = f"{prefix}::"

        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles_part)
            if mol is not None:
                return prefix + Chem.MolToSmiles(mol)
        except Exception:
            pass
        return prefix + smiles_part
