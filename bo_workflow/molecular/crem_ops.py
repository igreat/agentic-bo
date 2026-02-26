"""CReM-based scaffold editing operations.

Wraps CReM (Context-aware Replacements for Molecules) for scaffold
hopping with position-level mutation control.  Generates chemically
reasonable molecular variants via fragment replacement from a real
molecular database (ChEMBL-derived).

Key capabilities:
- Position-level control via SMARTS patterns (fix/allow specific regions)
- Mutation categorisation (ring_add, ring_remove, heteroatom_swap, etc.)
- Diverse seed selection from datasets
- CReM database auto-detection and download
"""

from __future__ import annotations

import gzip
import shutil
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# ---------------------------------------------------------------------------
# CReM database management
# ---------------------------------------------------------------------------

CREM_DB_URLS: dict[str, str] = {
    "chembl33_sa2_f5": (
        "https://zenodo.org/records/16909329/files/chembl33_sa2_f5.db.gz?download=1"
    ),
    "replacements02_sc2": (
        "http://www.qsar4u.com/files/cremdb/replacements02_sc2.db.gz"
    ),
}
DEFAULT_DB_NAME = "chembl33_sa2_f5"

# Well-known CReM database file name patterns (preferred order)
_KNOWN_DB_NAMES = [
    "chembl33_sa2_f5",
    "chembl22_sa2",
    "chembl22_sa2_hac12",
    "replacements02_sc2",
    "replacements02_sa2",
]

# Where to look for the database by default
_PROJECT_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def locate_crem_db(
    db_path: str | Path | None = None,
    db_name: str | None = None,
    search_dirs: list[Path] | None = None,
) -> Path | None:
    """Locate a CReM fragment database on disk.

    Search order:
    1. Explicit *db_path* if provided.
    2. ``data/<db_name>.db`` for the specified name.
    3. ``data/*.db`` — any CReM-like database in the project data dir.
    4. ``~/.crem/<db_name>.db`` in the user home directory.
    5. Each directory in *search_dirs*.

    Returns the ``Path`` if found, ``None`` otherwise.
    """
    if db_path is not None:
        p = Path(db_path)
        if p.exists():
            return p
        return None

    names_to_try = [db_name] if db_name else _KNOWN_DB_NAMES

    # Search data/ and ~/.crem/ for known names
    search_roots = [_PROJECT_DATA_DIR, Path.home() / ".crem"]
    if search_dirs:
        search_roots.extend(search_dirs)

    for name in names_to_try:
        for root in search_roots:
            c = root / f"{name}.db"
            if c.exists():
                return c

    # Fallback: any .db file in data/ that looks like a CReM database
    if _PROJECT_DATA_DIR.exists():
        for db_file in sorted(_PROJECT_DATA_DIR.glob("*.db")):
            # Quick heuristic: CReM DBs are > 10MB
            if db_file.stat().st_size > 10_000_000:
                return db_file

    return None


def download_crem_db(
    db_name: str = DEFAULT_DB_NAME,
    target_dir: Path | None = None,
    verbose: bool = False,
) -> Path:
    """Download and extract a CReM fragment database.

    Downloads the ``.gz`` file, extracts it, and returns the path to
    the ``.db`` file.  Default target: ``data/<db_name>.db``.
    """
    if db_name not in CREM_DB_URLS:
        raise ValueError(
            f"Unknown CReM database '{db_name}'. "
            f"Available: {list(CREM_DB_URLS)}"
        )

    url = CREM_DB_URLS[db_name]
    if target_dir is None:
        target_dir = _PROJECT_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    gz_path = target_dir / f"{db_name}.db.gz"
    db_path = target_dir / f"{db_name}.db"

    if db_path.exists():
        return db_path

    if verbose:
        print(f"[crem] Downloading {url} ...")  # noqa: T201
    urllib.request.urlretrieve(url, gz_path)  # noqa: S310

    if verbose:
        print(f"[crem] Extracting to {db_path} ...")  # noqa: T201
    with gzip.open(gz_path, "rb") as f_in, db_path.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()

    return db_path


# ---------------------------------------------------------------------------
# Position-level mutation constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MutationConstraint:
    """Defines which parts of a molecule can be mutated by CReM.

    Attributes:
        mutable_smarts: SMARTS patterns marking mutable regions.
            Atoms matching any of these are allowed to change.
            If empty, all atoms are mutable (no constraints).
        fixed_smarts: SMARTS patterns marking fixed regions.
            Atoms matching any of these are protected.
            Takes precedence over *mutable_smarts*.
        fixed_atom_indices: Explicit atom indices to protect (0-based).
    """

    mutable_smarts: list[str] = field(default_factory=list)
    fixed_smarts: list[str] = field(default_factory=list)
    fixed_atom_indices: list[int] = field(default_factory=list)


def resolve_protected_atoms(
    mol: Chem.Mol,
    constraint: MutationConstraint,
) -> set[int]:
    """Resolve a MutationConstraint into a set of protected atom indices.

    Logic:
    1. If *mutable_smarts* is non-empty: start with ALL atoms protected,
       then un-protect atoms matching any mutable pattern.
    2. Protect atoms matching any *fixed_smarts* pattern.
    3. Protect explicit *fixed_atom_indices*.
    """
    n_atoms = mol.GetNumAtoms()

    if constraint.mutable_smarts:
        # Start by protecting everything
        protected = set(range(n_atoms))
        # Un-protect atoms matching mutable patterns
        for smarts in constraint.mutable_smarts:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                continue
            for match in mol.GetSubstructMatches(pattern):
                protected -= set(match)
    else:
        # No mutable_smarts → nothing protected by default
        protected: set[int] = set()

    # Protect atoms matching fixed patterns (takes precedence)
    for smarts in constraint.fixed_smarts:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        for match in mol.GetSubstructMatches(pattern):
            protected |= set(match)

    # Protect explicit indices
    protected |= set(constraint.fixed_atom_indices)

    # Clamp to valid range
    return {i for i in protected if 0 <= i < n_atoms}


def constraint_from_dict(d: dict[str, Any] | None) -> MutationConstraint | None:
    """Build a MutationConstraint from a JSON-serialisable dict."""
    if d is None:
        return None
    return MutationConstraint(
        mutable_smarts=d.get("mutable_smarts", []),
        fixed_smarts=d.get("fixed_smarts", []),
        fixed_atom_indices=d.get("fixed_atom_indices", []),
    )


# ---------------------------------------------------------------------------
# Mutation categorisation
# ---------------------------------------------------------------------------


class MutationType:
    RING_ADD = "ring_add"
    RING_REMOVE = "ring_remove"
    HETEROATOM_SWAP = "heteroatom_swap"
    RING_RESIZE = "ring_resize"
    FRAGMENT_REPLACE = "fragment_replace"
    GROW = "grow"


@dataclass
class CremMutation:
    """A single CReM-generated mutation result."""

    smiles: str
    parent_smiles: str
    mutation_type: str
    ring_delta: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "parent_smiles": self.parent_smiles,
            "mutation_type": self.mutation_type,
            "ring_delta": self.ring_delta,
        }


def _ring_info(mol: Chem.Mol) -> tuple[int, list[int], Counter]:
    """Extract ring summary: (count, sizes, atom-type counter per ring)."""
    ri = mol.GetRingInfo()
    rings = ri.AtomRings()
    n_rings = len(rings)
    sizes = sorted(len(r) for r in rings)
    # Count element types across all ring atoms
    type_counts: Counter = Counter()
    for ring in rings:
        for idx in ring:
            sym = mol.GetAtomWithIdx(idx).GetSymbol()
            type_counts[sym] += 1
    return n_rings, sizes, type_counts


def categorize_mutation(
    parent_mol: Chem.Mol,
    child_mol: Chem.Mol,
) -> tuple[str, int]:
    """Categorise a mutation by comparing ring systems.

    Returns (mutation_type, ring_delta).
    """
    p_n, p_sizes, p_types = _ring_info(parent_mol)
    c_n, c_sizes, c_types = _ring_info(child_mol)
    ring_delta = c_n - p_n

    if ring_delta > 0:
        return MutationType.RING_ADD, ring_delta
    if ring_delta < 0:
        return MutationType.RING_REMOVE, ring_delta

    # Same number of rings — check for composition changes
    if p_sizes != c_sizes:
        return MutationType.RING_RESIZE, 0

    # Same ring count and sizes — check atom types
    if p_types != c_types:
        return MutationType.HETEROATOM_SWAP, 0

    return MutationType.FRAGMENT_REPLACE, 0


# ---------------------------------------------------------------------------
# CReM mutation generation
# ---------------------------------------------------------------------------


def generate_mutations(
    smiles: str,
    *,
    db_path: str | Path,
    constraint: MutationConstraint | None = None,
    max_size: int = 3,
    radius: int = 3,
    max_replacements: int = 100,
    min_freq: int = 10,
    operations: list[str] | None = None,
) -> list[CremMutation]:
    """Generate CReM mutations for a single seed molecule.

    Args:
        smiles: Seed molecule SMILES.
        db_path: Path to CReM fragment database.
        constraint: Position-level mutation control.
        max_size: Maximum heavy atoms in replacement fragment.
        radius: Context radius for fragment matching.
        max_replacements: Maximum mutations per operation.
        min_freq: Minimum fragment frequency in the database.
        operations: CReM operations to run. Default: ``["mutate"]``.
            ``"mutate"`` replaces fragments; ``"grow"`` adds atoms.

    Returns:
        Deduplicated list of CremMutation objects.
    """
    from crem.crem import mutate_mol, grow_mol

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")

    protected_ids = None
    if constraint is not None:
        prot = resolve_protected_atoms(mol, constraint)
        if prot:
            protected_ids = tuple(sorted(prot))

    if operations is None:
        operations = ["mutate"]

    db_str = str(db_path)
    results: dict[str, CremMutation] = {}  # canonical SMILES → mutation

    for op in operations:
        if op == "mutate":
            raw = list(
                mutate_mol(
                    mol,
                    db_name=db_str,
                    max_size=max_size,
                    radius=radius,
                    return_mol=False,
                    protected_ids=protected_ids,
                    min_freq=min_freq,
                    max_replacements=max_replacements,
                )
            )
        elif op == "grow":
            raw = list(
                grow_mol(
                    mol,
                    db_name=db_str,
                    max_atoms=max_size,
                    radius=radius,
                    return_mol=False,
                    protected_ids=protected_ids,
                    min_freq=min_freq,
                    max_replacements=max_replacements,
                )
            )
        else:
            continue

        for child_smi in raw:
            child_mol = Chem.MolFromSmiles(child_smi)
            if child_mol is None:
                continue
            canon = Chem.MolToSmiles(child_mol, canonical=True)
            if canon == smiles or canon in results:
                continue
            mut_type, ring_delta = categorize_mutation(mol, child_mol)
            if op == "grow":
                mut_type = MutationType.GROW
            results[canon] = CremMutation(
                smiles=canon,
                parent_smiles=smiles,
                mutation_type=mut_type,
                ring_delta=ring_delta,
            )

    return list(results.values())


def generate_candidate_pool(
    seed_smiles: list[str],
    *,
    db_path: str | Path,
    constraint: MutationConstraint | None = None,
    max_size: int = 3,
    radius: int = 3,
    max_replacements_per_seed: int = 100,
    min_freq: int = 10,
    operations: list[str] | None = None,
    exclude_smiles: set[str] | None = None,
    verbose: bool = False,
) -> list[CremMutation]:
    """Generate a deduplicated candidate pool from multiple seeds.

    Returns:
        Deduplicated list — each unique child SMILES appears once.
    """
    pool: dict[str, CremMutation] = {}
    exclude = exclude_smiles or set()

    for i, smi in enumerate(seed_smiles):
        try:
            muts = generate_mutations(
                smi,
                db_path=db_path,
                constraint=constraint,
                max_size=max_size,
                radius=radius,
                max_replacements=max_replacements_per_seed,
                min_freq=min_freq,
                operations=operations,
            )
        except (ValueError, Exception) as exc:
            if verbose:
                print(f"[crem] Seed {i+1}/{len(seed_smiles)} failed: {exc}")  # noqa: T201
            continue

        for m in muts:
            if m.smiles not in pool and m.smiles not in exclude:
                pool[m.smiles] = m

        if verbose:
            print(  # noqa: T201
                f"[crem] Seed {i+1}/{len(seed_smiles)}: "
                f"+{len(muts)} mutations, pool={len(pool)}"
            )

    return list(pool.values())


# ---------------------------------------------------------------------------
# Seed molecule selection
# ---------------------------------------------------------------------------


def select_seed_molecules(
    dataset_path: str | Path,
    target_column: str,
    objective: str,
    smiles_column: str,
    *,
    top_k: int = 10,
) -> list[str]:
    """Select diverse, high-performing seed molecules from a dataset.

    Strategy:
    1. Rank molecules by target value (best first per objective).
    2. From the top 2*top_k candidates, pick *top_k* using MaxMin
       Tanimoto diversity on Morgan fingerprints.
    """
    import pandas as pd

    data = pd.read_csv(dataset_path)
    if target_column not in data.columns:
        raise ValueError(f"Target '{target_column}' not in dataset.")
    if smiles_column not in data.columns:
        raise ValueError(f"SMILES column '{smiles_column}' not in dataset.")

    # Drop rows without target or SMILES
    df = data[[smiles_column, target_column]].dropna().copy()
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    df = df.dropna()

    if len(df) == 0:
        raise ValueError("No valid rows with SMILES + target.")

    # Sort by target (best first)
    ascending = objective == "min"
    df = df.sort_values(target_column, ascending=ascending).reset_index(drop=True)

    # Take top-2K candidates
    pool_size = min(2 * top_k, len(df))
    pool = df.head(pool_size)

    # Compute Morgan fingerprints for diversity picking
    smiles_list = pool[smiles_column].tolist()
    fps = []
    valid_smiles: list[str] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fps.append(fp)
            valid_smiles.append(str(smi))

    if len(valid_smiles) <= top_k:
        return valid_smiles

    # MaxMin diversity picking
    selected_indices = [0]  # Start with the best molecule
    remaining = set(range(1, len(valid_smiles)))

    while len(selected_indices) < top_k and remaining:
        best_idx = -1
        best_min_dist = -1.0
        for idx in remaining:
            min_dist = min(
                1.0 - DataStructs.TanimotoSimilarity(fps[idx], fps[s])
                for s in selected_indices
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        if best_idx < 0:
            break
        selected_indices.append(best_idx)
        remaining.discard(best_idx)

    return [valid_smiles[i] for i in selected_indices]
