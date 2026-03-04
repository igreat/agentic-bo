"""EGFR experiment utility functions.

Data loading and preparation helpers for EGFR molecular optimization scripts.
Used by bo_workflow.scripts.egfr_ic50_global_experiment and similar scripts.
"""

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from bo_workflow.converters.molecule_descriptors import (
    canonicalize_smiles,
    decode_nearest,
    encode_molecules,
    is_descriptor_col,
)


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------


def to_pic50(ic50_nm: float) -> float:
    """Convert IC50 in nM to pIC50."""
    if float(ic50_nm) <= 0:
        raise ValueError("ic50_nM must be > 0")
    return 9.0 - math.log10(float(ic50_nm))


# ---------------------------------------------------------------------------
# Molecule validation
# ---------------------------------------------------------------------------

_ALLOWED_ATOMIC_NUMS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}


def is_reasonable_seed_smiles(smiles: str) -> bool:
    """Basic med-chem filter: exclude salts, multi-fragments, inorganics, outlier MW."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    if "." in Chem.MolToSmiles(mol):
        return False
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    if not atoms or 6 not in atoms:
        return False
    if any(z not in _ALLOWED_ATOMIC_NUMS for z in atoms):
        return False
    heavy = mol.GetNumHeavyAtoms()
    if heavy < 8 or heavy > 90:
        return False
    return True


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_full_dataset(
    dataset_path: Path,
    target_column: str = "ic50_nM",
    *,
    max_pic50: float | None = 12.0,
    fix_tiny_ic50_as_molar: bool = False,
) -> list[tuple[str, str, float]]:
    """Load EGFR dataset, return (smiles, canonical_smiles, pIC50) triples.

    Handles both ic50_nM and pIC50 target columns.
    Drops duplicate canonical SMILES (keeps first occurrence).
    """
    with open(dataset_path) as f:
        rows = list(csv.DictReader(f))

    data: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    fixed_tiny = 0
    dropped_extreme = 0

    for row in rows:
        smi = row.get("smiles", "").strip()
        can = canonicalize_smiles(smi)
        if not can or can in seen:
            continue

        if target_column == "pIC50" and "pIC50" in row:
            raw = row["pIC50"]
            if not raw:
                continue
            y = float(raw)
        elif "ic50_nM" in row:
            raw = row["ic50_nM"]
            if not raw:
                continue
            ic50 = float(raw)
            if fix_tiny_ic50_as_molar and ic50 < 1e-6:
                ic50 *= 1e9
                fixed_tiny += 1
            if ic50 <= 0:
                continue
            y = to_pic50(ic50)
        elif "pIC50" in row:
            raw = row["pIC50"]
            if not raw:
                continue
            y = float(raw)
        else:
            continue

        if max_pic50 is not None and y > max_pic50:
            dropped_extreme += 1
            continue

        data.append((smi, can, y))
        seen.add(can)

    if not data:
        raise ValueError(f"No valid molecules found in {dataset_path}")

    if fixed_tiny:
        print(f"[data] Converted {fixed_tiny} tiny ic50_nM values (<1e-6) from M to nM")
    if dropped_extreme:
        print(f"[data] Dropped {dropped_extreme} rows with pIC50 > {max_pic50}")

    return data


def load_seed_rows(
    seed_csv: Path,
    smiles_column: str,
    target_column: str,
) -> list[tuple[str, float]]:
    """Load labeled seed rows from CSV, return (smiles, y) pairs."""
    frame = pd.read_csv(seed_csv)
    if smiles_column not in frame.columns:
        raise ValueError(f"Missing smiles column '{smiles_column}' in {seed_csv}")
    if target_column not in frame.columns:
        raise ValueError(f"Missing target column '{target_column}' in {seed_csv}")

    rows: list[tuple[str, float]] = []
    seen: set[str] = set()
    for _, row in frame.iterrows():
        smi = str(row[smiles_column]).strip()
        can = canonicalize_smiles(smi)
        if not smi or can is None or can in seen:
            continue
        y_raw = row[target_column]
        if pd.isna(y_raw):
            continue
        rows.append((smi, float(y_raw)))
        seen.add(can)

    if not rows:
        raise ValueError("No valid seed rows found")
    return rows


def load_candidate_smiles(candidate_csv: Path, smiles_column: str) -> list[str]:
    """Load unique candidate SMILES from CSV."""
    frame = pd.read_csv(candidate_csv)
    if smiles_column not in frame.columns:
        raise ValueError(f"Missing smiles column '{smiles_column}' in {candidate_csv}")

    smiles: list[str] = []
    seen: set[str] = set()
    for value in frame[smiles_column].dropna().tolist():
        smi = str(value).strip()
        can = canonicalize_smiles(smi)
        if not smi or can is None or can in seen:
            continue
        smiles.append(smi)
        seen.add(can)
    return smiles


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def build_runtime_dataset(
    output_csv: Path,
    seed_rows: list[tuple[str, float]],
    candidate_smiles: list[str],
) -> dict[str, int]:
    """Write runtime dataset CSV with labeled seeds and unlabeled candidates."""
    seed_canonical: set[str] = set()
    records: list[dict[str, Any]] = []

    for smi, y in seed_rows:
        can = canonicalize_smiles(smi)
        if can is None or can in seed_canonical:
            continue
        seed_canonical.add(can)
        records.append({"smiles": smi, "pIC50": float(y)})

    candidate_count = 0
    seen_candidate: set[str] = set()
    for smi in candidate_smiles:
        can = canonicalize_smiles(smi)
        if can is None or can in seed_canonical or can in seen_candidate:
            continue
        records.append({"smiles": smi, "pIC50": float("nan")})
        seen_candidate.add(can)
        candidate_count += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_csv, index=False)
    return {
        "seed_rows": len(seed_canonical),
        "candidate_rows": candidate_count,
        "total_rows": len(records),
    }


def build_descriptor_assets(
    runtime_csv: Path,
    fp_bits: int,
) -> tuple[pd.DataFrame, list[str], dict[str, dict[str, float]], dict[str, float]]:
    """Build descriptor catalog and canonical lookup maps from runtime CSV.

    Returns (catalog_df, descriptor_cols, descriptor_lookup, pic50_lookup).
    """
    _, catalog_df = encode_molecules(
        runtime_csv,
        smiles_cols=["smiles"],
        morgan_bits=int(fp_bits),
    )

    descriptor_cols = sorted([c for c in catalog_df.columns if is_descriptor_col(c)])

    catalog_df = catalog_df.copy()
    catalog_df["canonical_smiles"] = catalog_df["smiles"].map(canonicalize_smiles)
    catalog_df["pIC50"] = pd.to_numeric(catalog_df["pIC50"], errors="coerce")
    catalog_df = catalog_df.dropna(subset=["canonical_smiles"]).copy()
    catalog_df = (
        catalog_df.sort_values("pIC50", ascending=False, na_position="last")
        .drop_duplicates(subset=["canonical_smiles"], keep="first")
        .reset_index(drop=True)
    )

    descriptor_lookup: dict[str, dict[str, float]] = {}
    pic50_lookup: dict[str, float] = {}
    for _, row in catalog_df.iterrows():
        can = str(row["canonical_smiles"])
        descriptor_lookup[can] = {
            col: float(row[col]) if pd.notna(row[col]) else 0.0
            for col in descriptor_cols
        }
        if pd.notna(row["pIC50"]):
            pic50_lookup[can] = float(row["pIC50"])

    return catalog_df, descriptor_cols, descriptor_lookup, pic50_lookup


def write_train_features(
    output_csv: Path,
    train_canonical: set[str],
    descriptor_lookup: dict[str, dict[str, float]],
    pic50_lookup: dict[str, float],
) -> None:
    """Materialize canonical train set into descriptor feature rows."""
    rows: list[dict[str, float]] = []
    for can in sorted(train_canonical):
        desc = descriptor_lookup.get(can)
        target = pic50_lookup.get(can)
        if desc is None or target is None:
            continue
        payload = dict(desc)
        payload["pIC50"] = float(target)
        rows.append(payload)

    if not rows:
        raise ValueError("No training rows could be materialized in descriptor space")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def append_train_features(
    output_csv: Path,
    updates: list[dict[str, float | str]],
    descriptor_lookup: dict[str, dict[str, float]],
) -> int:
    """Append observed labels (mapped by canonical SMILES) to train feature CSV."""
    rows: list[dict[str, float]] = []
    for item in updates:
        can = str(item["canonical"])
        desc = descriptor_lookup.get(can)
        if desc is None:
            continue
        payload = dict(desc)
        payload["pIC50"] = float(item["y"])
        rows.append(payload)

    if rows:
        pd.DataFrame(rows).to_csv(output_csv, mode="a", header=False, index=False)
    return len(rows)


# ---------------------------------------------------------------------------
# Descriptor-space decode helpers
# ---------------------------------------------------------------------------


def observations_to_proposals(
    observations: list[dict],
    catalog_df: pd.DataFrame,
    descriptor_cols: list[str],
) -> list[dict[str, float | str]]:
    """Map descriptor-space observations back to nearest catalog molecules."""
    proposal_map: dict[str, dict[str, float | str]] = {}
    used_in_round: set[str] = set()
    k = min(40, len(catalog_df))
    if k < 1:
        return []

    for obs in observations:
        x = dict(obs.get("x", {}))
        query_vec = np.array(
            [float(x.get(c, 0.0)) for c in descriptor_cols], dtype=float
        )
        neighbors = decode_nearest(query_vec, catalog_df, descriptor_cols, k=k)

        chosen_smiles = None
        chosen_can = None
        for candidate in neighbors:
            smi = str(candidate.get("smiles", ""))
            can = str(candidate.get("canonical_smiles", ""))
            if not can:
                can = canonicalize_smiles(smi) or ""
            if not can or can in used_in_round:
                continue
            chosen_smiles = smi
            chosen_can = can
            break

        if chosen_smiles is None or chosen_can is None:
            continue

        used_in_round.add(chosen_can)
        pred = float(obs.get("y", 0.0))
        current = proposal_map.get(chosen_can)
        if current is None or pred > float(current.get("predicted", -999.0)):
            proposal_map[chosen_can] = {
                "smiles": chosen_smiles,
                "canonical": chosen_can,
                "predicted": pred,
            }

    return list(proposal_map.values())


def nearest_smiles(
    x: dict,
    catalog: pd.DataFrame,
    descriptor_cols: list[str],
    *,
    used_canonical: set[str],
    k: int = 20,
) -> tuple[str, str]:
    """Decode one descriptor point to the nearest non-used molecule in catalog."""
    query_vec = np.array([float(x.get(c, 0.0)) for c in descriptor_cols], dtype=float)
    neighbors = decode_nearest(
        query_vec, catalog, descriptor_cols, k=min(k, len(catalog))
    )

    for entry in neighbors:
        smi = str(entry.get("smiles", ""))
        can = str(entry.get("canonical_smiles", ""))
        if not can:
            can = canonicalize_smiles(smi) or ""
        if can and can not in used_canonical:
            return smi, can

    raise ValueError("Could not map suggestion to a candidate SMILES")


# ---------------------------------------------------------------------------
# Observation parsing
# ---------------------------------------------------------------------------


def parse_observation_rows(path: Path, target_column: str) -> list[dict[str, Any]]:
    """Parse observation CSV with smiles and target columns."""
    frame = pd.read_csv(path)
    if "smiles" not in frame.columns:
        raise ValueError("Observation CSV must include 'smiles' column")
    if target_column not in frame.columns:
        raise ValueError(f"Observation CSV missing target column '{target_column}'")

    rows: list[dict[str, Any]] = []
    has_suggestion_id = "suggestion_id" in frame.columns

    for _, row in frame.iterrows():
        smi = str(row["smiles"]).strip()
        if not smi:
            continue
        y_raw = row[target_column]
        if pd.isna(y_raw):
            continue
        payload: dict[str, Any] = {"smiles": smi, "y": float(y_raw)}
        if has_suggestion_id and pd.notna(row.get("suggestion_id")):
            payload["suggestion_id"] = str(row["suggestion_id"])
        rows.append(payload)

    if not rows:
        raise ValueError("No observation rows found")
    return rows


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------


def reselect_active_features(
    state_path: Path,
    descriptor_lookup: dict[str, dict[str, float]],
    descriptor_cols: list[str],
    observed_canonical: set[str],
    pic50_lookup: dict[str, float],
    *,
    max_dims: int = 15,
    verbose: bool = False,
) -> int:
    """Reselect top descriptor features via RF importance and update state.json.

    Fits a RandomForest on currently-observed labeled molecules and picks the
    top `max_dims` features by importance.  Updates `design_parameters` and
    `active_features` in state.json so the next engine.suggest() call uses
    the reduced space.

    Returns the number of selected features (0 if skipped due to too few labels).
    """
    labeled_X: list[list[float]] = []
    labeled_y: list[float] = []
    for can in observed_canonical:
        if can not in pic50_lookup or can not in descriptor_lookup:
            continue
        labeled_X.append([descriptor_lookup[can].get(c, 0.0) for c in descriptor_cols])
        labeled_y.append(pic50_lookup[can])

    if len(labeled_X) < 5:
        return 0

    X = np.array(labeled_X, dtype=float)
    y = np.array(labeled_y, dtype=float)

    # Drop constant columns
    variable_mask = X.std(axis=0) > 0
    variable_cols = [c for c, v in zip(descriptor_cols, variable_mask) if v]
    X_var = X[:, variable_mask]

    if not variable_cols:
        return 0

    keep_n = min(max_dims, len(variable_cols))
    if len(variable_cols) > keep_n:
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1)
        rf.fit(X_var, y)
        top_idx = np.argsort(rf.feature_importances_)[::-1][:keep_n]
        selected = [variable_cols[i] for i in top_idx]
    else:
        selected = list(variable_cols)

    design_params = []
    selected_final = []
    for col in selected:
        col_idx = variable_cols.index(col)
        vals = X_var[:, col_idx]
        lb, ub = float(vals.min()), float(vals.max())
        if np.isclose(lb, ub):
            continue
        selected_final.append(col)
        design_params.append({"name": col, "type": "num", "lb": lb, "ub": ub})

    if not selected_final:
        return 0

    state = json.loads(state_path.read_text())
    state["active_features"] = selected_final
    state["design_parameters"] = design_params
    state_path.write_text(json.dumps(state, indent=2))

    if verbose:
        print(f"  Active features: {len(selected_final)} (reduced from {len(descriptor_cols)})")

    return len(selected_final)
