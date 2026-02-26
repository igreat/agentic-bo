from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def to_pic50(ic50_nm: float) -> float:
    if ic50_nm <= 0:
        raise ValueError("ic50_nM must be > 0")
    return 9.0 - math.log10(float(ic50_nm))


def is_reasonable_seed_smiles(smiles: str) -> bool:
    """Basic med-chem seed filter: exclude salts/multi-fragments/inorganics."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    if "." in Chem.MolToSmiles(mol):
        return False

    allowed = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    if not atoms:
        return False
    if any(z not in allowed for z in atoms):
        return False

    heavy = mol.GetNumHeavyAtoms()
    if heavy < 8 or heavy > 90:
        return False

    if 6 not in atoms:
        return False

    return True


def load_full_dataset(
    dataset_path: Path,
    target_column: str,
    *,
    max_pic50: float | None,
    fix_tiny_ic50_as_molar: bool,
) -> list[tuple[str, str, float]]:
    with open(dataset_path) as f:
        rows = list(csv.DictReader(f))

    data: list[tuple[str, str, float]] = []
    dropped_extreme = 0
    fixed_tiny = 0
    for row in rows:
        smi = row.get("smiles", "")
        can = canonicalize_smiles(smi)
        if not can:
            continue

        if target_column == "pIC50":
            y = float(row["pIC50"])
        else:
            ic50_nm = float(row["ic50_nM"])
            if fix_tiny_ic50_as_molar and ic50_nm < 1e-6:
                ic50_nm = ic50_nm * 1e9
                fixed_tiny += 1
            y = to_pic50(ic50_nm)

        if max_pic50 is not None and y > max_pic50:
            dropped_extreme += 1
            continue

        data.append((smi, can, y))

    if not data:
        raise ValueError("No valid molecules found in dataset")

    if fixed_tiny > 0:
        print(
            f"Data quality: converted {fixed_tiny} tiny ic50_nM values (<1e-6) from M to nM before pIC50"
        )
    if dropped_extreme > 0:
        print(
            f"Data quality: dropped {dropped_extreme} rows with pIC50 > {max_pic50}"
        )

    return data


def reselect_active_features(
    *,
    state_path: Path,
    train_csv: Path,
    desc_lookup_path: Path,
    seed: int,
    max_dims: int = 15,
    verbose: bool = False,
) -> int:
    """Reselect top descriptor features from currently observed labeled data."""
    state = json.loads(state_path.read_text())
    desc_lookup = json.loads(desc_lookup_path.read_text())
    train_df = pd.read_csv(train_csv)

    target_col = str(state.get("target_column", "pIC50"))
    smiles_col = str(state.get("smiles_direct", {}).get("smiles_column", "smiles"))
    if target_col not in train_df.columns or smiles_col not in train_df.columns:
        return 0

    y_series = pd.to_numeric(train_df[target_col], errors="coerce")
    valid_rows: list[dict[str, float]] = []
    y_vals: list[float] = []
    for smi, y in zip(train_df[smiles_col].astype(str), y_series):
        if pd.isna(y):
            continue
        row_desc = desc_lookup.get(smi)
        if row_desc is None:
            continue
        valid_rows.append(row_desc)
        y_vals.append(float(y))

    if len(valid_rows) < 5:
        return 0

    x_imp = pd.DataFrame(valid_rows)
    variable_cols: list[str] = []
    for col in x_imp.columns:
        vals = pd.to_numeric(x_imp[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        if np.isclose(float(vals.min()), float(vals.max())):
            continue
        variable_cols.append(str(col))

    if not variable_cols:
        return 0

    x_var = x_imp[variable_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_arr = np.asarray(y_vals, dtype=float)

    keep_n = min(max_dims, len(variable_cols))
    if len(variable_cols) > keep_n:
        rf = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1)
        rf.fit(x_var, y_arr)
        top_idx = np.argsort(rf.feature_importances_)[::-1][:keep_n]
        selected = [variable_cols[i] for i in top_idx]
    else:
        selected = list(variable_cols)

    design_params: list[dict[str, float | str]] = []
    selected_final: list[str] = []
    for col in selected:
        vals = pd.to_numeric(x_var[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        lb = float(vals.min())
        ub = float(vals.max())
        if np.isclose(lb, ub):
            continue
        selected_final.append(col)
        design_params.append({"name": col, "type": "num", "lb": lb, "ub": ub})

    if not selected_final:
        return 0

    state["active_features"] = selected_final
    state["design_parameters"] = design_params
    state["updated_at"] = pd.Timestamp.utcnow().isoformat()
    state_path.write_text(json.dumps(state, indent=2))

    if verbose:
        n_energy = sum(1 for f in selected_final if str(f).startswith("dft_"))
        print(f"  Reselected {len(selected_final)} active features (energy={n_energy})")

    return len(selected_final)


def load_seed_rows(
    seed_csv: Path,
    smiles_column: str,
    target_column: str,
) -> list[tuple[str, float]]:
    frame = pd.read_csv(seed_csv)
    if smiles_column not in frame.columns:
        raise ValueError(f"Missing smiles column '{smiles_column}' in {seed_csv}")
    if target_column not in frame.columns:
        raise ValueError(f"Missing target column '{target_column}' in {seed_csv}")

    rows: list[tuple[str, float]] = []
    for _, row in frame.iterrows():
        smi = str(row[smiles_column]).strip()
        if not smi:
            continue
        can = canonicalize_smiles(smi)
        if can is None:
            continue

        y_raw = row[target_column]
        if pd.isna(y_raw):
            continue
        rows.append((can, float(y_raw)))

    if len(rows) < 5:
        raise ValueError("Need at least 5 valid labeled seed rows.")
    return rows


def load_candidate_smiles(candidate_csv: Path, smiles_column: str) -> list[str]:
    frame = pd.read_csv(candidate_csv)
    if smiles_column not in frame.columns:
        raise ValueError(f"Missing smiles column '{smiles_column}' in {candidate_csv}")

    smiles: list[str] = []
    seen: set[str] = set()
    for value in frame[smiles_column].dropna().tolist():
        smi = str(value).strip()
        if not smi:
            continue
        can = canonicalize_smiles(smi)
        if can is None or can in seen:
            continue
        seen.add(can)
        smiles.append(can)
    return smiles


def build_runtime_dataset(
    out_csv: Path,
    seed_rows: list[tuple[str, float]],
    candidate_smiles: list[str],
) -> dict[str, int]:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    seed_map = {smi: y for smi, y in seed_rows}
    all_smiles = list(seed_map.keys())
    for smi in candidate_smiles:
        if smi not in seed_map:
            all_smiles.append(smi)

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles", "pIC50"])
        for smi in all_smiles:
            y = seed_map.get(smi)
            writer.writerow([smi, "" if y is None else y])

    return {
        "seed_count": len(seed_map),
        "candidate_count": len(candidate_smiles),
        "total_pool": len(all_smiles),
    }


def parse_observation_rows(
    obs_path: Path,
    target_column: str,
) -> list[dict[str, Any]]:
    frame = pd.read_csv(obs_path)

    required = {"smiles", target_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Observation CSV missing columns: {sorted(missing)}")

    has_suggestion_id = "suggestion_id" in frame.columns

    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        smi = str(row["smiles"]).strip()
        can = canonicalize_smiles(smi)
        if can is None:
            continue

        y_raw = row[target_column]
        if pd.isna(y_raw):
            continue

        rec: dict[str, Any] = {
            "x": {"smiles": can},
            "y": float(y_raw),
        }
        if has_suggestion_id and pd.notna(row["suggestion_id"]):
            rec["suggestion_id"] = str(row["suggestion_id"])
        rows.append(rec)

    if not rows:
        raise ValueError("No valid observations parsed from CSV.")
    return rows