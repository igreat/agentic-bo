from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def detect_smiles_column(df: pd.DataFrame, hint: str | None = None) -> str:
    """Auto-detect which column in *df* contains SMILES strings."""
    from rdkit import Chem

    def _smiles_hit_rate(series: pd.Series) -> float:
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return 0.0
        hits = sum(1 for v in sample if Chem.MolFromSmiles(str(v)) is not None)
        return hits / len(sample)

    if hint is not None:
        if hint not in df.columns:
            raise ValueError(
                f"Specified SMILES column '{hint}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        rate = _smiles_hit_rate(df[hint])
        if rate >= 0.5:
            return hint
        raise ValueError(
            f"Column '{hint}' does not appear to contain SMILES "
            f"(only {rate:.0%} parsed successfully)."
        )

    smiles_names = {"smiles", "smi", "molecule", "mol", "structure", "canonical_smiles"}
    candidates: list[tuple[str, float]] = []

    for col in df.columns:
        if col.lower().strip() in smiles_names:
            rate = _smiles_hit_rate(df[col])
            if rate >= 0.7:
                return col
            candidates.append((col, rate))

    for col in df.select_dtypes(include=["object", "string"]).columns:
        if col.lower().strip() in smiles_names:
            continue
        rate = _smiles_hit_rate(df[col])
        if rate >= 0.7:
            candidates.append((col, rate))

    if candidates:
        best = max(candidates, key=lambda x: x[1])
        if best[1] >= 0.5:
            return best[0]

    raise ValueError(
        "Cannot auto-detect a SMILES column. Please specify --smiles-column. "
        f"Columns checked: {list(df.columns)}"
    )


def infer_design_parameters(
    frame: pd.DataFrame,
    *,
    max_categories: int = 64,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """Infer HEBO design parameters from a feature frame."""
    params: list[dict[str, Any]] = []
    fixed_features: dict[str, Any] = {}
    dropped_features: list[str] = []

    for col in frame.columns:
        series = frame[col]

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                dropped_features.append(col)
                continue
            lb = float(numeric.min())
            ub = float(numeric.max())
            if np.isclose(lb, ub):
                fixed_features[col] = lb
                continue
            params.append({"name": col, "type": "num", "lb": lb, "ub": ub})
            continue

        categories = sorted({str(v) for v in series.dropna().tolist()})
        if not categories:
            dropped_features.append(col)
            continue
        if len(categories) == 1:
            fixed_features[col] = categories[0]
            continue
        if len(categories) > max_categories:
            raise ValueError(
                f"Feature '{col}' has {len(categories)} categories; max supported is {max_categories}."
            )
        params.append({"name": col, "type": "cat", "categories": categories})

    if not params:
        raise ValueError("No optimizable features were inferred from the dataset.")

    return params, fixed_features, dropped_features
