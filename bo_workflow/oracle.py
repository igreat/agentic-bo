"""Oracle training, loading, and prediction.

Standalone module — operates on run_dir: Path, not engine: BOEngine.
Reads/writes state.json and oracle files directly using RunPaths + utils.
"""

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .utils import Objective, RunPaths, read_json, utc_now_iso, write_json


# ------------------------------------------------------------------
# Objective-scale helpers
# ------------------------------------------------------------------


def _normalize_objective_values(
    values: np.ndarray, objective: Objective
) -> tuple[np.ndarray, float]:
    """Map objective values to internal minimization scale."""
    if objective == "min":
        return values.astype(float), float("nan")
    target_max = float(np.max(values))
    return (target_max - values).astype(float), target_max


def _restore_objective_values(
    values: np.ndarray, objective: Objective, target_max: float
) -> np.ndarray:
    """Restore internal minimization values back to user objective scale."""
    if objective == "min":
        return values
    return target_max - values


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, file=sys.stderr)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def build_proxy_oracle(
    run_dir: str | Path,
    *,
    model_candidates: tuple[str, ...] = ("random_forest", "extra_trees"),
    cv_folds: int = 5,
    max_features: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Train/select a proxy oracle and persist model + metadata."""
    run_dir = Path(run_dir)
    paths = RunPaths(run_dir=run_dir)
    state = read_json(paths.state)

    dataset = pd.read_csv(state["dataset_path"])
    _log(
        verbose,
        f"[oracle] run_dir={run_dir.name} rows={len(dataset)} cv_folds={cv_folds}",
    )

    target_column = state["target_column"]
    y_raw = pd.to_numeric(dataset[target_column], errors="coerce")
    valid_mask = y_raw.notna()
    if valid_mask.sum() < 5:
        raise ValueError("Need at least 5 non-null target rows to train an oracle.")

    active_features = list(state["active_features"])
    x_full = dataset.loc[valid_mask, active_features].copy()
    y_full = y_raw.loc[valid_mask].to_numpy(dtype=float)

    y_internal, target_max = _normalize_objective_values(y_full, state["objective"])

    if (
        max_features is not None
        and max_features > 0
        and len(active_features) > max_features
    ):
        x_for_importance = x_full.copy()
        for col in x_for_importance.columns:
            if pd.api.types.is_numeric_dtype(x_for_importance[col]):
                x_for_importance[col] = pd.to_numeric(
                    x_for_importance[col], errors="coerce"
                ).fillna(x_for_importance[col].median())
            else:
                codes, _ = pd.factorize(x_for_importance[col].astype(str), sort=True)
                x_for_importance[col] = codes

        selector = RandomForestRegressor(
            n_estimators=200, random_state=state["seed"], n_jobs=1
        )
        selector.fit(x_for_importance, y_internal)
        ranked = np.argsort(selector.feature_importances_)[::-1]
        keep_idx = ranked[:max_features]
        keep_features = [x_for_importance.columns[i] for i in keep_idx]
        ignored = [name for name in active_features if name not in keep_features]

        state["active_features"] = keep_features
        state["ignored_features"] = ignored
        if "original_design_parameters" not in state:
            state["original_design_parameters"] = list(state["design_parameters"])
        state["design_parameters"] = [
            p for p in state["design_parameters"] if p["name"] in set(keep_features)
        ]
        active_features = keep_features
        x_full = x_full[active_features]

    numeric_cols = [
        c for c in active_features if pd.api.types.is_numeric_dtype(x_full[c])
    ]
    categorical_cols = [c for c in active_features if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    model_pool: dict[str, Any] = {}
    if "random_forest" in model_candidates:
        model_pool["random_forest"] = RandomForestRegressor(
            n_estimators=200,
            random_state=state["seed"],
            n_jobs=1,
        )
    if "extra_trees" in model_candidates:
        model_pool["extra_trees"] = ExtraTreesRegressor(
            n_estimators=240,
            random_state=state["seed"],
            n_jobs=1,
        )
    if not model_pool:
        raise ValueError("No supported model candidates were provided.")

    n_rows = len(x_full)
    n_splits = min(max(2, cv_folds), n_rows)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=state["seed"])

    scores: dict[str, float] = {}
    trained_pipelines: dict[str, Pipeline] = {}
    for model_name, regressor in model_pool.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", regressor),
            ]
        )
        cv_scores = cross_val_score(
            pipeline,
            x_full,
            y_internal,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,
        )
        rmse = float(-np.mean(cv_scores))
        scores[model_name] = rmse
        trained_pipelines[model_name] = pipeline
        _log(verbose, f"[oracle] {model_name}: cv_rmse={rmse:.4f}")

    best_model_name = min(scores, key=lambda k: scores[k])
    best_pipeline = trained_pipelines[best_model_name]
    best_pipeline.fit(x_full, y_internal)

    paths.run_dir.mkdir(parents=True, exist_ok=True)
    with paths.oracle_model.open("wb") as handle:
        pickle.dump(best_pipeline, handle)

    oracle_meta = {
        "built_at": utc_now_iso(),
        "model_candidates": list(model_pool.keys()),
        "cv_rmse": scores,
        "selected_model": best_model_name,
        "selected_rmse": scores[best_model_name],
        "rows_used": int(n_rows),
        "active_features": list(active_features),
        "objective_internal": "min",
        "target_max_for_restore": target_max,
    }
    write_json(paths.oracle_meta, oracle_meta)

    state["objective_transform"] = {
        "internal_objective": "min",
        "target_max_for_restore": target_max,
    }

    state["oracle"] = oracle_meta
    state["status"] = "oracle_ready"
    state["updated_at"] = utc_now_iso()
    write_json(paths.state, state)
    _log(
        verbose,
        f"[oracle] selected={best_model_name} rmse={scores[best_model_name]:.4f}",
    )

    return {
        "run_id": state["run_id"],
        "status": state["status"],
        "active_features": state["active_features"],
        "ignored_features": state["ignored_features"],
        "selected_model": best_model_name,
        "selected_rmse": scores[best_model_name],
        "cv_rmse": scores,
    }


def load_oracle(run_dir: str | Path) -> Pipeline:
    """Load a previously persisted oracle pipeline from disk."""
    run_dir = Path(run_dir)
    paths = RunPaths(run_dir=run_dir)
    if not paths.oracle_model.exists():
        raise FileNotFoundError(
            f"Oracle not found at {paths.oracle_model}. Build it first with build-oracle."
        )
    with paths.oracle_model.open("rb") as handle:
        model = pickle.load(handle)
    return model


def predict_original_scale(
    run_dir: str | Path,
    state: dict[str, Any],
    x_df: pd.DataFrame,
) -> np.ndarray:
    """Run oracle prediction and map back to the user's objective scale."""
    model = load_oracle(run_dir)
    y_internal = np.asarray(model.predict(x_df), dtype=float)
    target_max = float(state["oracle"]["target_max_for_restore"])
    return _restore_objective_values(y_internal, state["objective"], target_max)
