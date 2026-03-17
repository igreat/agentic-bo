"""Oracle training, loading, and prediction."""

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from ..engine import _infer_design_parameters
from ..utils import EvaluationBackendPaths, read_json, utc_now_iso, write_json


# ------------------------------------------------------------------
# Objective-scale helpers
# ------------------------------------------------------------------


def _normalize_objective_values(values: np.ndarray, objective: str) -> np.ndarray:
    """Map objective values to the engine's internal minimization scale."""
    if objective == "min":
        return values.astype(float)
    if objective == "max":
        return (-values).astype(float)
    raise ValueError(f"Unknown objective: '{objective}'")


def _restore_objective_values(values: np.ndarray, objective: str) -> np.ndarray:
    """Restore internal minimization values back to the user objective scale."""
    if objective == "min":
        return values
    if objective == "max":
        return -values
    raise ValueError(f"Unknown objective: '{objective}'")


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, file=sys.stderr)


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3:
        return None
    val = spearmanr(a, b).statistic
    return None if (val is None or np.isnan(val)) else float(val)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def build_proxy_oracle(
    *,
    dataset_path: str | Path,
    target_column: str,
    objective: str,
    backend_dir: str | Path,
    drop_cols: list[str] | None = None,
    seed: int = 7,
    default_engine: str = "hebo",
    model_candidates: tuple[str, ...] = ("random_forest", "extra_trees", "gradient_boosting", "gaussian_process"),
    cv_folds: int = 5,
    max_features: int | None = None,
    top_k_pct: float = 3.0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Train/select a proxy oracle and persist model + metadata."""
    resolved_dataset = Path(dataset_path).resolve()
    if not resolved_dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {resolved_dataset}")
    if objective not in {"min", "max"}:
        raise ValueError("objective must be either 'min' or 'max'")
    if default_engine not in {"hebo", "bo_lcb", "random", "botorch"}:
        raise ValueError("default_engine must be one of: hebo, bo_lcb, random, botorch")

    backend_paths = EvaluationBackendPaths(backend_dir=Path(backend_dir))
    dataset = pd.read_csv(resolved_dataset)
    if target_column not in dataset.columns:
        raise ValueError(
            f"Target column '{target_column}' is not in dataset columns: {list(dataset.columns)}"
        )

    feature_frame = dataset.drop(columns=[target_column])
    if drop_cols:
        unknown = [c for c in drop_cols if c not in feature_frame.columns]
        if unknown:
            raise ValueError(f"--drop-cols contains unknown columns: {unknown}")
        feature_frame = feature_frame.drop(columns=drop_cols)

    design_params, fixed_features, dropped_features = _infer_design_parameters(
        feature_frame, max_categories=64
    )
    active_features = [p["name"] for p in design_params]
    ignored_features = [*list(drop_cols or []), *list(fixed_features.keys()), *dropped_features]

    _log(
        verbose,
        f"[oracle] dataset={resolved_dataset.name} rows={len(dataset)} cv_folds={cv_folds}",
    )

    y_raw = pd.to_numeric(dataset[target_column], errors="coerce")
    valid_mask = y_raw.notna()
    if valid_mask.sum() < 5:
        raise ValueError("Need at least 5 non-null target rows to train an oracle.")

    x_full = dataset.loc[valid_mask, active_features].copy()
    y_full = y_raw.loc[valid_mask].to_numpy(dtype=float)

    y_internal = _normalize_objective_values(y_full, objective)

    if max_features is not None and max_features > 0 and len(active_features) > max_features:
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
            n_estimators=200, random_state=seed, n_jobs=1
        )
        selector.fit(x_for_importance, y_internal)
        ranked = np.argsort(selector.feature_importances_)[::-1]
        keep_features = [x_for_importance.columns[i] for i in ranked[:max_features]]
        ignored = [name for name in active_features if name not in keep_features]

        active_features = keep_features
        ignored_features.extend(ignored)
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
                        ("imputer", SimpleImputer(strategy="constant", fill_value="no_value")),
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
            random_state=seed,
            n_jobs=1,
        )
    if "extra_trees" in model_candidates:
        model_pool["extra_trees"] = ExtraTreesRegressor(
            n_estimators=200,
            random_state=seed,
            n_jobs=1,
        )
    if "gradient_boosting" in model_candidates:
        model_pool["gradient_boosting"] = HistGradientBoostingRegressor(
            max_iter=200,
            random_state=seed,
        )
    if "gaussian_process" in model_candidates:
        kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3)
        model_pool["gaussian_process"] = GaussianProcessRegressor(
            kernel=kernel,
            random_state=seed,
            normalize_y=True,
        )
    if not model_pool:
        raise ValueError("No supported model candidates were provided.")

    n_rows = len(x_full)
    n_splits = min(max(2, cv_folds), n_rows)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Top-K% mask for Spearman selection (internal scale: lower = better)
    n_top_k = max(1, int(n_rows * top_k_pct / 100))
    top_k_threshold = float(np.partition(y_internal.copy(), n_top_k - 1)[n_top_k - 1])
    top_k_mask = y_internal <= top_k_threshold

    scores: dict[str, dict[str, float | None]] = {}
    trained_pipelines: dict[str, Pipeline] = {}

    for model_name, regressor in model_pool.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", regressor),
            ]
        )
        cv_preds = cross_val_predict(pipeline, x_full, y_internal, cv=cv)

        rmse = float(np.sqrt(np.mean((cv_preds - y_internal) ** 2)))
        sr_all = _safe_spearman(cv_preds, y_internal)
        sr_top_k = _safe_spearman(cv_preds[top_k_mask], y_internal[top_k_mask]) if top_k_mask.sum() >= 3 else None

        scores[model_name] = {
            "rmse": rmse,
            "spearman_all": sr_all,
            "spearman_top_k": sr_top_k,
        }
        trained_pipelines[model_name] = pipeline
        sr_all_str = f"{sr_all:.3f}" if sr_all is not None else "n/a"
        sr_top_k_str = f"{sr_top_k:.3f}" if sr_top_k is not None else "n/a"
        _log(
            verbose,
            f"[oracle] {model_name}: rmse={rmse:.4f} spearman_all={sr_all_str} spearman_top_k={sr_top_k_str}",
        )

    def _selection_key(name: str) -> tuple[float, float, float]:
        s = scores[name]
        # Higher Spearman is better → negate for min-sort. Lower RMSE is better → keep.
        top_k = s["spearman_top_k"] if s["spearman_top_k"] is not None else s["spearman_all"]
        top_k = top_k if top_k is not None else -1.0
        sr_all = s["spearman_all"] if s["spearman_all"] is not None else -1.0
        return (-top_k, -sr_all, s["rmse"])

    best_model_name = min(scores, key=_selection_key)
    best_pipeline = trained_pipelines[best_model_name]
    best_pipeline.fit(x_full, y_internal)

    backend_paths.backend_dir.mkdir(parents=True, exist_ok=True)
    with backend_paths.oracle_model.open("wb") as handle:
        pickle.dump(best_pipeline, handle)

    oracle_meta = {
        "backend_id": backend_paths.backend_dir.name,
        "built_at": utc_now_iso(),
        "target_column": target_column,
        "objective": objective,
        "default_engine": default_engine,
        "seed": int(seed),
        "model_candidates": list(model_pool.keys()),
        "scores": scores,
        "selected_model": best_model_name,
        "selected_rmse": scores[best_model_name]["rmse"],
        "selected_spearman_all": scores[best_model_name]["spearman_all"],
        "selected_spearman_top_k": scores[best_model_name]["spearman_top_k"],
        "top_k_pct": top_k_pct,
        "rows_used": int(n_rows),
        "active_features": list(active_features),
        "ignored_features": list(dict.fromkeys(ignored_features)),
        "objective_internal": "min",
    }
    write_json(backend_paths.oracle_meta, oracle_meta)
    _log(
        verbose,
        f"[oracle] selected={best_model_name} rmse={scores[best_model_name]['rmse']:.4f}",
    )

    return {
        "backend_id": backend_paths.backend_dir.name,
        "backend_dir": str(backend_paths.backend_dir),
        "dataset_path": str(resolved_dataset),
        "target_column": target_column,
        "objective": objective,
        "active_features": list(active_features),
        "ignored_features": list(dict.fromkeys(ignored_features)),
        "selected_model": best_model_name,
        "scores": scores,
    }


def read_backend_meta(backend_dir: str | Path) -> dict[str, Any]:
    """Load oracle backend metadata from disk."""
    backend_paths = EvaluationBackendPaths(backend_dir=Path(backend_dir))
    if not backend_paths.oracle_meta.exists():
        raise FileNotFoundError(
            f"Oracle metadata not found at {backend_paths.oracle_meta}. Build it first with build-oracle."
        )
    return read_json(backend_paths.oracle_meta)


def load_oracle(backend_dir: str | Path) -> Pipeline:
    """Load a previously persisted oracle pipeline from disk."""
    backend_paths = EvaluationBackendPaths(backend_dir=Path(backend_dir))
    if not backend_paths.oracle_model.exists():
        raise FileNotFoundError(
            f"Oracle not found at {backend_paths.oracle_model}. Build it first with build-oracle."
        )
    with backend_paths.oracle_model.open("rb") as handle:
        model = pickle.load(handle)
    return model


def predict_original_scale(
    backend_dir: str | Path,
    objective: str,
    x_df: pd.DataFrame,
) -> np.ndarray:
    """Run oracle prediction and map back to the user's objective scale."""
    model = load_oracle(backend_dir)
    y_internal = np.asarray(model.predict(x_df), dtype=float)
    return _restore_objective_values(y_internal, objective)
