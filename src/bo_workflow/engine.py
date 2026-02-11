from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import pickle
import secrets
from typing import Any, Literal

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .plotting import plot_optimization_convergence

Objective = Literal["min", "max"]

_RUN_ADJECTIVES = (
    "amber",
    "brisk",
    "crisp",
    "daring",
    "eager",
    "fuzzy",
    "gentle",
    "jolly",
    "lively",
    "nimble",
    "rapid",
    "steady",
    "sunny",
    "vivid",
)

_RUN_NOUNS = (
    "otter",
    "falcon",
    "heron",
    "lynx",
    "fox",
    "orca",
    "panda",
    "sparrow",
    "badger",
    "koala",
    "wolf",
    "tiger",
    "eagle",
    "whale",
)


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _generate_run_id() -> str:
    adjective = secrets.choice(_RUN_ADJECTIVES)
    noun = secrets.choice(_RUN_NOUNS)
    suffix = secrets.randbelow(10000)
    return f"{adjective}-{noun}-{suffix:04d}"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _row_to_python_dict(row: pd.Series) -> dict[str, Any]:
    return {str(k): _to_python_scalar(v) for k, v in row.to_dict().items()}


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path

    @property
    def state(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def intent(self) -> Path:
        return self.run_dir / "intent.json"

    @property
    def input_spec(self) -> Path:
        return self.run_dir / "input_spec.json"

    @property
    def suggestions(self) -> Path:
        return self.run_dir / "suggestions.jsonl"

    @property
    def observations(self) -> Path:
        return self.run_dir / "observations.jsonl"

    @property
    def oracle_model(self) -> Path:
        return self.run_dir / "oracle.pkl"

    @property
    def oracle_meta(self) -> Path:
        return self.run_dir / "oracle_meta.json"

    @property
    def report(self) -> Path:
        return self.run_dir / "report.json"

    @property
    def convergence_plot(self) -> Path:
        return self.run_dir / "convergence.pdf"


def _infer_design_parameters(
    frame: pd.DataFrame,
    *,
    max_categories: int = 64,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
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


def _normalize_objective_values(
    values: np.ndarray, objective: Objective
) -> tuple[np.ndarray, float]:
    if objective == "min":
        return values.astype(float), float("nan")
    target_max = float(np.max(values))
    return (target_max - values).astype(float), target_max


def _restore_objective_values(
    values: np.ndarray, objective: Objective, target_max: float
) -> np.ndarray:
    if objective == "min":
        return values
    return target_max - values


class BOEngine:
    def __init__(self, runs_root: str | Path = "runs") -> None:
        self.runs_root = Path(runs_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def _paths(self, run_id: str) -> RunPaths:
        return RunPaths(run_dir=self.runs_root / run_id)

    def _load_state(self, run_id: str) -> dict[str, Any]:
        paths = self._paths(run_id)
        if not paths.state.exists():
            raise FileNotFoundError(f"Run '{run_id}' not found at {paths.run_dir}")
        return _read_json(paths.state)

    def _save_state(self, run_id: str, state: dict[str, Any]) -> None:
        _write_json(self._paths(run_id).state, state)

    def init_run(
        self,
        *,
        dataset_path: str | Path,
        target_column: str,
        objective: Objective,
        run_id: str | None = None,
        num_initial_random_samples: int = 10,
        default_batch_size: int = 1,
        seed: int = 0,
        max_categories: int = 64,
        intent: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if objective not in {"min", "max"}:
            raise ValueError("objective must be either 'min' or 'max'")

        dataset_path = Path(dataset_path).resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        data = pd.read_csv(dataset_path)
        if target_column not in data.columns:
            raise ValueError(
                f"Target column '{target_column}' is not in dataset columns: {list(data.columns)}"
            )

        feature_frame = data.drop(columns=[target_column])
        design_params, fixed_features, dropped_features = _infer_design_parameters(
            feature_frame,
            max_categories=max_categories,
        )

        if run_id is None:
            for _ in range(20):
                candidate = _generate_run_id()
                if not self._paths(candidate).state.exists():
                    run_id = candidate
                    break
            if run_id is None:
                raise RuntimeError("Failed to generate a unique run_id after retries.")
        elif self._paths(run_id).state.exists():
            raise ValueError(
                f"Run '{run_id}' already exists. Provide a different --run-id or omit it."
            )

        numeric_target = pd.to_numeric(data[target_column], errors="coerce")
        if objective == "max":
            if numeric_target.notna().sum() == 0:
                raise ValueError(
                    "Target column has no numeric values required for max objective transform."
                )
            target_max_for_restore = float(numeric_target.max())
        else:
            target_max_for_restore = float("nan")

        state = {
            "run_id": run_id,
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "status": "initialized",
            "dataset_path": str(dataset_path),
            "target_column": target_column,
            "objective": objective,
            "seed": int(seed),
            "num_initial_random_samples": int(num_initial_random_samples),
            "default_batch_size": int(default_batch_size),
            "design_parameters": design_params,
            "active_features": [p["name"] for p in design_params],
            "fixed_features": fixed_features,
            "dropped_features": dropped_features,
            "ignored_features": [],
            "oracle": None,
            "objective_transform": {
                "internal_objective": "min",
                "target_max_for_restore": target_max_for_restore,
            },
        }
        self._save_state(run_id, state)
        if intent is not None:
            intent_payload = {
                "run_id": run_id,
                "created_at": _utc_now_iso(),
                "intent": intent,
                "resolved": {
                    "dataset_path": str(dataset_path),
                    "target_column": target_column,
                    "objective": objective,
                    "seed": int(seed),
                    "num_initial_random_samples": int(num_initial_random_samples),
                    "default_batch_size": int(default_batch_size),
                    "max_categories": int(max_categories),
                },
            }
            _write_json(self._paths(run_id).intent, intent_payload)
        return state

    def init_from_spec(
        self,
        spec: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        required = {"dataset_path", "target_column", "objective"}
        missing = sorted(required - set(spec.keys()))
        if missing:
            raise ValueError(f"Spec missing required keys: {', '.join(missing)}")

        state = self.init_run(
            dataset_path=spec["dataset_path"],
            target_column=spec["target_column"],
            objective=spec["objective"],
            run_id=run_id or spec.get("run_id"),
            num_initial_random_samples=int(spec.get("num_initial_random_samples", 10)),
            default_batch_size=int(spec.get("default_batch_size", 1)),
            seed=int(spec.get("seed", 0)),
            max_categories=int(spec.get("max_categories", 64)),
            intent=spec.get("intent"),
        )
        _write_json(self._paths(state["run_id"]).input_spec, spec)
        return state

    def build_oracle(
        self,
        run_id: str,
        *,
        model_candidates: tuple[str, ...] = ("random_forest", "extra_trees"),
        cv_folds: int = 5,
        max_features: int | None = None,
    ) -> dict[str, Any]:
        state = self._load_state(run_id)
        dataset = pd.read_csv(state["dataset_path"])

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
                    codes, _ = pd.factorize(
                        x_for_importance[col].astype(str), sort=True
                    )
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

        best_model_name = min(scores, key=scores.get)
        best_pipeline = trained_pipelines[best_model_name]
        best_pipeline.fit(x_full, y_internal)

        paths = self._paths(run_id)
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        with paths.oracle_model.open("wb") as handle:
            pickle.dump(best_pipeline, handle)

        oracle_meta = {
            "built_at": _utc_now_iso(),
            "model_candidates": list(model_pool.keys()),
            "cv_rmse": scores,
            "selected_model": best_model_name,
            "selected_rmse": scores[best_model_name],
            "rows_used": int(n_rows),
            "active_features": list(active_features),
            "objective_internal": "min",
            "target_max_for_restore": target_max,
        }
        _write_json(paths.oracle_meta, oracle_meta)

        state["objective_transform"] = {
            "internal_objective": "min",
            "target_max_for_restore": target_max,
        }

        state["oracle"] = oracle_meta
        state["status"] = "oracle_ready"
        state["updated_at"] = _utc_now_iso()
        self._save_state(run_id, state)

        return {
            "run_id": run_id,
            "status": state["status"],
            "active_features": state["active_features"],
            "ignored_features": state["ignored_features"],
            "selected_model": best_model_name,
            "selected_rmse": scores[best_model_name],
            "cv_rmse": scores,
        }

    def _build_optimizer(
        self, state: dict[str, Any], observations: list[dict[str, Any]]
    ) -> HEBO:
        np.random.seed(int(state["seed"]))
        design_space = DesignSpace().parse(state["design_parameters"])
        optimizer = HEBO(
            design_space,
            model_name="gp",
            rand_sample=int(state["num_initial_random_samples"]),
            scramble_seed=int(state["seed"]),
        )

        if observations:
            x_rows = [
                {k: row["x"][k] for k in state["active_features"]}
                for row in observations
            ]
            x_obs = pd.DataFrame(x_rows)
            y_obs = np.array(
                [float(row["y_internal"]) for row in observations], dtype=float
            ).reshape(-1, 1)
            optimizer.observe(x_obs, y_obs)
        return optimizer

    def suggest(self, run_id: str, *, batch_size: int | None = None) -> dict[str, Any]:
        state = self._load_state(run_id)
        if state["status"] not in {"oracle_ready", "running"}:
            raise ValueError(
                f"Run '{run_id}' is not ready for suggestions. Current status: {state['status']}"
            )

        size = int(batch_size or state["default_batch_size"])
        observations = _read_jsonl(self._paths(run_id).observations)
        optimizer = self._build_optimizer(state, observations)
        proposals = optimizer.suggest(n_suggestions=size)

        rows = []
        for _, row in proposals.iterrows():
            x = _row_to_python_dict(row)
            x.update(state["fixed_features"])

            payload = {
                "event_time": _utc_now_iso(),
                "suggestion_id": secrets.token_hex(16),
                "iteration": len(observations),
                "x": x,
            }
            _append_jsonl(self._paths(run_id).suggestions, payload)
            rows.append(payload)

        state["status"] = "running"
        state["updated_at"] = _utc_now_iso()
        self._save_state(run_id, state)

        return {
            "run_id": run_id,
            "num_suggestions": len(rows),
            "suggestions": rows,
        }

    def _load_oracle(self, run_id: str) -> Pipeline:
        paths = self._paths(run_id)
        if not paths.oracle_model.exists():
            raise FileNotFoundError(
                f"Oracle for run '{run_id}' not found. Build it first with build-oracle."
            )
        with paths.oracle_model.open("rb") as handle:
            model = pickle.load(handle)
        return model

    def _oracle_predict_original_scale(
        self, run_id: str, state: dict[str, Any], x_df: pd.DataFrame
    ) -> np.ndarray:
        model = self._load_oracle(run_id)
        y_internal = np.asarray(model.predict(x_df), dtype=float)
        target_max = float(state["oracle"]["target_max_for_restore"])
        return _restore_objective_values(y_internal, state["objective"], target_max)

    def observe(
        self,
        run_id: str,
        observations: list[dict[str, Any]],
        *,
        source: str = "user",
    ) -> dict[str, Any]:
        state = self._load_state(run_id)
        if not observations:
            raise ValueError("No observations provided.")

        target_col = state["target_column"]
        existing = _read_jsonl(self._paths(run_id).observations)
        next_iteration = len(existing)
        rows = []

        for idx, obs in enumerate(observations):
            x = dict(obs.get("x", {}))
            for feature in state["active_features"]:
                if feature not in x:
                    if feature in state["fixed_features"]:
                        x[feature] = state["fixed_features"][feature]
                    else:
                        raise ValueError(
                            f"Observation missing required feature '{feature}'."
                        )

            y_value = obs.get("y", obs.get(target_col))
            if y_value is None:
                raise ValueError(
                    f"Observation missing objective value. Provide 'y' or '{target_col}'."
                )
            y_float = float(y_value)

            if state["objective"] == "min":
                y_internal = y_float
            else:
                transform = state.get("objective_transform") or {}
                target_max_for_restore = transform.get("target_max_for_restore")
                if target_max_for_restore is None:
                    raise ValueError(
                        "Missing target_max_for_restore for max objective. Reinitialize run."
                    )
                y_internal = float(target_max_for_restore) - y_float

            payload = {
                "event_time": _utc_now_iso(),
                "iteration": next_iteration + idx,
                "source": source,
                "x": {k: _to_python_scalar(v) for k, v in x.items()},
                "y": y_float,
                "y_internal": y_internal,
            }
            _append_jsonl(self._paths(run_id).observations, payload)
            rows.append(payload)

        state["updated_at"] = _utc_now_iso()
        self._save_state(run_id, state)
        return {"run_id": run_id, "recorded": len(rows), "observations": rows}

    def evaluate_last_suggestions(
        self, run_id: str, *, max_new: int | None = None
    ) -> dict[str, Any]:
        state = self._load_state(run_id)
        if state.get("oracle") is None:
            raise ValueError(
                f"Oracle not built for run '{run_id}'. Run build-oracle first."
            )
        suggestions = _read_jsonl(self._paths(run_id).suggestions)
        observations = _read_jsonl(self._paths(run_id).observations)

        already_seen = {json.dumps(row["x"], sort_keys=True) for row in observations}
        pending = [
            s
            for s in suggestions
            if json.dumps(s["x"], sort_keys=True) not in already_seen
        ]
        if max_new is not None:
            pending = pending[:max_new]

        if not pending:
            return {"run_id": run_id, "evaluated": 0, "observations": []}

        x_df = pd.DataFrame([row["x"] for row in pending])[state["active_features"]]
        y_pred = self._oracle_predict_original_scale(run_id, state, x_df)

        payloads = []
        for row, y_val in zip(pending, y_pred, strict=True):
            payloads.append({"x": row["x"], "y": float(y_val)})

        observed = self.observe(run_id, payloads, source="proxy-oracle")
        return {
            "run_id": run_id,
            "evaluated": observed["recorded"],
            "observations": observed["observations"],
        }

    def run_proxy_optimization(
        self,
        run_id: str,
        *,
        num_iterations: int,
        batch_size: int = 1,
    ) -> dict[str, Any]:
        for _ in range(num_iterations):
            self.suggest(run_id, batch_size=batch_size)
            self.evaluate_last_suggestions(run_id, max_new=batch_size)
        state = self._load_state(run_id)
        state["status"] = "completed"
        state["updated_at"] = _utc_now_iso()
        self._save_state(run_id, state)
        return self.report(run_id)

    def status(self, run_id: str) -> dict[str, Any]:
        state = self._load_state(run_id)
        observations = _read_jsonl(self._paths(run_id).observations)

        payload: dict[str, Any] = {
            "run_id": run_id,
            "status": state["status"],
            "objective": state["objective"],
            "target_column": state["target_column"],
            "active_features": state["active_features"],
            "ignored_features": state["ignored_features"],
            "num_observations": len(observations),
        }
        if observations:
            y_values = np.asarray(
                [float(row["y"]) for row in observations], dtype=float
            )
            if state["objective"] == "min":
                best_idx = int(np.argmin(y_values))
                best_value = float(np.min(y_values))
            else:
                best_idx = int(np.argmax(y_values))
                best_value = float(np.max(y_values))
            payload["best_value"] = best_value
            payload["best_iteration"] = best_idx
            payload["best_x"] = observations[best_idx]["x"]

        if state["oracle"] is not None:
            payload["oracle"] = {
                "selected_model": state["oracle"]["selected_model"],
                "selected_rmse": state["oracle"]["selected_rmse"],
            }
        return payload

    def report(self, run_id: str) -> dict[str, Any]:
        state = self._load_state(run_id)
        observations = _read_jsonl(self._paths(run_id).observations)
        if not observations:
            report = {
                "run_id": run_id,
                "message": "No observations recorded yet.",
                "generated_at": _utc_now_iso(),
            }
            _write_json(self._paths(run_id).report, report)
            return report

        y_values = np.asarray([float(row["y"]) for row in observations], dtype=float)
        history = y_values.reshape(1, -1)

        plot_optimization_convergence(
            {"HEBO": history},
            title=f"Run {run_id}",
            ylabel=state["target_column"],
            objective=state["objective"],
            fig_path=str(self._paths(run_id).convergence_plot),
            show=False,
        )

        status = self.status(run_id)
        report = {
            "run_id": run_id,
            "generated_at": _utc_now_iso(),
            "num_observations": len(observations),
            "objective": state["objective"],
            "target_column": state["target_column"],
            "best_value": status.get("best_value"),
            "best_iteration": status.get("best_iteration"),
            "best_x": status.get("best_x"),
            "oracle": status.get("oracle"),
            "artifacts": {
                "plot": str(self._paths(run_id).convergence_plot),
                "state": str(self._paths(run_id).state),
                "observations": str(self._paths(run_id).observations),
            },
        }
        _write_json(self._paths(run_id).report, report)
        return report
