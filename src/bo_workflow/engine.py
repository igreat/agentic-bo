"""Core BO engine.

This module keeps optimization state on disk (`runs/<run_id>/`) and rebuilds
optimizers from logged observations when needed. That replay-first design keeps
the workflow resumable and robust for human-in-the-loop usage.
"""

from pathlib import Path
import secrets
import sys
from typing import Any

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.bo import BO
from hebo.optimizers.hebo import HEBO
import numpy as np
import pandas as pd
from tqdm import tqdm

from .plotting import plot_optimization_convergence
from .utils import (
    Objective,
    OptimizerName,
    RunPaths,
    append_jsonl,
    generate_run_id,
    read_json,
    read_jsonl,
    row_to_python_dict,
    to_python_scalar,
    utc_now_iso,
    write_json,
)


def _infer_design_parameters(
    frame: pd.DataFrame,
    *,
    max_categories: int = 64,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """Infer HEBO design parameters from a feature frame.

    Returns a tuple of:
    - optimizable parameters
    - fixed features (constant columns)
    - dropped features (empty/unusable columns)
    """
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


class BOEngine:
    """Dataset-driven Bayesian optimization engine with persisted run state."""

    def __init__(self, runs_root: str | Path = "runs") -> None:
        self.runs_root = Path(runs_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def _paths(self, run_id: str) -> RunPaths:
        return RunPaths(run_dir=self.runs_root / run_id)

    def _load_state(self, run_id: str) -> dict[str, Any]:
        paths = self._paths(run_id)
        if not paths.state.exists():
            raise FileNotFoundError(f"Run '{run_id}' not found at {paths.run_dir}")
        return read_json(paths.state)

    def _save_state(self, run_id: str, state: dict[str, Any]) -> None:
        write_json(self._paths(run_id).state, state)

    def _log(self, verbose: bool, message: str) -> None:
        if verbose:
            print(message, file=sys.stderr)

    def init_run(
        self,
        *,
        dataset_path: str | Path,
        target_column: str,
        objective: Objective,
        default_engine: OptimizerName = "hebo",
        run_id: str | None = None,
        num_initial_random_samples: int = 10,
        default_batch_size: int = 1,
        seed: int = 7,
        max_categories: int = 64,
        intent: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Initialize a run from a dataset and inferred design space."""
        if objective not in {"min", "max"}:
            raise ValueError("objective must be either 'min' or 'max'")
        if default_engine not in {"hebo", "bo_lcb", "random"}:
            raise ValueError("default_engine must be one of: hebo, bo_lcb, random")

        dataset_path = Path(dataset_path).resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        data = pd.read_csv(dataset_path)
        self._log(verbose, f"[init] dataset={dataset_path} rows={len(data)}")
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
                candidate = generate_run_id()
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
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "status": "initialized",
            "dataset_path": str(dataset_path),
            "target_column": target_column,
            "objective": objective,
            "default_engine": default_engine,
            "seed": int(seed),
            "num_initial_random_samples": int(num_initial_random_samples),
            "default_batch_size": int(default_batch_size),
            "design_parameters": design_params,
            "active_features": [p["name"] for p in design_params],
            "fixed_features": fixed_features,
            "dropped_features": dropped_features,
            "ignored_features": [],
            "objective_transform": {
                "internal_objective": "min",
                "target_max_for_restore": target_max_for_restore,
            },
        }
        self._save_state(run_id, state)
        self._log(
            verbose,
            f"[init] run_id={run_id} engine={default_engine} features={len(state['active_features'])}",
        )
        if intent is not None:
            intent_payload = {
                "run_id": run_id,
                "created_at": utc_now_iso(),
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
            write_json(self._paths(run_id).intent, intent_payload)
        return state

    def _build_optimizer(
        self,
        state: dict[str, Any],
        observations: list[dict[str, Any]],
        engine_name: OptimizerName,
    ) -> HEBO | BO:
        """Build optimizer from replayed observation history.

        We reconstruct from history (instead of keeping in-memory optimizer
        state) so commands remain resumable and deterministic from run files.
        """
        np.random.seed(int(state["seed"]) + len(observations))
        design_space = DesignSpace().parse(state["design_parameters"])
        if engine_name == "hebo":
            optimizer: HEBO | BO = HEBO(
                design_space,
                model_name="gp",
                rand_sample=int(state["num_initial_random_samples"]),
                scramble_seed=int(state["seed"]),
            )
            if observations:
                # Replay Sobol sequence position to match previous suggestions.
                optimizer.sobol.fast_forward(len(observations))
        elif engine_name == "bo_lcb":
            optimizer = BO(
                design_space,
                model_name="gp",
                rand_sample=int(state["num_initial_random_samples"]),
            )
        else:
            raise ValueError(f"Unsupported optimizer engine '{engine_name}'")

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

    def suggest(
        self,
        run_id: str,
        *,
        batch_size: int | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        state = self._load_state(run_id)
        if state["status"] not in {"initialized", "oracle_ready", "running"}:
            raise ValueError(
                f"Run '{run_id}' is not ready for suggestions. Current status: {state['status']}"
            )

        engine = str(state.get("default_engine", "hebo"))
        if engine not in {"hebo", "bo_lcb", "random"}:
            raise ValueError("default_engine must be one of: hebo, bo_lcb, random")
        engine_typed: OptimizerName = engine  # type: ignore[assignment]  # validated above

        size = int(batch_size or state["default_batch_size"])
        observations = read_jsonl(self._paths(run_id).observations)
        if engine_typed == "random":
            np.random.seed(int(state["seed"]) + len(observations))
            proposals = DesignSpace().parse(state["design_parameters"]).sample(size)
        else:
            if engine_typed == "bo_lcb" and size != 1:
                raise ValueError("bo_lcb currently supports batch-size=1 only.")
            optimizer = self._build_optimizer(state, observations, engine_typed)
            proposals = optimizer.suggest(n_suggestions=size)

        rows = []
        for _, row in proposals.iterrows():
            x = row_to_python_dict(row)
            x.update(state["fixed_features"])

            payload = {
                "event_time": utc_now_iso(),
                "suggestion_id": secrets.token_hex(16),
                "iteration": len(observations),
                "engine": engine_typed,
                "x": x,
            }
            append_jsonl(self._paths(run_id).suggestions, payload)
            rows.append(payload)

        state["status"] = "running"
        state["updated_at"] = utc_now_iso()
        self._save_state(run_id, state)
        self._log(
            verbose,
            f"[suggest] run_id={run_id} engine={engine_typed} n={len(rows)}",
        )

        return {
            "run_id": run_id,
            "engine": engine_typed,
            "num_suggestions": len(rows),
            "suggestions": rows,
        }

    def observe(
        self,
        run_id: str,
        observations: list[dict[str, Any]],
        *,
        source: str = "user",
        verbose: bool = False,
    ) -> dict[str, Any]:
        state = self._load_state(run_id)
        if not observations:
            raise ValueError("No observations provided.")

        target_col = state["target_column"]
        existing = read_jsonl(self._paths(run_id).observations)
        next_iteration = len(existing)
        rows = []

        for idx, obs in enumerate(observations):
            x = dict(obs.get("x", {}))
            engine = str(obs.get("engine", state.get("default_engine", "hebo")))
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
                "event_time": utc_now_iso(),
                "iteration": next_iteration + idx,
                "source": source,
                "engine": engine,
                "suggestion_id": obs.get("suggestion_id"),
                "x": {k: to_python_scalar(v) for k, v in x.items()},
                "y": y_float,
                "y_internal": y_internal,
            }
            append_jsonl(self._paths(run_id).observations, payload)
            rows.append(payload)

        state["updated_at"] = utc_now_iso()
        self._save_state(run_id, state)
        self._log(
            verbose,
            f"[observe] run_id={run_id} source={source} recorded={len(rows)}",
        )
        return {"run_id": run_id, "recorded": len(rows), "observations": rows}

    def run_optimization(
        self,
        run_id: str,
        *,
        observer: Any,
        num_iterations: int,
        batch_size: int = 1,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Run a BO loop with a pluggable observer for evaluation."""
        self._log(
            verbose,
            f"[run] run_id={run_id} iterations={num_iterations} batch_size={batch_size} observer={observer.source}",
        )
        progress = tqdm(
            range(num_iterations),
            desc=f"run {run_id}",
            unit="iter",
            disable=not verbose,
            file=sys.stderr,
        )
        for _ in progress:
            result = self.suggest(run_id, batch_size=batch_size, verbose=verbose)
            observations = observer.evaluate(result["suggestions"])
            if observations:
                self.observe(run_id, observations, source=observer.source, verbose=verbose)
            if verbose:
                status = self.status(run_id)
                best = status.get("best_value")
                if best is not None:
                    progress.set_postfix(best=f"{float(best):.4f}")
        state = self._load_state(run_id)
        state["status"] = "completed"
        state["updated_at"] = utc_now_iso()
        self._save_state(run_id, state)
        self._log(verbose, f"[run] run_id={run_id} completed")
        return self.report(run_id, verbose=verbose)

    def status(self, run_id: str) -> dict[str, Any]:
        state = self._load_state(run_id)
        observations = read_jsonl(self._paths(run_id).observations)

        payload: dict[str, Any] = {
            "run_id": run_id,
            "status": state["status"],
            "objective": state["objective"],
            "default_engine": state.get("default_engine", "hebo"),
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

        if state.get("oracle") is not None:
            payload["oracle"] = {
                "selected_model": state["oracle"]["selected_model"],
                "selected_rmse": state["oracle"]["selected_rmse"],
            }
        return payload

    def report(self, run_id: str, *, verbose: bool = False) -> dict[str, Any]:
        """Generate report JSON and convergence plot for a run."""
        state = self._load_state(run_id)
        observations = read_jsonl(self._paths(run_id).observations)
        if not observations:
            report = {
                "run_id": run_id,
                "message": "No observations recorded yet.",
                "generated_at": utc_now_iso(),
            }
            write_json(self._paths(run_id).report, report)
            return report

        grouped: dict[str, list[float]] = {}
        for row in observations:
            engine = str(row.get("engine", state.get("default_engine", "hebo")))
            grouped.setdefault(engine, []).append(float(row["y"]))

        methods_data: dict[str, np.ndarray] = {}
        for engine, values in grouped.items():
            label = {
                "hebo": "HEBO",
                "bo_lcb": "BO (LCB)",
                "random": "Random Search",
            }.get(engine, engine)
            methods_data[label] = np.asarray(values, dtype=float)

        plot_optimization_convergence(
            methods_data,
            title=f"Run {run_id}",
            ylabel=state["target_column"],
            objective=state["objective"],
            fig_path=str(self._paths(run_id).convergence_plot),
            show=False,
        )

        status = self.status(run_id)
        report = {
            "run_id": run_id,
            "generated_at": utc_now_iso(),
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
        write_json(self._paths(run_id).report, report)
        self._log(
            verbose,
            f"[report] run_id={run_id} observations={len(observations)} best={report.get('best_value')}",
        )
        return report
