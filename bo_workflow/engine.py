"""Core BO engine.

By default, this module keeps optimization state on disk under
`<runs_root>/<run_id>` (with `runs_root` defaulting to `bo_runs`) and rebuilds
optimizers from logged observations when needed. That replay-first design keeps
the workflow resumable and robust for human-in-the-loop usage.
"""

from pathlib import Path
import secrets
import sys
from typing import Any

from .constraints import load_constraints
from .observers.base import Observer
from .plotting import save_run_convergence_plot

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.bo import BO
from hebo.optimizers.hebo import HEBO
import numpy as np
import pandas as pd
from tqdm import tqdm

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


def _validate_search_space_spec(
    spec: dict[str, Any], *, max_categories: int = 64
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate an explicit search-space spec and normalize it for engine state."""
    raw_params = spec.get("design_parameters")
    if not isinstance(raw_params, list) or not raw_params:
        raise ValueError("search_space_spec must include a non-empty design_parameters list.")

    raw_fixed = spec.get("fixed_features", {})
    if raw_fixed is None:
        raw_fixed = {}
    if not isinstance(raw_fixed, dict):
        raise ValueError("search_space_spec.fixed_features must be an object if provided.")

    params: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for raw in raw_params:
        if not isinstance(raw, dict):
            raise ValueError("Each design parameter must be an object.")
        name = raw.get("name")
        kind = raw.get("type")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each design parameter must have a non-empty string name.")
        if name in seen_names:
            raise ValueError(f"Duplicate design parameter name: '{name}'")
        seen_names.add(name)

        if kind == "num":
            try:
                lb = float(raw["lb"])
                ub = float(raw["ub"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Numeric design parameter '{name}' must include numeric lb/ub."
                ) from exc
            if not np.isfinite(lb) or not np.isfinite(ub) or lb >= ub:
                raise ValueError(
                    f"Numeric design parameter '{name}' must satisfy finite lb < ub."
                )
            params.append({"name": name, "type": "num", "lb": lb, "ub": ub})
            continue

        if kind == "cat":
            raw_categories = raw.get("categories")
            if not isinstance(raw_categories, list) or not raw_categories:
                raise ValueError(
                    f"Categorical design parameter '{name}' must include a non-empty categories list."
                )
            categories = []
            for value in raw_categories:
                if value is None:
                    raise ValueError(
                        f"Categorical design parameter '{name}' cannot include null categories."
                    )
                categories.append(str(value))
            categories = list(dict.fromkeys(categories))
            if len(categories) < 2:
                raise ValueError(
                    f"Categorical design parameter '{name}' must define at least 2 unique categories."
                )
            if len(categories) > max_categories:
                raise ValueError(
                    f"Feature '{name}' has {len(categories)} categories; max supported is {max_categories}."
                )
            params.append({"name": name, "type": "cat", "categories": categories})
            continue

        raise ValueError(
            f"Design parameter '{name}' must have type 'num' or 'cat', got '{kind}'."
        )

    fixed_features: dict[str, Any] = {}
    for name, value in raw_fixed.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("fixed_features keys must be non-empty strings.")
        if name in seen_names:
            raise ValueError(
                f"fixed_features overlaps with design_parameters for feature '{name}'."
            )
        fixed_features[name] = value

    return params, fixed_features


def _to_internal_objective(value: float, objective: Objective) -> float:
    if objective == "min":
        return value
    if objective == "max":
        return -value
    raise ValueError(f"Unknown objective: '{objective}'")


def _suggest_botorch(
    state: dict[str, Any],
    observations: list[dict[str, Any]],
    batch_size: int,
) -> pd.DataFrame:
    """Suggest candidates using BoTorch qLogNEI acquisition."""
    try:
        import torch
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.gp_regression_mixed import MixedSingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.optim import optimize_acqf
        from botorch.optim.optimize_mixed import optimize_acqf_mixed_alternating
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError as exc:
        raise ValueError(
            "BoTorch engine requires botorch, torch, and gpytorch to be installed. "
            "Install via: uv add botorch"
        ) from exc

    params = state["design_parameters"]
    feature_names = [p["name"] for p in params]
    cat_dims = [i for i, p in enumerate(params) if p["type"] == "cat"]

    if len(observations) < state["num_initial_random_samples"]:
        np.random.seed(int(state["seed"]) + len(observations))
        return DesignSpace().parse(params).sample(batch_size)

    def _encode_feature_value(param: dict[str, Any], value: Any) -> float:
        if param["type"] == "num":
            lb = float(param["lb"])
            ub = float(param["ub"])
            raw = float(value)
            return 0.0 if np.isclose(lb, ub) else (raw - lb) / (ub - lb)

        categories = list(param["categories"])
        try:
            return float(categories.index(str(value)))
        except ValueError as exc:
            raise ValueError(
                f"Unknown categorical value '{value}' for feature '{param['name']}'."
            ) from exc

    def _decode_feature_value(param: dict[str, Any], value: float) -> Any:
        if param["type"] == "num":
            lb = float(param["lb"])
            ub = float(param["ub"])
            scaled = float(np.clip(value, 0.0, 1.0))
            return lb + scaled * (ub - lb)

        categories = list(param["categories"])
        idx = int(np.clip(round(float(value)), 0, len(categories) - 1))
        return categories[idx]

    X_train = torch.tensor(
        [
            [
                _encode_feature_value(param, obs["x"][param["name"]])
                for param in params
            ]
            for obs in observations
        ],
        dtype=torch.double,
    )
    # y_internal is already negated for max objectives, so minimizing y_internal = maximizing y
    # BoTorch maximizes, so we negate y_internal (minimize y_internal = maximize -y_internal)
    Y_train = torch.tensor(
        [[-float(obs["y_internal"])] for obs in observations],
        dtype=torch.double,
    )

    bounds = torch.tensor(
        [
            [
                0.0 if p["type"] == "num" else 0.0
                for p in params
            ],
            [
                1.0 if p["type"] == "num" else float(len(p["categories"]) - 1)
                for p in params
            ],
        ],
        dtype=torch.double,
    )

    torch.manual_seed(int(state["seed"]) + len(observations))
    if cat_dims:
        model = MixedSingleTaskGP(
            X_train,
            Y_train,
            cat_dims=cat_dims,
            outcome_transform=Standardize(m=1),
        )
    else:
        model = SingleTaskGP(
            X_train,
            Y_train,
            outcome_transform=Standardize(m=1),
        )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    acqf = qLogNoisyExpectedImprovement(model=model, X_baseline=X_train)
    if cat_dims:
        cat_choices = {
            i: list(range(len(params[i]["categories"])))
            for i in cat_dims
        }
        candidates, _ = optimize_acqf_mixed_alternating(
            acq_function=acqf,
            bounds=bounds,
            cat_dims=cat_choices,
            q=batch_size,
            num_restarts=10,
            raw_samples=512,
        )
    else:
        candidates, _ = optimize_acqf(
            acqf,
            bounds=bounds,
            q=batch_size,
            num_restarts=10,
            raw_samples=512,
        )

    decoded_rows = []
    for row in candidates.detach().cpu().numpy():
        decoded_rows.append(
            {
                param["name"]: _decode_feature_value(param, row[i])
                for i, param in enumerate(params)
            }
        )
    return pd.DataFrame(decoded_rows, columns=feature_names)


class BOEngine:
    """Bayesian optimization engine with persisted run state."""

    def __init__(self, runs_root: str | Path = "bo_runs") -> None:
        self.runs_root = Path(runs_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def get_run_dir(self, run_id: str) -> Path:
        """Return the run directory for *run_id*."""
        return self.runs_root / run_id

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

    def _trajectory_summary(
        self,
        state: dict[str, Any],
        observations: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not observations:
            return None

        objective = str(state["objective"])
        init_random = int(state.get("num_initial_random_samples", 0))
        total = len(observations)
        y_values = np.asarray([float(row["y"]) for row in observations], dtype=float)

        if objective == "min":
            best_idx = int(np.argmin(y_values))
            compare = np.less
            # Lower is better, so a positive delta means the guided phase improved.
            improve_delta = lambda a, b: float(a - b)
        else:
            best_idx = int(np.argmax(y_values))
            compare = np.greater
            # Higher is better, so a positive delta means the guided phase improved.
            improve_delta = lambda a, b: float(b - a)

        best_so_far = float(y_values[0])
        last_improvement_idx = 0
        for idx, value in enumerate(y_values[1:], start=1):
            if compare(value, best_so_far):
                best_so_far = float(value)
                last_improvement_idx = idx

        summary: dict[str, Any] = {
            "best_observation_number": best_idx + 1,
            "last_improvement_observation": last_improvement_idx + 1,
            "observed_range": {
                "min_value": float(np.min(y_values)),
                "max_value": float(np.max(y_values)),
            },
        }

        seed_count = min(max(int(state.get("seed_observations_count", 0)), 0), total)
        if seed_count > 0:
            seed_values = y_values[:seed_count]
            if objective == "min":
                seed_best_idx = int(np.argmin(seed_values))
                seed_best_value = float(np.min(seed_values))
            else:
                seed_best_idx = int(np.argmax(seed_values))
                seed_best_value = float(np.max(seed_values))
            summary["seed_phase"] = {
                "num_observations": seed_count,
                "start_observation": 1,
                "end_observation": seed_count,
                "min_value": float(np.min(seed_values)),
                "max_value": float(np.max(seed_values)),
                "best_value": seed_best_value,
                "best_observation_number": seed_best_idx + 1,
            }

        if seed_count > 0:
            phase_indices = list(range(seed_count, total))
        else:
            phase_indices = [
                idx
                for idx, row in enumerate(observations)
                if row.get("suggestion_id") is not None
            ]
            if not phase_indices:
                phase_indices = list(range(total))

        if not phase_indices:
            return summary

        phase_engines = {
            str(observations[idx].get("engine", state.get("default_engine", "hebo")))
            for idx in phase_indices
        }
        if phase_engines == {"random"}:
            random_indices = phase_indices
        else:
            random_budget_remaining = max(0, init_random - seed_count)
            random_indices = phase_indices[: min(random_budget_remaining, len(phase_indices))]
        if random_indices:
            random_values = y_values[random_indices]
            if objective == "min":
                random_best_idx = int(np.argmin(random_values))
                random_best_value = float(np.min(random_values))
            else:
                random_best_idx = int(np.argmax(random_values))
                random_best_value = float(np.max(random_values))
            summary["random_phase"] = {
                "num_observations": len(random_indices),
                "start_observation": random_indices[0] + 1,
                "end_observation": random_indices[-1] + 1,
                "min_value": float(np.min(random_values)),
                "max_value": float(np.max(random_values)),
                "best_value": random_best_value,
                "best_observation_number": random_indices[random_best_idx] + 1,
            }

        guided_indices = phase_indices[len(random_indices) :]
        if guided_indices:
            guided_values = y_values[guided_indices]
            if objective == "min":
                guided_best_idx = int(np.argmin(guided_values))
                guided_best_value = float(np.min(guided_values))
            else:
                guided_best_idx = int(np.argmax(guided_values))
                guided_best_value = float(np.max(guided_values))

            guided_summary = {
                "num_observations": len(guided_indices),
                "start_observation": guided_indices[0] + 1,
                "end_observation": guided_indices[-1] + 1,
                "min_value": float(np.min(guided_values)),
                "max_value": float(np.max(guided_values)),
                "best_value": guided_best_value,
                "best_observation_number": guided_indices[guided_best_idx] + 1,
            }
            if "random_phase" in summary:
                guided_summary["improvement_over_random_best"] = improve_delta(
                    summary["random_phase"]["best_value"],
                    guided_best_value,
                )
            elif "seed_phase" in summary:
                guided_summary["improvement_over_seed_best"] = improve_delta(
                    summary["seed_phase"]["best_value"],
                    guided_best_value,
                )
            summary["model_guided_phase"] = guided_summary

        return summary

    def init_run(
        self,
        *,
        target_column: str,
        objective: Objective,
        dataset_path: str | Path | None = None,
        search_space_spec: dict[str, Any] | None = None,
        default_engine: OptimizerName = "hebo",
        hebo_model: str = "gp",
        run_id: str | None = None,
        num_initial_random_samples: int = 10,
        default_batch_size: int = 1,
        seed: int = 7,
        max_categories: int = 64,
        drop_cols: list[str] | None = None,
        constraints: list[dict[str, Any]] | None = None,
        intent: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Initialize a run from a dataset or explicit search-space spec."""
        if objective not in {"min", "max"}:
            raise ValueError("objective must be either 'min' or 'max'")
        if default_engine not in {"hebo", "bo_lcb", "random", "botorch"}:
            raise ValueError("default_engine must be one of: hebo, bo_lcb, random, botorch")
        if hebo_model not in {"gp", "rf"}:
            raise ValueError("hebo_model must be one of: gp, rf")
        if default_engine != "hebo" and hebo_model != "gp":
            raise ValueError("hebo_model is only supported when --engine hebo is selected.")
        if (dataset_path is None) == (search_space_spec is None):
            raise ValueError(
                "Provide exactly one of dataset_path or search_space_spec."
            )

        input_source: str
        resolved_dataset_path: Path | None = None
        dropped_features: list[str]
        if dataset_path is not None:
            resolved_dataset_path = Path(dataset_path).resolve()
            if not resolved_dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {resolved_dataset_path}")

            data = pd.read_csv(resolved_dataset_path)
            self._log(
                verbose,
                f"[init] dataset={resolved_dataset_path} rows={len(data)}",
            )
            if target_column not in data.columns:
                raise ValueError(
                    f"Target column '{target_column}' is not in dataset columns: {list(data.columns)}"
                )

            feature_frame = data.drop(columns=[target_column])
            if drop_cols:
                unknown = [c for c in drop_cols if c not in feature_frame.columns]
                if unknown:
                    raise ValueError(f"--drop-cols contains unknown columns: {unknown}")
                feature_frame = feature_frame.drop(columns=drop_cols)
            design_params, fixed_features, dropped_features = _infer_design_parameters(
                feature_frame,
                max_categories=max_categories,
            )
            input_source = "dataset"
        else:
            if drop_cols:
                raise ValueError("drop_cols is only supported for dataset-backed init.")
            design_params, fixed_features = _validate_search_space_spec(
                search_space_spec or {},
                max_categories=max_categories,
            )
            dropped_features = []
            input_source = "search_space_json"

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
        state = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "status": "initialized",
            "input_source": input_source,
            "dataset_path": (
                str(resolved_dataset_path) if resolved_dataset_path is not None else None
            ),
            "target_column": target_column,
            "objective": objective,
            "default_engine": default_engine,
            "hebo_model": hebo_model,
            "seed": int(seed),
            "num_initial_random_samples": int(num_initial_random_samples),
            "default_batch_size": int(default_batch_size),
            "design_parameters": design_params,
            "active_features": [p["name"] for p in design_params],
            "fixed_features": fixed_features,
            "dropped_features": dropped_features,
            "drop_cols": list(drop_cols) if drop_cols else [],
            "ignored_features": [],
            "constraints": [],
            "seed_observations_count": 0,
        }
        if constraints:
            # Validate and round-trip through constraint objects to catch errors early.
            loaded = load_constraints({"constraints": constraints})
            active_set = set(state["active_features"])
            constrained_cols: set[str] = set()
            numeric_features = {p["name"] for p in design_params if p["type"] == "num"}
            for c in loaded:
                serialized = c.to_dict()
                if serialized["type"] == "simplex":
                    unknown = [col for col in serialized["cols"] if col not in active_set]
                    if unknown:
                        raise ValueError(
                            f"Simplex constraint references columns not in active features: {unknown}"
                        )
                    non_numeric = [col for col in serialized["cols"] if col not in numeric_features]
                    if non_numeric:
                        raise ValueError(
                            f"Simplex constraint columns must be numeric features: {non_numeric}"
                        )
                    overlap = sorted(set(serialized["cols"]) & constrained_cols)
                    if overlap:
                        raise ValueError(
                            f"Constraint columns may only belong to one simplex group: {overlap}"
                        )
                    constrained_cols.update(serialized["cols"])
            state["constraints"] = [c.to_dict() for c in loaded]

        self._save_state(run_id, state)
        input_spec_payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "input_source": input_source,
            "dataset_path": state["dataset_path"],
            "target_column": target_column,
            "objective": objective,
            "design_parameters": state["design_parameters"],
            "fixed_features": state["fixed_features"],
            "dropped_features": state["dropped_features"],
            "drop_cols": state["drop_cols"],
            "constraints": state["constraints"],
        }
        write_json(self._paths(run_id).input_spec, input_spec_payload)
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
                    "input_source": input_source,
                    "dataset_path": state["dataset_path"],
                    "target_column": target_column,
                    "objective": objective,
                    "default_engine": default_engine,
                    "hebo_model": hebo_model,
                    "seed": int(seed),
                    "num_initial_random_samples": int(num_initial_random_samples),
                    "default_batch_size": int(default_batch_size),
                    "max_categories": int(max_categories),
                    "drop_cols": list(drop_cols) if drop_cols else [],
                    "design_parameters": state["design_parameters"],
                    "fixed_features": state["fixed_features"],
                    "constraints": state["constraints"],
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
                model_name=str(state.get("hebo_model", "gp")),
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
        if state["status"] not in {"initialized", "running"}:
            raise ValueError(
                f"Run '{run_id}' is not ready for suggestions. Current status: {state['status']}"
            )

        engine = str(state.get("default_engine", "hebo"))
        if engine not in {"hebo", "bo_lcb", "random", "botorch"}:
            raise ValueError("default_engine must be one of: hebo, bo_lcb, random, botorch")
        engine_typed: OptimizerName = engine  # type: ignore[assignment]  # validated above

        size = int(batch_size or state["default_batch_size"])
        observations = read_jsonl(self._paths(run_id).observations)
        if engine_typed == "random":
            np.random.seed(int(state["seed"]) + len(observations))
            proposals = DesignSpace().parse(state["design_parameters"]).sample(size)
        elif engine_typed == "botorch":
            proposals = _suggest_botorch(state, observations, size)
        else:
            if engine_typed == "bo_lcb" and size != 1:
                raise ValueError("bo_lcb currently supports batch-size=1 only.")
            optimizer = self._build_optimizer(state, observations, engine_typed)
            proposals = optimizer.suggest(n_suggestions=size)

        active_constraints = load_constraints(state)
        for constraint in active_constraints:
            proposals = constraint.apply(proposals)

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
        existing_suggestions = read_jsonl(self._paths(run_id).suggestions)
        next_iteration = len(existing)
        rows = []
        reserved_obs_keys = {
            "event_time",
            "iteration",
            "source",
            "x",
            "y",
            "y_internal",
            "engine",
            "suggestion_id",
            target_col,
        }

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

            y_internal = _to_internal_objective(y_float, state["objective"])
            extras = {
                str(key): to_python_scalar(value)
                for key, value in obs.items()
                if key not in reserved_obs_keys
            }

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
            payload.update(extras)
            append_jsonl(self._paths(run_id).observations, payload)
            rows.append(payload)

        state["updated_at"] = utc_now_iso()
        if not existing_suggestions:
            state["seed_observations_count"] = int(
                state.get("seed_observations_count", 0)
            ) + len(rows)
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
        observer: Observer,
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
                self.observe(
                    run_id, observations, source=observer.source, verbose=verbose
                )
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
            "hebo_model": state.get("hebo_model", "gp"),
            "target_column": state["target_column"],
            "active_features": state["active_features"],
            "ignored_features": state["ignored_features"],
            "constraints": state.get("constraints", []),
            "num_observations": len(observations),
            "observation_sources": sorted(
                {str(row.get("source", "unknown")) for row in observations}
            ),
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
            payload["best_observation_number"] = best_idx + 1
            payload["best_x"] = observations[best_idx]["x"]

        if state.get("oracle") is not None:
            payload["oracle"] = {
                "source": state["oracle"].get("source", "unknown"),
                "selected_model": state["oracle"]["selected_model"],
                "selected_rmse": state["oracle"]["selected_rmse"],
                "cv_rmse": state["oracle"].get("cv_rmse", {}),
                "active_features": state["oracle"].get("active_features", []),
            }
        return payload

    def report(self, run_id: str, *, verbose: bool = False) -> dict[str, Any]:
        """Generate report JSON for a run."""
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

        status = self.status(run_id)
        report = {
            "run_id": run_id,
            "generated_at": utc_now_iso(),
            "num_observations": len(observations),
            "objective": state["objective"],
            "default_engine": state.get("default_engine", "hebo"),
            "hebo_model": state.get("hebo_model", "gp"),
            "target_column": state["target_column"],
            "best_value": status.get("best_value"),
            "best_iteration": status.get("best_iteration"),
            "best_observation_number": status.get("best_observation_number"),
            "best_x": status.get("best_x"),
            "oracle": status.get("oracle"),
            "observation_sources": status.get("observation_sources", []),
            "trajectory": self._trajectory_summary(state, observations),
            "artifacts": {
                "state": str(self._paths(run_id).state),
                "observations": str(self._paths(run_id).observations),
                "convergence_plot": str(self._paths(run_id).convergence_plot),
            },
        }
        write_json(self._paths(run_id).report, report)
        save_run_convergence_plot(
            [float(row["y"]) for row in observations],
            objective=state["objective"],
            output_path=self._paths(run_id).convergence_plot,
        )
        self._log(
            verbose,
            f"[report] run_id={run_id} observations={len(observations)} best={report.get('best_value')}",
        )
        return report
