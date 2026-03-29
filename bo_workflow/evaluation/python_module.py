"""Run BO against a locally written Python evaluator module.

TODO: after the first working open-world Claude runs, fold this into the
backend-id flow via a persisted `backend_kind="python_module"` backend shape so
`run-evaluator --backend-id ...` becomes the normalized evaluator entrypoint.
"""

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from typing import Any

from ..engine import BOEngine
from ..observers.callback import CallbackObserver
from ..utils import read_jsonl, utc_now_iso


@contextmanager
def _module_parent_on_sys_path(module_path: Path):
    parent = str(module_path.parent)
    inserted_index: int | None = None
    if parent not in sys.path:
        sys.path.insert(0, parent)
        inserted_index = 0
    try:
        yield
    finally:
        if inserted_index is not None and len(sys.path) > inserted_index:
            if sys.path[inserted_index] == parent:
                sys.path.pop(inserted_index)


def _validate_python_evaluator_preconditions(
    engine: BOEngine,
    run_id: str,
    batch_size: int,
    num_iterations: int,
) -> dict[str, Any]:
    state = engine._load_state(run_id)
    if state["status"] not in {"initialized", "running"}:
        raise ValueError(
            f"Run '{run_id}' is not ready for suggestions. Current status: {state['status']}"
        )

    if int(num_iterations) < 0:
        raise ValueError("num_iterations must be >= 0.")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1.")

    engine_name = str(state.get("default_engine", "hebo"))
    if engine_name == "bo_lcb" and int(batch_size) != 1:
        raise ValueError("bo_lcb currently supports batch-size=1 only.")

    return state


def _pending_suggestions(engine: BOEngine, run_id: str) -> list[dict[str, Any]]:
    """Return suggestions that were logged but not yet observed."""
    paths = engine._paths(run_id)
    suggestions = read_jsonl(paths.suggestions)
    observations = read_jsonl(paths.observations)
    observed_ids = {
        str(row["suggestion_id"])
        for row in observations
        if row.get("suggestion_id") is not None
    }
    return [
        row
        for row in suggestions
        if row.get("suggestion_id") is not None
        and str(row["suggestion_id"]) not in observed_ids
    ]


def _load_python_evaluator(module_path: str | Path, function_name: str) -> Any:
    module_path = Path(module_path).resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Python evaluator module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(
        f"_bo_local_eval_{module_path.stem}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python evaluator module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    with _module_parent_on_sys_path(module_path):
        spec.loader.exec_module(module)

    fn = getattr(module, function_name, None)
    if fn is None or not callable(fn):
        raise AttributeError(
            f"Python evaluator module {module_path} has no callable '{function_name}'"
        )
    return fn


def _normalize_python_evaluator_result(
    *,
    result: Any,
    suggestion: dict[str, Any],
    target_column: str,
    default_engine: str,
) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    if isinstance(result, dict):
        if "y" in result:
            y_value = result["y"]
        elif target_column in result:
            y_value = result[target_column]
        else:
            raise ValueError(
                "Python evaluator dict output must include 'y' or the target column."
            )
        extras = {
            str(key): value
            for key, value in result.items()
            if key not in {"y", target_column, "x", "engine", "suggestion_id"}
        }
    else:
        y_value = result

    payload = {
        "x": suggestion["x"],
        "y": float(y_value),
        "engine": suggestion.get("engine", default_engine),
        "suggestion_id": suggestion.get("suggestion_id"),
    }
    payload.update(extras)
    return payload


def _attach_python_evaluator_summary(
    engine: BOEngine,
    run_id: str,
    *,
    module_path: Path,
    function_name: str,
) -> None:
    state = engine._load_state(run_id)
    state["oracle"] = {
        "source": "python_evaluator_module",
        "selected_model": "python-callback",
        "selected_rmse": None,
        "cv_rmse": {},
        "active_features": state.get("active_features", []),
        "module_path": str(module_path),
        "function_name": function_name,
    }
    state["updated_at"] = utc_now_iso()
    engine._save_state(run_id, state)


def run_python_module_evaluator(
    engine: BOEngine,
    *,
    run_id: str,
    module_path: str | Path,
    function_name: str = "evaluate",
    num_iterations: int,
    batch_size: int = 1,
    verbose: bool = False,
) -> dict[str, Any]:
    state = _validate_python_evaluator_preconditions(
        engine,
        run_id,
        batch_size,
        num_iterations,
    )
    module_path = Path(module_path).resolve()
    evaluator = _load_python_evaluator(module_path, function_name)
    _attach_python_evaluator_summary(
        engine,
        run_id,
        module_path=module_path,
        function_name=function_name,
    )
    target_column = str(state["target_column"])
    default_engine = str(state.get("default_engine", "hebo"))
    paths = engine._paths(run_id)
    recorded_before = len(read_jsonl(paths.observations))

    def callback(suggestions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        observations = []
        with _module_parent_on_sys_path(module_path):
            for suggestion in suggestions:
                result = evaluator(dict(suggestion["x"]))
                observations.append(
                    _normalize_python_evaluator_result(
                        result=result,
                        suggestion=suggestion,
                        target_column=target_column,
                        default_engine=default_engine,
                    )
                )
        return observations

    class PythonEvaluatorObserver(CallbackObserver):
        @property
        def source(self) -> str:
            return "python-evaluator"

    pending = _pending_suggestions(engine, run_id)
    if pending:
        engine.observe(
            run_id,
            callback(pending),
            source="python-evaluator",
            verbose=verbose,
        )

    if int(num_iterations) > 0:
        observer = PythonEvaluatorObserver(callback)
        report = engine.run_optimization(
            run_id,
            observer=observer,
            num_iterations=int(num_iterations),
            batch_size=int(batch_size),
            verbose=verbose,
        )
    else:
        report = engine.report(run_id, verbose=verbose)

    recorded = len(read_jsonl(paths.observations)) - recorded_before
    return {
        "run_id": run_id,
        "module_path": str(module_path),
        "function_name": function_name,
        "iterations": int(num_iterations),
        "batch_size": int(batch_size),
        "recorded": recorded,
        "resolved_pending": len(pending),
        "best_value": report.get("best_value"),
        "best_iteration": report.get("best_iteration"),
        "best_observation_number": report.get("best_observation_number"),
        "report_path": str(paths.report),
        "convergence_plot_path": str(paths.convergence_plot),
    }
