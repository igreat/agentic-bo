"""Integration tests for the BO workflow.

Each test exercises the Python API end-to-end (no subprocess CLI calls),
using tmp_path for full isolation between tests.
"""

import math
from pathlib import Path

import pytest

from bo_workflow.engine import BOEngine
from bo_workflow.oracle import build_proxy_oracle
from bo_workflow.observers.proxy import ProxyObserver
from bo_workflow.utils import RunPaths, read_json, read_jsonl

ITERATIONS = 5


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _run_full_proxy_loop(
    engine: BOEngine,
    dataset_path: Path,
    target: str,
    objective: str,
    *,
    iterations: int = ITERATIONS,
    max_features: int | None = None,
) -> tuple[str, RunPaths]:
    """Init → build-oracle → run-proxy → report. Returns (run_id, paths)."""
    state = engine.init_run(
        dataset_path=dataset_path,
        target_column=target,
        objective=objective,
        seed=42,
    )
    run_id = state["run_id"]
    run_dir = engine.get_run_dir(run_id)

    build_proxy_oracle(run_dir, max_features=max_features)

    observer = ProxyObserver(run_dir)
    engine.run_optimization(
        run_id,
        observer=observer,
        num_iterations=iterations,
    )

    paths = RunPaths(run_dir=run_dir)
    return run_id, paths


def _assert_standard_artifacts(paths: RunPaths, iterations: int = ITERATIONS) -> None:
    """Assert the 7 standard run artifacts exist and have expected content."""
    assert paths.state.exists()
    assert paths.oracle_model.exists()
    assert paths.oracle_meta.exists()
    assert paths.suggestions.exists()
    assert paths.observations.exists()
    assert paths.convergence_plot.exists()
    assert paths.report.exists()

    state = read_json(paths.state)
    assert state["status"] == "completed"

    oracle_meta = read_json(paths.oracle_meta)
    rmse = oracle_meta["selected_rmse"]
    assert math.isfinite(rmse) and rmse > 0

    report = read_json(paths.report)
    assert math.isfinite(report["best_value"])

    observations = read_jsonl(paths.observations)
    assert len(observations) == iterations

    suggestions = read_jsonl(paths.suggestions)
    assert len(suggestions) == iterations


# ------------------------------------------------------------------
# Happy-path full proxy loop tests
# ------------------------------------------------------------------


def test_her_full_proxy_loop(engine: BOEngine, her_csv: Path) -> None:
    """HER dataset, max objective, full proxy loop."""
    _, paths = _run_full_proxy_loop(engine, her_csv, "Target", "max")
    _assert_standard_artifacts(paths)


def test_hea_full_proxy_loop(engine: BOEngine, hea_csv: Path) -> None:
    """HEA dataset, max objective, full proxy loop."""
    _, paths = _run_full_proxy_loop(engine, hea_csv, "target", "max")
    _assert_standard_artifacts(paths)


def test_oer_mixed_variables(engine: BOEngine, oer_csv: Path) -> None:
    """OER dataset, min objective, verifies categorical detection."""
    _, paths = _run_full_proxy_loop(
        engine, oer_csv, "Overpotential mV @10 mA cm-2", "min",
    )
    _assert_standard_artifacts(paths)

    state = read_json(paths.state)
    cat_params = [p for p in state["design_parameters"] if p["type"] == "cat"]
    assert len(cat_params) >= 1, "OER dataset should have at least one categorical parameter"


@pytest.mark.slow
def test_bh_feature_selection(engine: BOEngine, bh_csv: Path) -> None:
    """BH dataset, max objective, feature selection with max_features=20."""
    _, paths = _run_full_proxy_loop(
        engine, bh_csv, "yield", "max", max_features=20
    )
    _assert_standard_artifacts(paths)

    state = read_json(paths.state)
    assert len(state["active_features"]) == 20
    assert len(state["ignored_features"]) > 0
    assert "original_design_parameters" in state


# ------------------------------------------------------------------
# Human-in-the-loop test
# ------------------------------------------------------------------


def test_human_loop_suggest_observe(engine: BOEngine, her_csv: Path) -> None:
    """Suggest/observe cycle without oracle (human-in-the-loop pattern)."""
    state = engine.init_run(
        dataset_path=her_csv,
        target_column="Target",
        objective="max",
        seed=42,
    )
    run_id = state["run_id"]

    for _ in range(2):
        result = engine.suggest(run_id)
        suggestion = result["suggestions"][0]
        assert "x" in suggestion
        assert set(state["active_features"]).issubset(suggestion["x"].keys())

        engine.observe(run_id, [{"x": suggestion["x"], "y": 1.23}])

    paths = RunPaths(run_dir=engine.get_run_dir(run_id))
    observations = read_jsonl(paths.observations)
    assert len(observations) == 2

    final_state = read_json(paths.state)
    assert final_state["status"] == "running"


# ------------------------------------------------------------------
# Negative / error-path tests
# ------------------------------------------------------------------


def test_proxy_observer_missing_oracle(engine: BOEngine, her_csv: Path) -> None:
    """ProxyObserver raises FileNotFoundError when oracle hasn't been built."""
    state = engine.init_run(
        dataset_path=her_csv,
        target_column="Target",
        objective="max",
    )
    run_dir = engine.get_run_dir(state["run_id"])

    with pytest.raises(FileNotFoundError, match="build-oracle"):
        ProxyObserver(run_dir)


def test_observe_missing_y_raises(engine: BOEngine, her_csv: Path) -> None:
    """Observing without a 'y' value raises ValueError."""
    state = engine.init_run(
        dataset_path=her_csv,
        target_column="Target",
        objective="max",
        seed=42,
    )
    run_id = state["run_id"]
    result = engine.suggest(run_id)
    suggestion = result["suggestions"][0]

    with pytest.raises(ValueError, match="[Mm]issing objective value"):
        engine.observe(run_id, [{"x": suggestion["x"]}])


def test_init_invalid_target_column(engine: BOEngine, her_csv: Path) -> None:
    """init_run with a nonexistent target column raises ValueError."""
    with pytest.raises(ValueError, match="not in dataset columns"):
        engine.init_run(
            dataset_path=her_csv,
            target_column="nonexistent_column",
            objective="max",
        )
