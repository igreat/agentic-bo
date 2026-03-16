"""Integration tests for the BO workflow.

Each test exercises the Python API end-to-end (no subprocess CLI calls),
using tmp_path for full isolation between tests.
"""

import math
from pathlib import Path

import pandas as pd
import pytest

from bo_workflow.engine import BOEngine
from bo_workflow.evaluation.cli import run_hidden_oracle_evaluator
from bo_workflow.evaluation.oracle import build_proxy_oracle
from bo_workflow.evaluation.proxy import ProxyObserver
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
    """Assert standard run artifacts exist and have expected content."""
    assert paths.state.exists()
    assert paths.input_spec.exists()
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


def test_her_full_proxy_loop_with_hebo_rf(engine: BOEngine, her_csv: Path) -> None:
    """HER dataset, max objective, full proxy loop using HEBO's RF surrogate."""
    state = engine.init_run(
        dataset_path=her_csv,
        target_column="Target",
        objective="max",
        default_engine="hebo",
        hebo_model="rf",
        seed=42,
    )
    run_id = state["run_id"]
    run_dir = engine.get_run_dir(run_id)

    build_proxy_oracle(run_dir)
    observer = ProxyObserver(run_dir)
    engine.run_optimization(run_id, observer=observer, num_iterations=ITERATIONS)

    paths = RunPaths(run_dir=run_dir)
    _assert_standard_artifacts(paths)

    final_state = read_json(paths.state)
    assert final_state["hebo_model"] == "rf"

    report = read_json(paths.report)
    assert report["default_engine"] == "hebo"
    assert report["hebo_model"] == "rf"


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


def test_botorch_supports_mixed_categorical_suggestions(
    engine: BOEngine, tmp_path: Path
) -> None:
    """BoTorch should support mixed categorical + numeric suggestions."""
    dataset = tmp_path / "mixed_botorch.csv"
    pd.DataFrame(
        [
            {"cat": "A", "num": 0.0, "target": 1.0},
            {"cat": "A", "num": 0.5, "target": 0.8},
            {"cat": "B", "num": 0.2, "target": 0.9},
            {"cat": "B", "num": 0.7, "target": 0.7},
            {"cat": "C", "num": 0.4, "target": 0.6},
            {"cat": "C", "num": 0.9, "target": 0.5},
        ]
    ).to_csv(dataset, index=False)

    state = engine.init_run(
        dataset_path=dataset,
        target_column="target",
        objective="min",
        default_engine="botorch",
        num_initial_random_samples=3,
        default_batch_size=1,
        seed=42,
    )
    run_id = state["run_id"]

    for y in [1.0, 0.9, 0.8]:
        suggestion = engine.suggest(run_id)["suggestions"][0]
        engine.observe(run_id, [{"x": suggestion["x"], "y": y}])

    result = engine.suggest(run_id, batch_size=2)
    assert result["engine"] == "botorch"
    assert len(result["suggestions"]) == 2
    for suggestion in result["suggestions"]:
        assert suggestion["x"]["cat"] in {"A", "B", "C"}
        assert 0.0 <= float(suggestion["x"]["num"]) <= 0.9


def test_oer_simplex_constraint_projects_suggestions(
    engine: BOEngine, oer_csv: Path
) -> None:
    """Simplex-constrained OER suggestions should sum to the declared total."""
    state = engine.init_run(
        dataset_path=oer_csv,
        target_column="Overpotential mV @10 mA cm-2",
        objective="min",
        seed=42,
        constraints=[
            {
                "type": "simplex",
                "cols": [
                    "Metal_1_Proportion",
                    "Metal_2_Proportion",
                    "Metal_3_Proportion",
                ],
                "total": 100.0,
            }
        ],
    )
    run_id = state["run_id"]

    result = engine.suggest(run_id, batch_size=4)

    assert state["constraints"] == [
        {
            "type": "simplex",
            "cols": [
                "Metal_1_Proportion",
                "Metal_2_Proportion",
                "Metal_3_Proportion",
            ],
            "total": 100.0,
        }
    ]
    for suggestion in result["suggestions"]:
        total = (
            float(suggestion["x"]["Metal_1_Proportion"])
            + float(suggestion["x"]["Metal_2_Proportion"])
            + float(suggestion["x"]["Metal_3_Proportion"])
        )
        assert total == pytest.approx(100.0)

    status = engine.status(run_id)
    assert status["constraints"] == state["constraints"]


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


@pytest.mark.slow
def test_simplex_constrained_columns_pinned_during_feature_selection(
    engine: BOEngine, oer_csv: Path
) -> None:
    """Simplex-constrained columns must survive --max-features feature selection."""
    simplex_cols = ["Metal_1_Proportion", "Metal_2_Proportion", "Metal_3_Proportion"]
    state = engine.init_run(
        dataset_path=oer_csv,
        target_column="Overpotential mV @10 mA cm-2",
        objective="min",
        seed=42,
        constraints=[{"type": "simplex", "cols": simplex_cols, "total": 100.0}],
    )
    run_id = state["run_id"]
    paths = RunPaths(run_dir=engine.get_run_dir(run_id))

    # Request far fewer features than the dataset has — constrained cols must survive.
    from bo_workflow.evaluation.oracle import build_proxy_oracle
    build_proxy_oracle(paths.run_dir, max_features=3)

    updated_state = read_json(paths.state)
    active = set(updated_state["active_features"])
    for col in simplex_cols:
        assert col in active, f"Constrained column '{col}' was dropped by feature selection"

    # Suggestions must still satisfy the constraint.
    result = engine.suggest(run_id, batch_size=2)
    for suggestion in result["suggestions"]:
        total = sum(float(suggestion["x"][c]) for c in simplex_cols)
        assert total == pytest.approx(100.0)


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


def test_search_space_init_suggest_observe_report_min(engine: BOEngine) -> None:
    """Runs initialized from explicit search-space JSON should work without a dataset."""
    state = engine.init_run(
        search_space_spec={
            "design_parameters": [
                {"name": "temperature_c", "type": "num", "lb": 20.0, "ub": 100.0},
                {"name": "solvent", "type": "cat", "categories": ["A", "B", "C"]},
            ],
            "fixed_features": {"pressure_bar": 1.0},
        },
        target_column="yield_pct",
        objective="min",
        seed=42,
    )
    run_id = state["run_id"]
    paths = RunPaths(run_dir=engine.get_run_dir(run_id))

    result = engine.suggest(run_id, batch_size=2)
    assert len(result["suggestions"]) == 2
    for suggestion in result["suggestions"]:
        assert 20.0 <= float(suggestion["x"]["temperature_c"]) <= 100.0
        assert suggestion["x"]["solvent"] in {"A", "B", "C"}
        assert suggestion["x"]["pressure_bar"] == 1.0

    engine.observe(
        run_id,
        [
            {"x": result["suggestions"][0]["x"], "y": 5.0},
            {"x": result["suggestions"][1]["x"], "y": 3.0},
        ],
    )
    report = engine.report(run_id)

    assert report["best_value"] == pytest.approx(3.0)
    input_spec = read_json(paths.input_spec)
    assert input_spec["input_source"] == "search_space_json"
    assert input_spec["dataset_path"] is None


def test_search_space_init_supports_max_objective(engine: BOEngine) -> None:
    """Max-objective runs initialized from search-space JSON should optimize correctly."""
    state = engine.init_run(
        search_space_spec={
            "design_parameters": [
                {"name": "temperature_c", "type": "num", "lb": 20.0, "ub": 100.0},
                {"name": "solvent", "type": "cat", "categories": ["A", "B"]},
            ]
        },
        target_column="yield_pct",
        objective="max",
        seed=42,
        default_engine="botorch",
        num_initial_random_samples=1,
    )
    run_id = state["run_id"]

    first = engine.suggest(run_id)["suggestions"][0]
    engine.observe(run_id, [{"x": first["x"], "y": 1.0}])

    second_batch = engine.suggest(run_id, batch_size=2)
    engine.observe(
        run_id,
        [
            {"x": second_batch["suggestions"][0]["x"], "y": 5.0},
            {"x": second_batch["suggestions"][1]["x"], "y": 3.0},
        ],
    )

    status = engine.status(run_id)
    report = engine.report(run_id)
    assert status["best_value"] == pytest.approx(5.0)
    assert report["best_value"] == pytest.approx(5.0)


def test_build_proxy_oracle_requires_labeled_dataset(engine: BOEngine) -> None:
    """Search-space-only runs cannot build a proxy oracle without labeled data."""
    state = engine.init_run(
        search_space_spec={
            "design_parameters": [
                {"name": "temperature_c", "type": "num", "lb": 20.0, "ub": 100.0}
            ]
        },
        target_column="yield_pct",
        objective="max",
    )

    with pytest.raises(ValueError, match="requires a labeled dataset"):
        build_proxy_oracle(engine.get_run_dir(state["run_id"]))


def test_hidden_oracle_evaluator_runs_search_space_loop(engine: BOEngine, tmp_path: Path) -> None:
    """Hidden evaluator should drive a search-space-only run from an external oracle dir."""
    dataset = tmp_path / "evaluator_dataset.csv"
    pd.DataFrame(
        [
            {"temperature_c": 20.0, "solvent": "A", "yield_pct": 1.0},
            {"temperature_c": 40.0, "solvent": "A", "yield_pct": 2.5},
            {"temperature_c": 60.0, "solvent": "B", "yield_pct": 4.0},
            {"temperature_c": 80.0, "solvent": "B", "yield_pct": 5.0},
            {"temperature_c": 100.0, "solvent": "A", "yield_pct": 3.5},
        ]
    ).to_csv(dataset, index=False)

    backend_state = engine.init_run(
        dataset_path=dataset,
        target_column="yield_pct",
        objective="max",
        seed=42,
    )
    backend_run_dir = engine.get_run_dir(backend_state["run_id"])
    build_proxy_oracle(backend_run_dir)

    search_state = engine.init_run(
        search_space_spec={
            "design_parameters": [
                {"name": "temperature_c", "type": "num", "lb": 20.0, "ub": 100.0},
                {"name": "solvent", "type": "cat", "categories": ["A", "B"]},
            ]
        },
        target_column="yield_pct",
        objective="max",
        seed=7,
    )
    run_id = search_state["run_id"]
    paths = RunPaths(run_dir=engine.get_run_dir(run_id))

    payload = run_hidden_oracle_evaluator(
        engine,
        run_id=run_id,
        oracle_dir=backend_run_dir,
        num_iterations=2,
        batch_size=2,
    )

    assert payload["recorded"] == 4
    assert paths.observations.exists()
    observations = read_jsonl(paths.observations)
    assert len(observations) == 4
    assert {row["source"] for row in observations} == {"benchmark-evaluator"}

    report = read_json(paths.report)
    assert report["observation_sources"] == ["benchmark-evaluator"]
    assert report["best_value"] is not None


def test_hidden_oracle_evaluator_resolves_pending_suggestions(
    engine: BOEngine, tmp_path: Path
) -> None:
    """Evaluator resume should observe already-pending suggestions before new rounds."""
    dataset = tmp_path / "resume_dataset.csv"
    pd.DataFrame(
        [
            {"temperature_c": 20.0, "solvent": "A", "yield_pct": 1.0},
            {"temperature_c": 40.0, "solvent": "A", "yield_pct": 2.5},
            {"temperature_c": 60.0, "solvent": "B", "yield_pct": 4.0},
            {"temperature_c": 80.0, "solvent": "B", "yield_pct": 5.0},
            {"temperature_c": 100.0, "solvent": "A", "yield_pct": 3.5},
        ]
    ).to_csv(dataset, index=False)

    backend_state = engine.init_run(
        dataset_path=dataset,
        target_column="yield_pct",
        objective="max",
        seed=42,
    )
    backend_run_dir = engine.get_run_dir(backend_state["run_id"])
    build_proxy_oracle(backend_run_dir)

    search_state = engine.init_run(
        search_space_spec={
            "design_parameters": [
                {"name": "temperature_c", "type": "num", "lb": 20.0, "ub": 100.0},
                {"name": "solvent", "type": "cat", "categories": ["A", "B"]},
            ]
        },
        target_column="yield_pct",
        objective="max",
        seed=7,
    )
    run_id = search_state["run_id"]
    paths = RunPaths(run_dir=engine.get_run_dir(run_id))

    engine.suggest(run_id, batch_size=1)

    payload = run_hidden_oracle_evaluator(
        engine,
        run_id=run_id,
        oracle_dir=backend_run_dir,
        num_iterations=0,
        batch_size=1,
    )

    assert payload["resolved_pending"] == 1
    assert payload["recorded"] == 1

    suggestions = read_jsonl(paths.suggestions)
    observations = read_jsonl(paths.observations)
    assert len(suggestions) == 1
    assert len(observations) == 1
    assert observations[0]["suggestion_id"] == suggestions[0]["suggestion_id"]


def test_init_hebo_rf_persists_in_status(engine: BOEngine, her_csv: Path) -> None:
    """HEBO surrogate selection should persist in state and status."""
    state = engine.init_run(
        dataset_path=her_csv,
        target_column="Target",
        objective="max",
        default_engine="hebo",
        hebo_model="rf",
        seed=42,
    )

    assert state["hebo_model"] == "rf"

    status = engine.status(state["run_id"])
    assert status["default_engine"] == "hebo"
    assert status["hebo_model"] == "rf"


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


def test_init_non_hebo_engine_rejects_hebo_model(
    engine: BOEngine, her_csv: Path
) -> None:
    """Non-HEBO engines should reject HEBO surrogate configuration."""
    with pytest.raises(ValueError, match="only supported when --engine hebo"):
        engine.init_run(
            dataset_path=her_csv,
            target_column="Target",
            objective="max",
            default_engine="random",
            hebo_model="rf",
        )


def test_init_simplex_constraint_unknown_feature_raises(
    engine: BOEngine, her_csv: Path
) -> None:
    """Simplex constraints must reference active features."""
    with pytest.raises(ValueError, match="not in active features"):
        engine.init_run(
            dataset_path=her_csv,
            target_column="Target",
            objective="max",
            constraints=[
                {
                    "type": "simplex",
                    "cols": ["unknown_a", "unknown_b"],
                    "total": 1.0,
                }
            ],
        )
