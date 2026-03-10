import pandas as pd
import pytest

from bo_workflow.constraints import SimplexConstraint, load_constraints


def test_simplex_constraint_normalizes_rows() -> None:
    constraint = SimplexConstraint(["a", "b", "c"], total=100.0)
    frame = pd.DataFrame(
        [
            {"a": 10.0, "b": 20.0, "c": 30.0},
            {"a": 4.0, "b": 1.0, "c": 0.0},
        ]
    )

    normalized = constraint.apply(frame)

    assert normalized[["a", "b", "c"]].sum(axis=1).tolist() == pytest.approx([100.0, 100.0])
    assert normalized.loc[0, ["a", "b", "c"]].tolist() == pytest.approx(
        [16.6666667, 33.3333333, 50.0]
    )


def test_simplex_constraint_handles_all_zero_rows() -> None:
    constraint = SimplexConstraint(["a", "b", "c"], total=9.0)
    frame = pd.DataFrame([{"a": 0.0, "b": 0.0, "c": 0.0}])

    normalized = constraint.apply(frame)

    assert normalized.loc[0, ["a", "b", "c"]].tolist() == pytest.approx([3.0, 3.0, 3.0])


def test_load_constraints_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown constraint type"):
        load_constraints({"constraints": [{"type": "mystery"}]})


def test_simplex_constraint_clips_negative_values() -> None:
    constraint = SimplexConstraint(["a", "b", "c"], total=1.0)
    frame = pd.DataFrame([{"a": -0.5, "b": 0.8, "c": 0.4}])

    normalized = constraint.apply(frame)

    assert normalized[["a", "b", "c"]].sum(axis=1).tolist() == pytest.approx([1.0])
    assert normalized.loc[0, "a"] == pytest.approx(0.0)


def test_simplex_constraint_all_negative_distributes_uniformly() -> None:
    constraint = SimplexConstraint(["a", "b"], total=10.0)
    frame = pd.DataFrame([{"a": -1.0, "b": -2.0}])

    normalized = constraint.apply(frame)

    assert normalized.loc[0, ["a", "b"]].tolist() == pytest.approx([5.0, 5.0])


def test_simplex_constraint_rejects_duplicate_columns() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        SimplexConstraint(["a", "a"], total=1.0)
