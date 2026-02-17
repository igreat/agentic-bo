"""ProxyObserver — evaluates suggestions using the trained proxy oracle."""

from pathlib import Path
from typing import Any

import pandas as pd

from ..oracle import predict_original_scale
from ..utils import RunPaths, read_json
from .base import Observer


class ProxyObserver(Observer):
    """Evaluates suggestions using the trained proxy oracle.

    Self-contained: captures all needed context (run_dir, features,
    objective, oracle metadata) at construction time.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self._run_dir = Path(run_dir)
        paths = RunPaths(run_dir=self._run_dir)
        state = read_json(paths.state)
        self._active_features = list(state["active_features"])
        self._objective = state["objective"]
        self._default_engine = state.get("default_engine", "hebo")
        self._state = state

    @property
    def source(self) -> str:
        return "proxy-oracle"

    def evaluate(self, suggestions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        x_df = pd.DataFrame([row["x"] for row in suggestions])[self._active_features]
        y_pred = predict_original_scale(self._run_dir, self._state, x_df)

        payloads = []
        for row, y_val in zip(suggestions, y_pred, strict=True):
            payloads.append(
                {
                    "x": row["x"],
                    "y": float(y_val),
                    "engine": row.get("engine", self._default_engine),
                    "suggestion_id": row.get("suggestion_id"),
                }
            )
        return payloads
