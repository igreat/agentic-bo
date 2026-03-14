"""ProxyObserver — evaluates suggestions using the trained proxy oracle."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..oracle import predict_with_uncertainty
from ..utils import RunPaths, read_json
from .base import Observer


class ProxyObserver(Observer):
    """Evaluates suggestions using the trained proxy oracle.

    Self-contained: captures all needed context (run_dir, features,
    objective, oracle metadata) at construction time.

    Observations fed to the BO engine are sampled from
    N(y_mean, y_std), where y_std is the inter-tree standard deviation
    of the ensemble. This communicates oracle uncertainty to the engine:
    suggestions in poorly-supported regions receive noisier observations,
    which naturally shifts the acquisition function toward exploration.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self._run_dir = Path(run_dir)
        paths = RunPaths(run_dir=self._run_dir)
        if not paths.oracle_model.exists():
            raise FileNotFoundError(
                f"Oracle not found at {paths.oracle_model}. "
                "Run 'build-oracle' first."
            )
        state = read_json(paths.state)
        self._active_features = list(state["active_features"])
        self._objective = state["objective"]
        self._default_engine = state.get("default_engine", "hebo")
        self._state = state
        self._rng = np.random.default_rng(state.get("seed"))

    @property
    def source(self) -> str:
        return "proxy-oracle"

    def evaluate(self, suggestions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        x_df = pd.DataFrame([row["x"] for row in suggestions])[self._active_features]
        y_mean, y_std = predict_with_uncertainty(self._run_dir, self._state, x_df)
        y_obs = self._rng.normal(y_mean, y_std)

        payloads = []
        for row, y_val in zip(suggestions, y_obs, strict=True):
            payloads.append(
                {
                    "x": row["x"],
                    "y": float(y_val),
                    "engine": row.get("engine", self._default_engine),
                    "suggestion_id": row.get("suggestion_id"),
                }
            )
        return payloads
