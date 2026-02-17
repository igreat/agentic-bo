"""Pluggable observer layer for BO evaluation.

An Observer produces observations for a batch of suggestions. The run loop
calls ``observer.evaluate()`` each iteration and records the returned
observations via ``engine.observe()``.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .engine import BOEngine


class Observer(ABC):
    """Abstract base for evaluation observers."""

    @abstractmethod
    def evaluate(
        self, engine: BOEngine, run_id: str, suggestions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Given suggestions from suggest(), return observation dicts.

        Return observations that need recording. Return ``[]`` if the
        observations were already recorded externally.
        """
        ...

    @property
    def source(self) -> str:
        """Provenance label for observations (e.g. 'proxy-oracle', 'external')."""
        return "observer"


class ProxyObserver(Observer):
    """Evaluates suggestions using the trained proxy oracle."""

    @property
    def source(self) -> str:
        return "proxy-oracle"

    def evaluate(
        self, engine: BOEngine, run_id: str, suggestions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        state = engine._load_state(run_id)
        x_df = pd.DataFrame([row["x"] for row in suggestions])[state["active_features"]]
        y_pred = engine._oracle_predict_original_scale(run_id, state, x_df)

        payloads = []
        for row, y_val in zip(suggestions, y_pred, strict=True):
            payloads.append(
                {
                    "x": row["x"],
                    "y": float(y_val),
                    "engine": row.get("engine", state.get("default_engine", "hebo")),
                    "suggestion_id": row.get("suggestion_id"),
                }
            )
        return payloads


class InteractiveObserver(Observer):
    """Delegates evaluation to a user-provided callback.

    The callback receives the list of suggestion dicts and returns
    a list of observation dicts (each with ``x``, ``y``, and optionally
    ``engine`` and ``suggestion_id``).
    """

    def __init__(
        self, callback: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
    ) -> None:
        self.callback = callback

    @property
    def source(self) -> str:
        return "interactive"

    def evaluate(
        self, engine: BOEngine, run_id: str, suggestions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return self.callback(suggestions)
