"""Constraint ABC — defines the enforcement interface for search-space constraints."""

from abc import ABC, abstractmethod

import pandas as pd


class Constraint(ABC):
    """Abstract base for search-space constraints.

    Constraints are enforced at suggest time: after the optimizer produces raw
    candidates, each constraint's ``apply()`` method projects them into the
    feasible region before they are returned to the caller.

    The current enforcement strategy for each constraint type is an
    implementation detail of the subclass — it can be upgraded (e.g. from
    post-hoc normalization to a bijection reparameterization) without changing
    the calling code in the engine.
    """

    @abstractmethod
    def apply(self, suggestions: pd.DataFrame) -> pd.DataFrame:
        """Project *suggestions* to satisfy this constraint.

        Receives and returns a DataFrame of candidate points. Implementations
        must not modify the input in-place — return a new or copied DataFrame.
        """
        ...

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to a state.json-compatible dict."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, spec: dict) -> "Constraint":
        """Deserialize from a state.json constraint spec."""
        ...
