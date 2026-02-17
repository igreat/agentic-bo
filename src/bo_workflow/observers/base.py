"""Observer ABC — defines the evaluation interface for BO run loops."""

from abc import ABC, abstractmethod
from typing import Any


class Observer(ABC):
    """Abstract base for evaluation observers."""

    @abstractmethod
    def evaluate(self, suggestions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Given suggestions from suggest(), return observation dicts.

        Return observations that need recording. Return ``[]`` if the
        observations were already recorded externally.
        """
        ...

    @property
    def source(self) -> str:
        """Provenance label for observations (e.g. 'proxy-oracle', 'external')."""
        return "observer"
