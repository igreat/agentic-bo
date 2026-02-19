"""CallbackObserver — delegates evaluation to a user-provided callback."""

from collections.abc import Callable
from typing import Any

from .base import Observer


class CallbackObserver(Observer):
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
        return "callback"

    def evaluate(self, suggestions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.callback(suggestions)
