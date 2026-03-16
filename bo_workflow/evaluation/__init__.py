"""Evaluation infrastructure: proxy oracles, proxy observers, and evaluator loops."""

from .cli import run_hidden_oracle_evaluator
from .oracle import build_proxy_oracle, load_oracle, predict_original_scale
from .proxy import ProxyObserver

__all__ = [
    "ProxyObserver",
    "build_proxy_oracle",
    "load_oracle",
    "predict_original_scale",
    "run_hidden_oracle_evaluator",
]
