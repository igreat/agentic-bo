"""Compatibility facade for domain strategy layer.

Primary implementation now lives in `bo_workflow.domain.strategies.experiment_strategies`.
"""

from .domain.strategies.experiment_strategies import *  # noqa: F401,F403
from .domain.strategies.experiment_strategies import _canonical, _tanimoto_fp
