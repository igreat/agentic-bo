from dataclasses import dataclass, field
from typing import Any, Mapping, Callable

from hebo.design_space.design_space import DesignSpace

from ..type_defs import OracleFn


@dataclass(frozen=True, slots=True)
class ProblemContext:
    name: str
    design_space: DesignSpace
    oracle: OracleFn
    metadata: Mapping[str, Any] = field(default_factory=dict)


type ProblemBuilder = Callable[..., ProblemContext]
