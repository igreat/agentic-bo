from __future__ import annotations

import argparse

from ....engine import BOEngine
from .handle_core import handle_core
from .handle_molecular import handle_molecular
from .handle_workflows import handle_workflows


def handle(args: argparse.Namespace, engine: BOEngine) -> int | None:
    for fn in (handle_core, handle_molecular, handle_workflows):
        result = fn(args, engine)
        if result is not None:
            return result
    return None
