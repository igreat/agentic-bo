from .engine import BOEngine

__all__ = ["BOEngine"]


"""bo_workflow package."""

__all__ = ["BOEngine"]


def __getattr__(name: str):
    if name == "BOEngine":
        from .engine import BOEngine
        return BOEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")