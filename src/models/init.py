"""
Model registry initialization.
"""

from .rrdb import build_rrdb_from_config
from ..model_registry import is_registered, register_model


def register_models() -> None:
    """Register built-in model backends."""
    if not is_registered("rrdb"):
        register_model("rrdb", build_rrdb_from_config)
