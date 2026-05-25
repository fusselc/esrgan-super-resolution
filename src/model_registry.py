"""
Model registry for super-resolution backends.

Provides a registry for model builders and clean stubs for planned transformer backends.
"""

from __future__ import annotations

from typing import Callable, Dict, List
import warnings

import torch.nn as nn


ModelBuilder = Callable[[dict], nn.Module]

_MODEL_REGISTRY: Dict[str, ModelBuilder] = {}
_STUB_MODELS = {"swinir", "swinir_lite", "hat"}


def register_model(name: str, builder: ModelBuilder) -> None:
    """Register a model builder by name."""
    key = name.lower()
    if key in _MODEL_REGISTRY:
        raise ValueError(f"Model '{key}' is already registered")
    _MODEL_REGISTRY[key] = builder


def is_registered(name: str) -> bool:
    """Return True if a model name is registered."""
    return name.lower() in _MODEL_REGISTRY


def list_models() -> List[str]:
    """List registered model names."""
    return sorted(_MODEL_REGISTRY.keys())


def create_model(cfg: dict) -> nn.Module:
    """Create a model from a config dict."""
    if cfg is None:
        raise ValueError("Model config must be provided")
    model_type = str(cfg.get("type", "rrdb")).lower()

    if model_type in _STUB_MODELS:
        warnings.warn(
            f"Model type '{model_type}' is not implemented. "
            "Transformer backends must be validated for scientific integrity "
            "before use.",
            RuntimeWarning,
            stacklevel=2,
        )
        raise NotImplementedError(
            f"Model type '{model_type}' is not implemented. "
            "Use 'rrdb' for now."
        )

    if model_type not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model type '{model_type}'. Registered: {list_models()}")

    return _MODEL_REGISTRY[model_type](cfg)
