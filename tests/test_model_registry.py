"""
Tests for model registry.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_registry import create_model, list_models
from src.models.init import register_models
from src.rrdb_net import RRDBNet


def _rrdb_cfg() -> dict:
    return {
        "type": "rrdb",
        "in_nc": 3,
        "out_nc": 3,
        "nf": 16,
        "nb": 2,
        "gc": 8,
        "scale": 4,
    }


class TestModelRegistry:
    def test_rrdb_registered(self):
        register_models()
        assert "rrdb" in list_models()

    def test_create_rrdb(self):
        register_models()
        model = create_model(_rrdb_cfg())
        assert isinstance(model, RRDBNet)

    def test_default_type_rrdb(self):
        register_models()
        cfg = _rrdb_cfg()
        cfg.pop("type")
        model = create_model(cfg)
        assert isinstance(model, RRDBNet)

    @pytest.mark.parametrize("model_type", ["swinir", "swinir_lite", "hat"])
    def test_stubbed_models_raise(self, model_type):
        register_models()
        cfg = _rrdb_cfg()
        cfg["type"] = model_type
        with pytest.raises(NotImplementedError):
            _ = create_model(cfg)
