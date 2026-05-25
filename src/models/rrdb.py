"""
RRDB model backend wiring.
"""

from typing import Any, Dict

from ..rrdb_net import RRDBNet


def _get(cfg: Dict[str, Any], *keys: str, default: Any) -> Any:
    for key in keys:
        if key in cfg:
            return cfg[key]
    return default


def build_rrdb_from_config(cfg: Dict[str, Any]) -> RRDBNet:
    """Build RRDBNet from a model config dict."""
    return RRDBNet(
        in_nc=_get(cfg, "in_nc", "in_channels", default=3),
        out_nc=_get(cfg, "out_nc", "out_channels", default=3),
        nf=_get(cfg, "nf", "num_features", default=64),
        nb=_get(cfg, "nb", "num_blocks", default=23),
        gc=_get(cfg, "gc", "growth_channels", default=32),
        scale=_get(cfg, "scale", default=4),
    )
