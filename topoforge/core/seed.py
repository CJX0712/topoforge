"""全局确定性种子入口。作者：晨星。"""

from __future__ import annotations

import random as _random

import numpy as np

from .errors import ConfigError

_DEFAULT_SEED = 20261004
_rng = np.random.default_rng(_DEFAULT_SEED)


def set_all(seed: int = _DEFAULT_SEED) -> np.random.Generator:
    """一次性设齐 random / numpy legacy / numpy Generator 种子。

    返回 numpy Generator 供后续确定性采样使用。
    """
    if not isinstance(seed, int) or seed < 0:
        raise ConfigError(f"seed 必须为非负整数: {seed!r}")
    _random.seed(seed)
    try:
        np.random.seed(seed % (2**32))
    except Exception:  # noqa: BLE001
        pass
    global _rng
    _rng = np.random.default_rng(seed)
    return _rng


def get_rng(seed: int | None = None) -> np.random.Generator:
    if seed is None:
        return _rng
    return np.random.default_rng(seed)
