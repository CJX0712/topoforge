"""合成流形点云生成器（固定 seed 可复现）。作者：晨星。

每个生成器：seed 控制采样，返回 X(n, d)。不同 seed 产生独立点云，
用于搭建 train/test 集。噪声 injection 模拟真实观测。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import DataError

# 已知 Betti 数（解析金标准），供不变量测试使用
BETTI = {
    "circle": {0: 1, 1: 1, 2: 0},
    "eight": {0: 1, 1: 2, 2: 0},
    "line": {0: 1, 1: 0, 2: 0},
    "sphere": {0: 1, 1: 0, 2: 1},
    "torus": {0: 1, 1: 2, 2: 1},
    "blob": {0: 1, 1: 0, 2: 0},
}


def _rng(seed: int) -> np.random.Generator:
    if not isinstance(seed, int):
        raise DataError(f"seed 必须为 int，收到 {type(seed).__name__}")
    return np.random.default_rng(seed)


def circle(seed: int, n: int = 80, radius: float = 5.0, noise: float = 0.15) -> np.ndarray:
    rng = _rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    X = np.stack([x, y], axis=1)
    X += rng.normal(0.0, noise, X.shape)
    return X


def eight(
    seed: int, n: int = 80, radius: float = 4.0, sep: float = 10.0, noise: float = 0.12
) -> np.ndarray:
    """两个分离圆的并集（8 字形），H1 = 2。"""
    rng = _rng(seed)
    half = n // 2
    t1 = rng.uniform(0.0, 2.0 * np.pi, half)
    t2 = rng.uniform(0.0, 2.0 * np.pi, n - half)
    c1 = np.stack([radius * np.cos(t1) - sep / 2.0, radius * np.sin(t1)], axis=1)
    c2 = np.stack([radius * np.cos(t2) + sep / 2.0, radius * np.sin(t2)], axis=1)
    X = np.vstack([c1, c2])
    X += rng.normal(0.0, noise, X.shape)
    return X


def line(
    seed: int, n: int = 80, length: float = 12.0, noise: float = 0.12, dim: int = 2
) -> np.ndarray:
    rng = _rng(seed)
    t = rng.uniform(0.0, 1.0, n)
    base = np.zeros((n, dim))
    base[:, 0] = t * length
    base += rng.normal(0.0, noise, base.shape)
    return base


def sphere(seed: int, n: int = 120, radius: float = 5.0, noise: float = 0.15) -> np.ndarray:
    rng = _rng(seed)
    v = rng.normal(0.0, 1.0, (n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    X = v * radius
    X += rng.normal(0.0, noise, X.shape)
    return X


def torus(
    seed: int, n: int = 140, R: float = 5.0, r: float = 2.0, noise: float = 0.15
) -> np.ndarray:
    rng = _rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    X = np.stack([x, y, z], axis=1)
    X += rng.normal(0.0, noise, X.shape)
    return X


def blob(
    seed: int, n: int = 80, spread: float = 3.0, dim: int = 3, noise: float = 0.0
) -> np.ndarray:
    rng = _rng(seed)
    X = rng.normal(0.0, spread, (n, dim))
    if noise > 0.0:
        X += rng.normal(0.0, noise, X.shape)
    return X


GENERATORS = {
    "circle": circle,
    "eight": eight,
    "line": line,
    "sphere": sphere,
    "torus": torus,
    "blob": blob,
}
