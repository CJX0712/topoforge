"""TopoForge 全局配置（ENV_TOPOFORGE_* 覆盖 + schema 校验）。作者：晨星。"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import ConfigError


@dataclass
class Config:
    seed: int = 20261004
    max_dimension: int = 2
    filtration_quantile: float = 0.60
    abs_eps_max: float | None = None
    n_jobs: int = 1
    classifier: str = "nearest_centroid"
    standardize: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ConfigError(f"seed 必须为非负整数: {self.seed!r}")
        if not (0 <= self.max_dimension <= 3):
            raise ConfigError(f"max_dimension 应在 [0,3]: {self.max_dimension}")
        if not (0.0 < self.filtration_quantile <= 1.0):
            raise ConfigError(f"filtration_quantile 应在 (0,1]: {self.filtration_quantile}")
        if self.n_jobs != 1:
            # Windows 受限环境 n_jobs=-1 会崩（pitfalls §D）；强制单进程
            self.n_jobs = 1

    @staticmethod
    def from_env() -> Config:
        def _int(name: str, default: int) -> int:
            v = os.environ.get(name)
            if v is None:
                return default
            try:
                return int(v)
            except ValueError as e:
                raise ConfigError(f"{name} 非整数: {v!r}") from e

        def _float(name: str, default: float) -> float:
            v = os.environ.get(name)
            if v is None:
                return default
            try:
                return float(v)
            except ValueError as e:
                raise ConfigError(f"{name} 非数值: {v!r}") from e

        def _bool(name: str, default: bool) -> bool:
            v = os.environ.get(name)
            if v is None:
                return default
            return v.strip().lower() in ("1", "true", "yes", "on")

        return Config(
            seed=_int("TOPOFORGE_SEED", 20261004),
            max_dimension=_int("TOPOFORGE_MAX_DIMENSION", 2),
            filtration_quantile=_float("TOPOFORGE_FILTRATION_QUANTILE", 0.60),
            n_jobs=_int("TOPOFORGE_N_JOBS", 1),
            classifier=os.environ.get("TOPOFORGE_CLASSIFIER", "nearest_centroid"),
            standardize=_bool("TOPOFORGE_STANDARDIZE", True),
        )
