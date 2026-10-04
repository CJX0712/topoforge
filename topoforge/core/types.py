"""TopoForge 核心数据类型。作者：晨星。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PersistenceDiagram:
    """单维持久同调图：births/deaths 成对（deaths 为 inf 表示本质类）。"""

    dim: int
    births: np.ndarray
    deaths: np.ndarray

    def __post_init__(self) -> None:
        self.births = np.asarray(self.births, dtype=np.float64).reshape(-1)
        self.deaths = np.asarray(self.deaths, dtype=np.float64).reshape(-1)
        if self.births.shape[0] != self.deaths.shape[0]:
            raise ValueError("births 与 deaths 长度不一致")

    @property
    def n_points(self) -> int:
        return int(self.births.shape[0])

    def finite(self) -> PersistenceDiagram:
        mask = np.isfinite(self.deaths)
        return PersistenceDiagram(self.dim, self.births[mask], self.deaths[mask])

    def essential(self) -> PersistenceDiagram:
        mask = ~np.isfinite(self.deaths)
        return PersistenceDiagram(self.dim, self.births[mask], self.deaths[mask])

    def persistences(self, cap: float | None = None) -> np.ndarray:
        d = self.deaths.astype(np.float64).copy()
        if cap is not None:
            d = np.where(np.isfinite(d), d, float(cap))
        return d - self.births


@dataclass
class BettiCurve:
    """Betti 数随尺度变化的曲线。"""

    dim: int
    scales: np.ndarray
    values: np.ndarray


@dataclass
class TopoResult:
    """一次拓扑分析的结果集合。"""

    diagrams: dict[int, PersistenceDiagram] = field(default_factory=dict)
    betti_numbers: dict[int, int] = field(default_factory=dict)
    betti_curves: dict[int, BettiCurve] = field(default_factory=dict)
    eps_max: float = 0.0
    n_points: int = 0

    def betti(self, dim: int) -> int:
        """解析 Betti 数（由约简算法经 beta_d = n_d - nz_d - nz_{d+1} 得到）。"""
        return int(self.betti_numbers.get(dim, 0))

    def persistence_entropy(self, dim: int, cap: float | None = None) -> float:
        pd = self.diagrams.get(dim)
        if pd is None or pd.n_points == 0:
            return 0.0
        pers = pd.persistences(cap)
        with np.errstate(invalid="ignore", divide="ignore"):
            tot = float(np.nansum(pers))
        if tot <= 0.0 or not np.isfinite(tot):
            return 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            p = pers / tot
            p = p[np.isfinite(p) & (p > 0.0)]
        if p.size == 0:
            return 0.0
        return float(-float((p * np.log(p)).sum()))
