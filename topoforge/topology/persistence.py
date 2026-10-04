"""持久同调引擎。作者：晨星。

标准约简算法（Edelsbrunner），逐列在 GF(2) 上约简边界矩阵：
- 约简后列为空 → 该单形是边界（死亡映射的像）。
- 约简后列非空（low=l）→ l 被该列“杀死”（配对 (l, j)）。

Betti 数（与秩公式等价的快速算法）:
    beta_d = n_d - nz_d - nz_{d+1}
其中 nz_d 为约简后非空的 d-维列数（= rank(∂_d)），nz_{d+1} 为约简后非空的
(d+1)-维列数（= rank(∂_{d+1})）。为正确计算 H2，须构建到 3-单形
（见 vr_complex.build_filtration 的 build_dim 约定）。

持久同调图:
- 有限特征：正 (d+1)-单形 j（low=l 且 l 为正） → 配对 (birth=l, death=j)，维数 = dim(l)。
- 本质特征：个数 = beta_d - 有限特征数；出生单形取“正 d-单形且非任何 (d+1)-单形之 low”，
  按出生尺度降序取前 (beta_d - 有限数) 个。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import TopologyError
from ..core.types import PersistenceDiagram, TopoResult


def _reduce(simplices: list, boundary: list) -> tuple[dict, dict]:
    n = len(simplices)
    if n == 0:
        raise TopologyError("空单形集合，无法计算持久同调")
    order = sorted(range(n), key=lambda s: (simplices[s][0], simplices[s][1]))
    low_map: dict[int, int] = {}
    reduced: dict[int, set] = {}
    for j in order:
        col: set = set(boundary[j])
        while col:
            low = max(col)
            if low in low_map:
                col ^= reduced[low_map[low]]
            else:
                break
        reduced[j] = col
        if col:
            low = max(col)
            low_map[low] = j
    return reduced, low_map


def compute_persistence(
    simplices: list,
    boundary: list,
    max_dim: int = 2,
    eps_max: float = float("inf"),
) -> TopoResult:
    reduced, low_map = _reduce(simplices, boundary)
    n = len(simplices)
    n_by_dim = {d: sum(1 for s in range(n) if simplices[s][1] == d) for d in range(4)}
    nz = {d: sum(1 for s in range(n) if simplices[s][1] == d and reduced[s]) for d in range(4)}
    betti_numbers = {d: int(n_by_dim[d] - nz[d] - nz[d + 1]) for d in range(max_dim + 1)}

    # 有限特征：正 (d+1)-单形 j，其 low=l 为正 d-单形 → 配对 (l, j)
    finite: dict[int, list] = {d: [] for d in range(max_dim + 1)}
    for j in range(n):
        if not reduced[j]:
            continue
        low = max(reduced[j])
        dj = simplices[j][1]
        dim_low = simplices[low][1]
        if dj == dim_low + 1:
            if dim_low == 0 or reduced[low]:  # H0 的 low 为顶点（列恒空），特例放行
                if dim_low <= max_dim:
                    finite[dim_low].append((float(simplices[low][0]), float(simplices[j][0])))

    # 本质特征：正 d-单形且非任何 (d+1)-单形之 low → 出生单形；个数补齐到 beta_d
    ess_births: dict[int, list] = {d: [] for d in range(max_dim + 1)}
    for s in range(n):
        d = simplices[s][1]
        if d <= max_dim and reduced[s] and s not in low_map:
            ess_births[d].append(float(simplices[s][0]))

    diagrams: dict[int, PersistenceDiagram] = {}
    for d in range(max_dim + 1):
        births: list[float] = []
        deaths: list[float] = []
        for b, de in finite[d]:
            births.append(b)
            deaths.append(de)
        n_ess = max(0, betti_numbers[d] - len(finite[d]))
        for b in sorted(ess_births[d], reverse=True)[:n_ess]:
            births.append(b)
            deaths.append(float("inf"))
        diagrams[d] = PersistenceDiagram(d, np.array(births), np.array(deaths))

    return TopoResult(
        diagrams=diagrams,
        betti_numbers=betti_numbers,
        eps_max=float(eps_max),
        n_points=int(n_by_dim[0]),
    )


def multi_scale_betti(
    simplices: list,
    boundary: list,
    max_dim: int = 2,
    n_windows: int = 30,
    eps_max: float | None = None,
) -> list[tuple[float, dict]]:
    """一次性约简后，沿排序序的「前缀」扫描各尺度窗的 Betti 数。

    关键优化：约简是按 (value, dim) 顺序进行的，列 j 的约简只依赖更早的列，
    故任意前缀 k 的约简结果 = 全约简在前 k 列的取值。因此只需一次约简，
    各尺度窗 Betti 由前缀计数 O(n) 推出，避免对每个窗重复约简（O(窗×约简)）。
    返回 [(scale, {dim: betti}), ...]。
    """
    reduced, _ = _reduce(simplices, boundary)
    n = len(simplices)
    order = sorted(range(n), key=lambda s: (simplices[s][0], simplices[s][1]))
    values = np.array([float(simplices[s][0]) for s in order], dtype=np.float64)
    eps = float(eps_max) if eps_max is not None else (float(values[-1]) if n else 0.0)
    if n == 0 or eps <= 0.0:
        return []
    # 前缀计数：n_by_dim[k][d] = 前 k 个单形中 d-维数量；nz 同理（约简后非空）
    n_by_dim = {d: [0] * (n + 1) for d in range(4)}
    nz_by_dim = {d: [0] * (n + 1) for d in range(4)}
    for idx, j in enumerate(order, start=1):
        d = simplices[j][1]
        for dd in range(4):
            n_by_dim[dd][idx] = n_by_dim[dd][idx - 1]
            nz_by_dim[dd][idx] = nz_by_dim[dd][idx - 1]
        n_by_dim[d][idx] += 1
        if reduced[j]:
            nz_by_dim[d][idx] += 1
    out: list[tuple[float, dict]] = []
    for frac in np.linspace(0.10, 0.92, n_windows):
        r = float(frac * eps)
        k = int(np.searchsorted(values, r, side="right"))
        betti = {
            d: int(n_by_dim[d][k] - nz_by_dim[d][k] - nz_by_dim[d + 1][k])
            for d in range(max_dim + 1)
        }
        out.append((r, betti))
    return out


def persistence_summary(diagrams: dict[int, PersistenceDiagram], cap: float | None = None) -> dict:
    out: dict = {}
    for d, pd in diagrams.items():
        finite = pd.finite()
        ess = pd.essential()
        out[d] = {
            "n_finite": int(finite.n_points),
            "n_essential": int(ess.n_points),
            "max_finite_persistence": (
                float(finite.persistences(cap).max()) if finite.n_points else 0.0
            ),
        }
    return out
