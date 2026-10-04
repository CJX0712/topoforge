"""Vietoris-Rips 复形构建。作者：晨星。

给定距离矩阵，按过滤尺度构造 0/1/2/(3) 维单形，并返回每个单形的边界下标表。
排序约定 (value, dim) 保证面先于余面。

为正确计算 Betti 数，默认会多构建一维单形（死亡映射）：
- 计算 H_d 需要 ∂_{d+1}，故 build_dim = min(max_dim + 1, 3)。
- 例如 max_dim=2 时构建到 3-单形，使 H2 的死亡映射存在（否则 H2 被严重高估）。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import TopologyError


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise TopologyError("X 必须为 2D (n, d)")
    if X.shape[0] < 2:
        raise TopologyError("点云至少需要 2 个点")
    if not np.all(np.isfinite(X)):
        raise TopologyError("X 含非有限坐标 (NaN/inf)")
    diff = X[:, None, :] - X[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(d, 0.0)
    return d


def build_filtration(
    dists: np.ndarray,
    max_dim: int = 2,
    eps_max: float = float("inf"),
    build_dim: int | None = None,
) -> tuple[list, list]:
    if max_dim < 0 or max_dim > 2:
        raise TopologyError("max_dim 应在 [0,2]")
    if build_dim is None:
        build_dim = min(max_dim + 1, 3)
    else:
        build_dim = min(build_dim, 3)
        if build_dim < max_dim:
            raise TopologyError("build_dim 必须 >= max_dim")
    if not np.all(np.isfinite(dists)):
        raise TopologyError("距离矩阵含非有限值")
    n = dists.shape[0]
    simplices: list = []
    simplex_index: dict = {}
    boundary: list = []

    for i in range(n):
        simplex_index[(i,)] = len(simplices)
        simplices.append((0.0, 0, (i,)))
        boundary.append([])

    if build_dim >= 1:
        for i in range(n):
            for j in range(i + 1, n):
                v = dists[i, j]
                if v <= eps_max:
                    a = simplex_index[(i,)]
                    b = simplex_index[(j,)]
                    simplex_index[(i, j)] = len(simplices)
                    simplices.append((float(v), 1, (i, j)))
                    boundary.append([a, b])

    if build_dim >= 2:
        for i in range(n):
            di = dists[i]
            for j in range(i + 1, n):
                dij = di[j]
                if dij > eps_max:
                    continue
                for k in range(j + 1, n):
                    v = max(dij, di[k], dists[j, k])
                    if v <= eps_max:
                        a = simplex_index[(i, j)]
                        b = simplex_index[(i, k)]
                        c = simplex_index[(j, k)]
                        simplex_index[(i, j, k)] = len(simplices)
                        simplices.append((float(v), 2, (i, j, k)))
                        boundary.append([a, b, c])

    if build_dim >= 3:
        # 3-单形仅在 v <= eps_max 时构建，提供 H2 的死亡映射。
        for i in range(n):
            di = dists[i]
            for j in range(i + 1, n):
                dij = di[j]
                if dij > eps_max:
                    continue
                for k in range(j + 1, n):
                    dijk = max(dij, di[k], dists[j, k])
                    if dijk > eps_max:
                        continue
                    for w in range(k + 1, n):
                        v = max(dijk, di[w], dists[j, w], dists[k, w])
                        if v <= eps_max:
                            a = simplex_index[(i, j, k)]
                            b = simplex_index[(i, j, w)]
                            c = simplex_index[(i, k, w)]
                            e = simplex_index[(j, k, w)]
                            simplex_index[(i, j, k, w)] = len(simplices)
                            simplices.append((float(v), 3, (i, j, k, w)))
                            boundary.append([a, b, c, e])

    return simplices, boundary
