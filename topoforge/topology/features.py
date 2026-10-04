"""从持久同调图抽取定长拓扑特征向量。作者：晨星。

每个维度 d ∈ {0,1,2} 抽取 6 个统计：有限点数 / 本质点数 / 总持续性 /
最大持续性 / 持续性熵 / Betti 曲线积分。合计固定 18 维。
"""

from __future__ import annotations

import numpy as np

from ..core.types import TopoResult


def betti_curve(result: TopoResult, dim: int, scales: np.ndarray) -> np.ndarray:
    pd = result.diagrams.get(dim)
    if pd is None or pd.n_points == 0:
        return np.zeros_like(scales)
    b = pd.births
    de = pd.deaths
    eps = result.eps_max
    out = np.empty(len(scales), dtype=np.float64)
    for i, t in enumerate(scales):
        alive = np.sum((b <= t) & ((de > t) | ((~np.isfinite(de)) & (t <= eps))))
        out[i] = float(alive)
    return out


def extract_topo_features(
    result: TopoResult,
    dims: tuple[int, ...] = (0, 1, 2),
    n_scales: int = 20,
) -> np.ndarray:
    feats: list[float] = []
    scales = (
        np.linspace(0.0, result.eps_max, n_scales) if result.eps_max > 0 else np.zeros(n_scales)
    )
    for d in dims:
        pd = result.diagrams.get(d)
        if pd is None or pd.n_points == 0:
            feats += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            continue
        finite = pd.finite()
        ess = pd.essential()
        n_finite = finite.n_points
        n_ess = ess.n_points
        total = float(finite.persistences(None).sum()) if finite.n_points else 0.0
        maxp = float(finite.persistences(None).max()) if finite.n_points else 0.0
        ent = result.persistence_entropy(d, None)
        betti = betti_curve(result, d, scales)
        betti_int = float(np.trapezoid(betti, scales)) if n_scales > 1 else 0.0
        feats += [float(n_finite), float(n_ess), total, maxp, ent, betti_int]
    return np.array(feats, dtype=np.float64)


def extract_euclidean_features(X: np.ndarray, n_pca: int = 5) -> np.ndarray:
    """拓扑盲的欧氏/统计特征（强基线）。作者：晨星。"""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    feats: list[float] = []
    k = min(d, 5)
    feats += list(X.mean(axis=0)[:k])
    feats += list(X.std(axis=0)[:k])
    norms = np.linalg.norm(X, axis=1)
    feats += [float(norms.mean()), float(norms.std()), float(norms.min()), float(norms.max())]
    D = _pairwise(X)
    iu = np.triu_indices(n, 1)
    pd = D[iu]
    feats += [float(pd.mean()), float(pd.std()), float(pd.min()), float(pd.max())]
    Xc = X - X.mean(axis=0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    ev = (S**2) / max(n - 1, 1)
    tot = ev.sum()
    explained = (np.cumsum(ev) / tot) if tot > 0 else np.zeros_like(ev)
    feats += list(explained[:n_pca])
    target = 5 + 5 + 4 + 4 + n_pca
    if len(feats) < target:
        feats += [0.0] * (target - len(feats))
    else:
        feats = feats[:target]
    return np.array(feats, dtype=np.float64)


def _pairwise(X: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - X[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(d, 0.0)
    return d
