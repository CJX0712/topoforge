"""TopoForge 分类器（Tier-0 sklearn RandomForest + Tier-1 纯 numpy 最近质心离线兜底）。作者：晨星。

设计：
  - Tier-0（首选顶级开源）：sklearn RandomForestClassifier，强基线对照用同款分类器保证公平。
  - Tier-1（离线兜底，零下载）：纯 numpy 最近质心分类器（NearestCentroid），sklearn 不可用时自动降级。
  - 全局确定性：RandomForest 固定 random_state；最近质心无随机性。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import ClassifyError

try:  # 隔离导入，sklearn 缺失不阻断整包
    from sklearn.ensemble import RandomForestClassifier as _SkRF  # type: ignore
    from sklearn.preprocessing import StandardScaler as _SkScaler  # type: ignore

    _HAS_SK = True
except Exception:  # noqa: BLE001
    _SkRF = None  # type: ignore
    _SkScaler = None  # type: ignore
    _HAS_SK = False


def available_random_forest() -> bool:
    """探测 Tier-0 后端（sklearn RandomForest）是否可用。"""
    return _HAS_SK


class RandomForestTopoClassifier:
    """Tier-0：sklearn RandomForest。强后端，性能对照的公平基线。"""

    def __init__(
        self, n_estimators: int = 200, random_state: int = 0, standardize: bool = True
    ) -> None:
        if not _HAS_SK:
            raise ClassifyError(
                "sklearn 不可用，无法构造 RandomForestTopoClassifier；"
                "请改用 NearestCentroidTopoClassifier（Tier-1 离线兜底）"
            )
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.standardize = standardize
        self.scaler = _SkScaler() if standardize else None
        self.clf = _SkRF(n_estimators=n_estimators, random_state=random_state, n_jobs=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestTopoClassifier:
        X = np.asarray(X, dtype=np.float64)
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        self.clf.fit(X, np.asarray(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.clf.predict(X)


class NearestCentroidTopoClassifier:
    """Tier-1 离线兜底：纯 numpy 最近质心（无 sklearn 依赖，零下载可跑）。"""

    def __init__(self, standardize: bool = True) -> None:
        self.standardize = standardize
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._centroids: np.ndarray | None = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        if not self.standardize or self._mean is None or self._std is None:
            return X
        return (X - self._mean) / np.where(self._std > 0.0, self._std, 1.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> NearestCentroidTopoClassifier:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if self.standardize:
            self._mean = X.mean(axis=0)
            self._std = X.std(axis=0)
            X = self._standardize(X)
        self._labels = np.unique(y)
        self._centroids = np.stack([X[y == lab].mean(axis=0) for lab in self._labels])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X = self._standardize(X)
        d = np.linalg.norm(X[:, None, :] - self._centroids[None, :, :], axis=2)
        idx = np.argmin(d, axis=1)
        return self._labels[idx]


def make_classifier(name: str, random_state: int = 0, standardize: bool = True):
    """按名称构造分类器；random_forest 不可用时自动降级 nearest_centroid（Tier-1 离线兜底）。"""
    name = (name or "nearest_centroid").lower()
    if name == "random_forest":
        if _HAS_SK:
            return RandomForestTopoClassifier(random_state=random_state, standardize=standardize)
        # 自动降级到 Tier-1 离线兜底（sklearn 不可用时零下载可跑）
        import warnings

        warnings.warn(
            "sklearn 不可用，random_forest 自动降级为 nearest_centroid（Tier-1 离线兜底）",
            stacklevel=2,
        )
        return NearestCentroidTopoClassifier(standardize=standardize)
    if name == "nearest_centroid":
        return NearestCentroidTopoClassifier(standardize=standardize)
    raise ClassifyError(f"未知分类器: {name!r}（支持 random_forest / nearest_centroid）")
