"""核心接口契约（Protocol）。作者：晨星。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FeatureExtractor(Protocol):
    def extract(self, X: np.ndarray) -> np.ndarray:
        """从点云 X(n, d) 抽取定长特征向量。"""
        ...


@runtime_checkable
class Classifier(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Classifier: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...
