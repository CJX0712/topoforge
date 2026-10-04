"""分类器测试：Tier-1 离线兜底（纯 numpy 最近质心）+ 自动降级 + 确定性。作者：晨星。"""

from __future__ import annotations

import numpy as np
import pytest

from topoforge.classify.models import (
    NearestCentroidTopoClassifier,
    RandomForestTopoClassifier,
    available_random_forest,
    make_classifier,
)
from topoforge.core.errors import ClassifyError


def _dummy(n_per, dim):
    rng = np.random.default_rng(0)
    X0 = rng.normal(-5.0, 1.0, (n_per, dim))  # 类 0 中心 -5
    X1 = rng.normal(5.0, 1.0, (n_per, dim))  # 类 1 中心 +5
    Xtr = np.vstack([X0[: n_per // 2], X1[: n_per // 2]])
    ytr = np.array([0] * (n_per // 2) + [1] * (n_per // 2))
    Xte = np.vstack([X0[n_per // 2 :], X1[n_per // 2 :]])
    yte = np.array([0] * (n_per - n_per // 2) + [1] * (n_per - n_per // 2))
    return Xtr, ytr, Xte, yte


def test_nearest_centroid_basic():
    Xtr, ytr, Xte, yte = _dummy(10, 6)
    clf = NearestCentroidTopoClassifier(standardize=True)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    assert pred.shape == (len(Xte),)
    assert np.array_equal(pred, yte)  # 两类中心分明，预测应全部正确


def test_nearest_centroid_deterministic():
    Xtr, ytr, Xte, _ = _dummy(10, 6)
    a = NearestCentroidTopoClassifier(standardize=False).fit(Xtr, ytr).predict(Xte)
    b = NearestCentroidTopoClassifier(standardize=False).fit(Xtr, ytr).predict(Xte)
    assert np.array_equal(a, b)


def test_make_classifier_nearest_centroid():
    clf = make_classifier("nearest_centroid", standardize=False)
    assert isinstance(clf, NearestCentroidTopoClassifier)


def test_random_forest_available_or_raises_cleanly():
    if available_random_forest():
        clf = make_classifier("random_forest", random_state=0)
        assert isinstance(clf, RandomForestTopoClassifier)
    else:
        with pytest.raises(ClassifyError):
            make_classifier("random_forest", random_state=0)


def test_make_classifier_unknown_raises():
    with pytest.raises(ClassifyError):
        make_classifier("does_not_exist")


def test_auto_degrade_when_sklearn_absent(monkeypatch):
    """sklearn 不可用时 random_forest 应自动降级为 nearest_centroid（而非抛错）。"""
    import topoforge.classify.models as M

    monkeypatch.setattr(M, "_HAS_SK", False)
    monkeypatch.setattr(M, "_SkRF", None)
    monkeypatch.setattr(M, "_SkScaler", None)
    clf = make_classifier("random_forest", random_state=0)
    assert isinstance(clf, NearestCentroidTopoClassifier)
    # nearest_centroid 仍可独立构造
    assert isinstance(make_classifier("nearest_centroid"), NearestCentroidTopoClassifier)
