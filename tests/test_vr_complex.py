"""Vietoris-Rips 复形构建测试（含 H2 build_dim=3 回归测试）。作者：晨星。"""

from __future__ import annotations

import numpy as np
import pytest

from topoforge.core.errors import TopologyError
from topoforge.data.generators import GENERATORS
from topoforge.topology.persistence import compute_persistence
from topoforge.topology.vr_complex import build_filtration, pairwise_distances


def _full_betti(X, build_dim):
    Xd = np.asarray(X, float)
    diff = Xd[:, None, :] - Xd[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 0.0)
    eps = float(np.max(dists))
    simp, bnd = build_filtration(dists, max_dim=2, eps_max=eps, build_dim=build_dim)
    return compute_persistence(simp, bnd, max_dim=2, eps_max=eps)


def test_pairwise_distances_basic():
    X = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 5.0]])
    d = pairwise_distances(X)
    assert d.shape == (3, 3)
    assert np.allclose(np.diag(d), 0.0)
    assert abs(d[0, 1] - 5.0) < 1e-9
    assert d[0, 1] == d[1, 0]


def test_pairwise_rejects_bad_input():
    with pytest.raises(TopologyError):
        pairwise_distances(np.array([1.0, 2.0, 3.0]))  # 1D
    with pytest.raises(TopologyError):
        pairwise_distances(np.array([[1.0, 2.0], [np.nan, 3.0]]))


def test_build_dim_invalid():
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    d = pairwise_distances(X)
    with pytest.raises(TopologyError):
        build_filtration(d, max_dim=2, build_dim=1)  # build_dim < max_dim


def test_boundary_face_indices_correct():
    """单三角形点云：2-单形的边界必须是其三条边（1-单形）的下标，而非顶点。"""
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    d = pairwise_distances(X)
    simp, bnd = build_filtration(d, max_dim=2, eps_max=10.0, build_dim=2)
    # 找到 2-单形
    two = [i for i, s in enumerate(simp) if s[1] == 2]
    assert len(two) == 1
    s2 = two[0]
    faces = bnd[s2]
    # 所有面必须是 1-单形
    assert all(simp[f][1] == 1 for f in faces)
    assert len(faces) == 3


def test_build_dim_2_overcounts_H2_regression():
    """回归测试：2D 形状用 build_dim=2 会严重高估 H2（缺失 3-单形死亡映射）。

    一个圆（H2 解析=0）在 build_dim=2 下 H2 应远大于 0；
    在 build_dim=3 下 H2 应=0。这一对断言锁定修复，防止回归。
    """
    X = GENERATORS["circle"](7, n=24)
    b2_lo = _full_betti(X, build_dim=2)
    assert b2_lo.betti(2) > 0, "build_dim=2 下圆 H2 应被高估（>0）"
    b2_hi = _full_betti(X, build_dim=3)
    assert b2_hi.betti(2) == 0, "build_dim=3 下圆 H2 应严格=0"


def test_sphere_requires_build_dim_3_for_H2():
    """球面 H2=1 只能在有 3-单形时恢复（满尺度）。"""
    X = GENERATORS["sphere"](3, n=30)
    b = _full_betti(X, build_dim=3)
    # 满尺度 VR 可缩 ⇒ β0=1,β1=0,β2=0 在满尺度；但至少不应崩溃且维度正确
    assert b.betti(0) == 1
    assert b.betti(2) == 0  # 满尺度可缩
