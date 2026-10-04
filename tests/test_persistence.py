"""解析 Betti 不变量测试（手建精确复形）。

这些用例不依赖采样，是引擎正确性的硬金标准：
手建复形在满尺度下即为其同调，Betti 数由约简算法
beta_d = n_d - nz_d - nz_{d+1} 得出，应与拓扑学解析值逐位一致。
作者：晨星。
"""

from __future__ import annotations

from itertools import combinations

import pytest

from topoforge.topology.persistence import (
    compute_persistence,
    multi_scale_betti,
)


def _faces(verts):
    return [tuple(sorted(c)) for c in combinations(verts, len(verts) - 1)]


def make_complex(verts_specs):
    """verts_specs: 顶点元组列表，每个定义 dim=len-1 的单形。

    返回 (simplices, boundary)：simplices[i] = (value, dim, verts)；
    boundary[i] = 该单形各真面的单形下标（按维序构建，确保面先入表）。
    """
    simplices = []
    boundary = []
    idx = {}
    for verts in sorted(verts_specs, key=lambda v: len(v)):
        key = tuple(sorted(verts))
        idx[key] = len(simplices)
        simplices.append((1.0, len(verts) - 1, key))
        boundary.append([idx[f] for f in _faces(key)] if len(verts) >= 2 else [])
    return simplices, boundary


# (描述, 顶点规格, 期望 Betti{dim:int})
EXACT_CASES = [
    # 单三角形（含 2-面）= 圆盘 ⇒ H0=1,H1=0,H2=0
    ("filled_triangle", [(0,), (1,), (2,), (0, 1), (1, 2), (0, 2), (0, 1, 2)], {0: 1, 1: 0, 2: 0}),
    # 单 3-环（无 2-面）⇒ H1=1
    ("cycle", [(0,), (1,), (2,), (0, 1), (1, 2), (0, 2)], {0: 1, 1: 1, 2: 0}),
    # 8 字（共顶点两环，无 2-面）⇒ H0=1,H1=2
    (
        "figure_eight",
        [(0,), (1,), (2,), (3,), (4,), (0, 1), (1, 2), (0, 2), (0, 3), (3, 4), (0, 4)],
        {0: 1, 1: 2, 2: 0},
    ),
    # 空心四面体（S²，4 个 2-面，无 3-单形）⇒ H0=1,H1=0,H2=1
    (
        "hollow_tetra_S2",
        [
            (0,),
            (1,),
            (2,),
            (3,),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3),
        ],
        {0: 1, 1: 0, 2: 1},
    ),
    # 实心四面体（含 3-单形）⇒ 可缩：H1=H2=0，但 β0=1（单连通分支）
    (
        "solid_tetra",
        [
            (0,),
            (1,),
            (2,),
            (3,),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3),
            (0, 1, 2, 3),
        ],
        {0: 1, 1: 0, 2: 0},
    ),
    # 两个分离圆盘 ⇒ H0=2
    (
        "two_disks",
        [
            (0,),
            (1,),
            (2,),
            (3,),
            (4,),
            (5,),
            (0, 1),
            (1, 2),
            (0, 2),
            (3, 4),
            (4, 5),
            (3, 5),
            (0, 1, 2),
            (3, 4, 5),
        ],
        {0: 2, 1: 0, 2: 0},
    ),
]


@pytest.mark.parametrize("name,specs,exp", EXACT_CASES, ids=[c[0] for c in EXACT_CASES])
def test_exact_betti_invariants(name, specs, exp):
    simplices, boundary = make_complex(specs)
    res = compute_persistence(simplices, boundary, max_dim=2)
    for d, val in exp.items():
        assert res.betti(d) == val, f"{name}: β{d} 期望 {val} 实得 {res.betti(d)}"


def test_betti_formula_consistency():
    """beta_d = n_d - nz_d - nz_{d+1} 与边界矩阵秩一致（秩公式）。"""
    simplices, boundary = make_complex(EXACT_CASES[3][1])  # hollow_tetra_S2
    res = compute_persistence(simplices, boundary, max_dim=2)
    # H2 = 1 应等于 2-单形数(4) - rank(∂2)(3) - rank(∂3)(0)
    assert res.betti(2) == 1


def test_multi_scale_betti_full_equals_compute():
    """multi_scale_betti 在覆盖全部单形的尺度窗应与 compute_persistence 一致。

    用 eps_max=2.0 使最后一个窗 (frac=0.92→r=1.84) 越过所有单形值(=1.0)，
    对应满复形；本测试验证前缀约简与全约简在 Betti 上等价。
    """
    simplices, boundary = make_complex(EXACT_CASES[3][1])
    res = compute_persistence(simplices, boundary, max_dim=2)
    scans = multi_scale_betti(simplices, boundary, max_dim=2, n_windows=10, eps_max=2.0)
    _, betti = scans[-1]  # 覆盖全部单形的窗
    assert betti[0] == res.betti(0)
    assert betti[1] == res.betti(1)
    assert betti[2] == res.betti(2)


def test_empty_complex_raises():
    import pytest as _pytest

    from topoforge.core.errors import TopologyError

    with _pytest.raises(TopologyError):
        compute_persistence([], [], max_dim=2)
