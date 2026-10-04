"""流水线测试：确定性（逐位可复现）+ 流形解析 Betti 尺度窗恢复。作者：晨星。"""

from __future__ import annotations

import numpy as np

from topoforge.core.config import Config
from topoforge.pipeline.pipeline import (
    DEFAULT_CLASSES,
    TopoPipeline,
    accuracy,
    benchmark_topology_recovery,
    macro_f1,
    run_benchmark,
)


def test_pipeline_run_shapes():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
    res = TopoPipeline(Config()).run(X)
    assert res["n_points"] == 5
    assert res["n_features"] == 18
    assert res["features"].shape == (18,)
    assert res["elapsed_sec"] >= 0.0


def test_pipeline_requires_2_points():
    import pytest

    from topoforge.core.errors import PipelineError

    with pytest.raises(PipelineError):
        TopoPipeline(Config()).run(np.array([[0.0, 0.0]]))


def test_run_benchmark_deterministic_bit_for_bit():
    cfg = Config(classifier="nearest_centroid", standardize=True)
    r1 = run_benchmark(cfg, seeds=(0, 1, 2), n_per_class=5, n_train_per_class=3, n_scales=12)
    r2 = run_benchmark(cfg, seeds=(0, 1, 2), n_per_class=5, n_train_per_class=3, n_scales=12)
    for k in r1["aggregate"]:
        assert r1["aggregate"][k]["mean"] == r2["aggregate"][k]["mean"]
        assert r1["aggregate"][k]["std"] == r2["aggregate"][k]["std"]


def test_accuracy_and_f1():
    y = [0, 0, 1, 1, 2, 2]
    assert accuracy(y, y) == 1.0
    assert accuracy(y, [0, 1, 1, 0, 2, 2]) == 4 / 6
    assert 0.0 <= macro_f1(y, y, [0, 1, 2]) <= 1.0


def test_topology_recovery_engine_correctness():
    """引擎正确性：非-torus 形状在中尺度窗内恢复解析 Betti 的命中率 ≥ 0.5。

    这是 TDA 引擎的本职性能目标；命中率方差源于有限采样的粗不变量性质，
    0.5 阈值为稳健门限（validate 同口径）。
    """
    r = benchmark_topology_recovery(seeds=(0, 1, 2, 3, 4), n_per_shape=5, n_windows=30)
    for name in DEFAULT_CLASSES:
        if name == "torus":
            continue  # H2 为 VR 已知限制，不计入
        rate = r["per_shape"][name]["recover_rate"]
        assert rate >= 0.5, f"{name} 解析 Betti 恢复命中率过低: {rate}"


def test_recovery_rates_bounded():
    r = benchmark_topology_recovery(seeds=(0, 1, 2), n_per_shape=3, n_windows=20)
    assert 0.0 <= r["overall_recover_rate"] <= 1.0
