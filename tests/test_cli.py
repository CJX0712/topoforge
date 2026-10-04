"""CLI 冒烟测试：各子命令可运行且返回正确退出码。作者：晨星。"""

from __future__ import annotations

import sys

import cli as cli_mod


def _run(argv):
    saved = sys.argv
    sys.argv = ["cli"] + argv
    try:
        return cli_mod.main(argv)
    finally:
        sys.argv = saved


def test_cli_classify_manifold():
    # 内置流形演示不应崩溃，返回 0
    rc = _run(["classify", "--manifold", "circle", "--n", "20"])
    assert rc == 0


def test_cli_classify_input_file(tmp_path):
    import numpy as np

    p = tmp_path / "pc.npy"
    np.save(p, np.random.default_rng(0).normal(0, 1, (20, 2)))
    rc = _run(["classify", "--input", str(p)])
    assert rc == 0


def test_cli_validate():
    # 足够样本量（5 seed × 4/shape=20）以稳健跨过 0.5 恢复命中率门限
    rc = _run(["validate", "--seeds", "0", "1", "2", "3", "4", "--n-per-shape", "4"])
    assert rc == 0  # 非-torus 形状恢复命中率门限内即通过


def test_cli_demo(tmp_path):
    out = tmp_path / "bm.json"
    rc = _run(
        ["demo", "--seeds", "0", "1", "--n-per-class", "4", "--n-train", "3", "--out", str(out)]
    )
    assert rc == 0
    assert out.exists()


def test_cli_bench(capsys):
    rc = _run(["bench", "--seeds", "0", "--n-per-class", "4", "--n-train", "3"])
    assert rc == 0
