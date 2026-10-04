"""TopoForge 端到端演示（落盘 benchmark.json）。作者：晨星。

一键复现：
    python examples/run_demo.py
产出 benchmark.json，含三部分：
  1) topology_recovery  —— 引擎正确性：多 seed 采样流形恢复解析 Betti 的命中率（引擎本职）
  2) classification     —— 特征效用：拓扑特征 vs 欧氏几何特征（诚实对照，几何在几何丰富任务上占优）
  3) determinism_pass   —— 二次运行核心指标逐位一致
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from topoforge.core.config import Config
from topoforge.core.seed import set_all
from topoforge.pipeline.pipeline import (
    benchmark_topology_recovery,
    run_benchmark,
)

SEEDS = [0, 1, 2]
N_PER_CLASS = 6
N_TRAIN = 4
N_SCALES = 20
OUT = Path(__file__).resolve().parent.parent / "benchmark.json"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    cfg = Config(classifier="random_forest", standardize=True)
    report: dict = {"system": "TopoForge", "author": "晨星", "version": "0.1.0"}

    # 1) 引擎正确性基准（多 seed 恢复解析 Betti）
    set_all(20261004)
    t0 = time.perf_counter()
    recovery = benchmark_topology_recovery(seeds=SEEDS, n_per_shape=N_PER_CLASS)
    recovery["elapsed_sec"] = time.perf_counter() - t0
    report["topology_recovery"] = recovery

    # 2) 特征效用对照（拓扑 vs 欧氏）
    set_all(20261004)
    t0 = time.perf_counter()
    cls = run_benchmark(
        cfg, seeds=SEEDS, n_per_class=N_PER_CLASS, n_train_per_class=N_TRAIN, n_scales=N_SCALES
    )
    cls["elapsed_sec"] = time.perf_counter() - t0
    report["classification"] = cls

    # 3) 确定性二次校验
    set_all(20261004)
    recovery2 = benchmark_topology_recovery(seeds=SEEDS, n_per_shape=N_PER_CLASS)
    cls2 = run_benchmark(
        cfg, seeds=SEEDS, n_per_class=N_PER_CLASS, n_train_per_class=N_TRAIN, n_scales=N_SCALES
    )
    det_rec = round(recovery["overall_recover_rate"], 10) == round(
        recovery2["overall_recover_rate"], 10
    )
    det_cls = all(
        round(cls["aggregate"][k]["mean"], 10) == round(cls2["aggregate"][k]["mean"], 10)
        and round(cls["aggregate"][k]["std"], 10) == round(cls2["aggregate"][k]["std"], 10)
        for k in cls["aggregate"]
    )
    report["determinism_pass"] = bool(det_rec and det_cls)

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    rr = recovery["overall_recover_rate"]
    a = cls["aggregate"]
    print("=" * 70)
    print(" TopoForge demo · 作者=晨星 · v0.1.0")
    print("=" * 70)
    print(
        f" [引擎正确性] 解析 Betti 恢复命中率 = {rr * 100:.1f}%  "
        f"(over {len(SEEDS)} seeds × {N_PER_CLASS}/shape，torus H2 为 VR 已知限制)"
    )
    print(
        f" [特征效用]   拓扑 acc={a['topo_accuracy']['mean']:.3f}  vs  "
        f"欧氏 acc={a['euclidean_accuracy']['mean']:.3f}  (欧氏在几何丰富任务占优，属 TDA 已知性质)"
    )
    print(
        f" [确定性]     二次运行核心指标逐位一致 = {'PASS ✅' if report['determinism_pass'] else 'FAIL ⚠️'}"
    )
    print(f" 落盘: {OUT}")
    return 0 if report["determinism_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
