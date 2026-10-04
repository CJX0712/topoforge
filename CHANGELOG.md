# Changelog

## [v0.1.0] - 2026-10-04

首个公开发布（作者：晨星）。

### 新增
- **核心引擎**：Vietoris–Rips 复形构建（0/1/2/3 维）+ 标准约简（Edelsbrunner, GF(2)）持久同调。
- **Betti 计算**：`β_d = n_d − nz_d − nz_{d+1}`，正确计算 H₂（须构建到 3-单形作死亡映射）。
- **尺度窗 Betti**：`multi_scale_betti` 单次约简 + 前缀计数，O(n) 逐窗扫描（支持中尺度解析拓扑恢复）。
- **特征向量**：固定 18 维拓扑特征（每维 6 统计）+ 欧氏几何特征（强基线）。
- **分类器**：Tier-0 sklearn RandomForest；Tier-1 纯 numpy 最近质心（sklearn 缺失自动降级）。
- **流水线**：`TopoPipeline`（端到端）、`run_benchmark`（分类效用对照）、`benchmark_topology_recovery`（引擎正确性）。
- **命令行**：`cli.py` 子命令 `demo / bench / classify / validate`。
- **演示**：`examples/run_demo.py` 落盘 `benchmark.json`（recovery / classification / determinism）。
- **测试**：手建精确复形不变量、H₂ build_dim=3 回归、确定性、离线兜底、CLI 冒烟（核心覆盖 ≥80%）。
- **CI**：`.github/workflows/ci.yml`（lint + pytest + demo，矩阵 3.12/3.13）。
- **文档**：`README.md`、`docs/architecture.md`、`docs/model_card.md`。

### 已知限制（诚实声明）
- VR 复形满尺度必可缩 ⇒ 满尺度 β₁=β₂=0 为数学必然。
- torus H₂=1 为 VR 有限采样已知限制，不计入失败。
- 拓扑特征在几何丰富分类任务上不敌欧氏特征（TDA 已知性质）。

### 依赖
- 运行时：numpy, scipy, scikit-learn。
- 开发/CI：pytest, ruff==0.16.10。
