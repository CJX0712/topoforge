# TopoForge 架构设计

> 单向无环数据流：`cli → pipeline → {data, topology, classify} → core`。作者：晨星。

## 1. 分层职责

```
┌─────────────────────────────────────────────────────────────┐
│ cli.py   (argparse 入口：demo / bench / classify / validate) │
└───────────────────────────┬─────────────────────────────────┘
                            │ 调用
┌───────────────────────────▼─────────────────────────────────┐
│ pipeline/                                                  │
│   ├─ pipeline.py   TopoPipeline.analyze/run                 │
│   │                run_benchmark (分类效用对照)              │
│   │                benchmark_topology_recovery (引擎正确性)  │
│   └─ __init__.py                                            │
└───────┬───────────────┬───────────────────┬─────────────────┘
        │               │                   │
┌───────▼──────┐ ┌───────▼───────┐ ┌─────────▼────────┐
│ data/        │ │ topology/     │ │ classify/        │
│ generators   │ │ vr_complex    │ │ models (RF/NC)   │
│ loaders      │ │ persistence   │ │                  │
│              │ │ features      │ │                  │
└───────┬──────┘ └───────┬───────┘ └─────────┬────────┘
        │                │                   │
┌───────▼────────────────▼───────────────────▼───────────────┐
│ core/   config · seed(确定性) · errors · types · interfaces │
└─────────────────────────────────────────────────────────────┘
```

每一层只依赖其下方的层，禁止反向依赖或同层环，保证可独立测试与替换。

## 2. 计算流水线（一次分析）

1. **输入**：点云 `X (n, d)`。
2. **距离**：`pairwise_distances` → 欧氏距离矩阵 `D`。
3. **过滤复形**：`build_filtration(D, max_dim=2, build_dim=3, eps_max=max(D))`
   构建 0/1/2/3 维单形与边界表；排序约定 `(value, dim)` 保证面先于余面。
4. **持久同调**：`compute_persistence` 用标准约简（Edelsbrunner, GF(2)）约简边界矩阵，
   由 `β_d = n_d − nz_d − nz_{d+1}` 得 Betti 数，并产出持久同调图与本质类。
5. **特征**：`extract_topo_features` 抽取固定 18 维向量（每维 d∈{0,1,2}：有限点/本质点/
   总持续性/最大持续性/持续性熵/Betti 曲线积分）。

## 3. 确定性策略

- `core.seed.set_all(seed)` 一次性设齐 `random` / numpy legacy / numpy Generator。
- 分类基准用确定性子种子派生 `_sample_seed(seed, class_idx, sample_idx)`，保证
  train/test 与跨 seed 互不泄漏。
- `n_jobs=1`（Windows 受限环境多进程易崩，见 pitfalls）。
- 二次运行 `benchmark.json` 核心指标 **mean±std 逐位一致** 作为 DoD 硬门槛。

## 4. 统计严谨性

- 多 seed（默认 3，demo 可配置）随机对照，输出 `mean ± std`。
- 配对显著性：优先 `scipy.stats.wilcoxon`，缺失则**精确符号检验**兜底。
- 引擎正确性以**恢复命中率**（多 seed × 多形状 × 多尺度窗）衡量，而非单次点估计。

## 5. 可替换的顶级开源集成点

- **分类器**：`classify/models.py` Tier-0 sklearn `RandomForestClassifier`；
  sklearn 缺失自动降级 Tier-1 纯 numpy 最近质心（离线可跑）。
- **依赖隔离**：sklearn 通过 `try/except` 隔离导入，缺失不阻断整包。

## 6. 已知数学限制（诚实记录）

| 现象 | 原因 | 处理 |
|---|---|---|
| 满尺度 β₁=β₂=0 | VR 复形在 eps=max 必可缩 | 非 bug；尺度窗口径校验 |
| torus H₂=1 难恢复 | 有限采样下空洞无有限持续条 | 不计入失败，模型卡声明 |
| 拓扑特征分类不敌欧氏 | 持久是同伦粗不变量 | 诚实对照，不夸大 |
