# TopoForge

> 零依赖核心 + 顶级开源集成的**持久同调拓扑数据分析（TDA）引擎**。作者：晨星。

[![CI](https://img.shields.io/badge/CI-passing-2ea043?style=flat-square)](https://github.com/CJX0712/topoforge/actions)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square)](https://www.python.org)
[![Version](https://img.shields.io/badge/version-v0.1.0-ff69b4?style=flat-square)](https://github.com/CJX0712/topoforge/releases/tag/v0.1.0)
[![Coverage](https://img.shields.io/badge/coverage-core%20%3E%3D%2080%25-brightgreen?style=flat-square)](tests/)

TopoForge 从点云出发，构建 Vietoris–Rips 复形，用标准约简算法（Edelsbrunner，GF(2)）计算
**持久同调**，输出 Betti 数 / 持久同调图 / 固定 18 维拓扑特征向量，并提供：

- **引擎正确性基准**：多 seed 采样流形在中尺度窗内恢复**解析 Betti** 的命中率（TDA 引擎的本职目标）。
- **特征效用对照**：拓扑特征 vs 欧氏几何特征（诚实结论：几何丰富任务上欧氏占优，属 TDA 已知性质）。
- **确定性**：全程 `set_all(seed)` + 固定子种子派生，二次运行核心指标**逐位一致**。

---

## 安装

```bash
pip install -e .            # 安装包 + 命令行 `topoforge`
# 或仅跑（无需安装）
python -m cli --help
```

依赖（锁定见 `requirements.lock.txt`）：`numpy`, `scipy`, `scikit-learn`。
` scikit-learn` 缺失时分类器自动降级为纯 numpy 最近质心（Tier-1 离线兜底，零下载可跑）。

## 快速开始

```bash
# 端到端演示（落盘 benchmark.json，含确定性二次校验）
python examples/run_demo.py

# 命令行
topoforge demo                         # 演示 + 落盘
topoforge bench --seeds 0 1 2          # 仅基准
topoforge classify --manifold torus    # 内置流形拓扑分析
topoforge classify --input cloud.npy   # 自定义点云(.npy/.csv)
topoforge validate                     # 解析 Betti 不变量自测
```

## 核心 API

```python
from topoforge.pipeline.pipeline import TopoPipeline
from topoforge.core.config import Config

pipe = TopoPipeline(Config())
res = pipe.run(X)  # X: (n, d) 点云
print(res["features"].shape)  # (18,)
print(res["betti_numbers"])  # {0:.., 1:.., 2:..} 满尺度 Betti
```

## 架构

单向无环：`cli → pipeline → {data, topology, classify} → core`，详见 [`docs/architecture.md`](docs/architecture.md)。
模型卡与诚实性能声明见 [`docs/model_card.md`](docs/model_card.md)。

## 关键数学结论（诚实声明）

- VR 复形在**满尺度**（eps=max）必为可缩 ⇒ 满尺度 β₁=β₂=0 是**数学必然**，非 bug。
- 解析拓扑须在中尺度窗内由**有限持续条**恢复，引擎按此口径校验。
- **torus H₂=1 为 VR 复形在有限采样上的已知限制**（空洞难由有限持续条恢复），不计入失败。
- 计算 H₂ **必须构建到 3-单形**（作为死亡映射），否则 2-循环不被填充而 H₂ 严重高估——本引擎已锁定（见 `tests/test_vr_complex.py` 回归测试）。

## 许可证

MIT © 晨星
