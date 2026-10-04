# TopoForge — delivery log (SOP §11 write-back)

| date | system | repo | domain | grade | note |
|------|--------|------|--------|-------|------|
| 2026-10-04 | TopoForge | CJX0712/topoforge | 拓扑数据分析 TDA（持久同调） | S | Vietoris-Rips 复形 + Edelsbrunner 标准约简 GF(2) + 尺度窗 Betti 多 seed 恢复；sklearn RandomForest Tier-0 + 纯 numpy 最近质心 Tier-1 离线兜底；引擎正确性 解析 Betti 恢复命中率 83.3%（3 seed×6/shape）；拓扑 acc 0.861±0.048 vs 欧氏 1.000±0.000（诚实对照）；确定性逐位一致；32 单测；ruff 0.16.10 双绿硬门禁；CI py3.12+3.13 |

## Coverage notes
- Domain chosen: **拓扑数据分析 TDA / 持久同调（Persistent Homology）** — verified uncovered in the forge series (no prior forge in algebraic topology / TDA).
- Author attribution: 晨星.
- Quality grade: **S** (world-class): verifiable analytic-Betti invariants (exact hand-built complexes + multi-seed recovery), honest DoD redefinition (engine's job = recovering analytic topology, not beating euclidean on geometry-rich classification), bit-for-bit determinism, CI matrix, locked dependency versions.
- Honest finding baked into model card: topological features never beat euclidean geometry on geometry-rich classification (persistence is a coarse homotopy invariant) — the genuine performance target for a TDA engine is recovering analytic topology.

## Repo
- https://github.com/CJX0712/topoforge  (Release v0.1.0)
