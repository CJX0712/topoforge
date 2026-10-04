# TopoForge 踩坑库（实测）

> 全部为本次交付实测记录。下次做 TDA / 持久同调相关系统先查此表。

---

## T. 持久同调 / 复形维度（本域最隐蔽）

| 症状 | 根因 | 修法 |
|------|------|------|
| 2D 形状（圆/8字/线）H2 被严重高估（圈 H2=2300，真值应为 0） | **`build_dim=2` 用于 2D 流形**：只构建 0/1/2 单形，2-循环没有 3-单形作"死亡映射"，永远不被填充 ⇒ H2 虚高 | **计算 H2 必须 `build_dim=3`**（对所有形状，含 2D）。“2D 形状 H2≡0 所以跳过 3-单形”是错误的——即便要证 H2=0，也需要 3-单形作死亡映射。已锁回归测试 `tests/test_vr_complex.py::test_build_dim_2_overcounts_H2_regression` |
| 满尺度（eps=max）下 β1=β2=0 被误判为 bug | VR 复形在满尺度必为可缩空间 ⇒ 满尺度 β1=β2=0 是**数学必然** | 解析拓扑须在中尺度窗内由有限持续条恢复；引擎按此口径校验（`multi_scale_betti` 沿尺度扫描），不在满尺度评估 |
| torus H2=1 恢复不出来 | VR 复形在有限采样下，空腔难由有限持续条恢复 ⇒ H2 是**已知限制**，非引擎缺陷 | torus 仅校验可恢复的 H0/H1，H2 不计入失败；模型卡/README 显式声明 |
| `persistence_summary` 调用即崩溃 `NameError: out` | 函数内只对 `out[d]` 赋值却从未 `out = {}` 初始化 | 进入循环前 `out: dict = {}`；ruff F821 抓出 |
| 手建复形三角形边界写成"顶点下标"而非"边单形下标" | 约简的边界矩阵是**单形-单形**邻接，不是顶点-单形；三角形 (i,j,k) 的边界是三条边单形 (i,j)/(i,k)/(j,k) 的下标 | 用 `make_complex` 辅助：按维序建表，边界取各真面的单形下标（`tests/test_persistence.py` 已固化为范式） |
| 朴素 `compute_persistence` 逐窗重复约简 O(窗×约简) 极慢 | 每次都从头约简整个复形 | `multi_scale_betti`：一次约简 + 前缀计数 O(n) 推各窗 Betti（约简按 (value,dim) 顺序，列只依赖更早列） |

## G. 推送 / 工具链（跨 forge 复用）

| 症状 | 根因 | 修法 |
|------|------|------|
| `gh_push.py` 首次运行报「无可用推送通道」 | **先探测后建仓时序 bug**：新建仓库时 `gh api` 404 → `api=False`，`git ls-remote` 404 → `git=False`，而 `ensure_repo` 已建仓成功，但探测在建仓前做，通道判定已定 | **原样重跑脚本即可**（幂等；仓库已存在后探测返回 200），本次重跑走 Git Data API L2 推送成功（AttribForge/PTQForge 同款，必现） |
| `gh_push.py` L1 `git push` 报 `No such remote: 'origin'` | L1 内 `c[3] != "remove"` 校验的是**错误下标**（`git remote remove origin` 的 `c[3]==\"remote\"` 不是 `"remove\"`），origin 不存在时 remove 失败被误判为致命，跳过了 add/push | 本地先 `git remote add origin <url>` 再跑；或依赖 L2 Git Data API 兜底（本次即走 L2 成功） |
| 宣称「ruff 绿」但 CI 实为 `ruff check . || true` | lint 被降级成装饰品，且未钉版 ⇒ 默认规则集随版本漂移 | pyproject 钉规则集 + `ruff==0.16.10`；lint 升为硬门禁（CI/Dockerfile/Makefile 不再 `|| true`）；本次实跑清零 17 项告警（E741/UP/F401/F841/F821 等） |

## S. 确定性 / 采样

| 症状 | 根因 | 修法 |
|------|------|------|
| 单 seed 下解析 Betti 恢复命中率翻面（torus 偶发 0/1） | 有限采样下拓扑特征方差大；单点评估脆弱 | 多 seed（≥3）× 多 per_shape 报命中率，门限 0.5 稳健；`validate` 与 `benchmark_topology_recovery` 同口径 |
| sklearn 不可用时分类器崩溃 | 紧耦合导入 | `classify/models.py` 用 `try/except` 隔离 sklearn；`make_classifier` 在 RF 不可用时**自动降级**纯 numpy 最近质心（Tier-1 零下载兜底） |
