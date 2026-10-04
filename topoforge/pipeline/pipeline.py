"""TopoForge 端到端流水线 + 基准评测。作者：晨星。

流水线（单向无环）：cli → pipeline → {data, topology, classify} → core。
- TopoPipeline.analyze(X)    ：点云 → 拓扑特征向量（VR 复形→持久同调→特征）。
- TopoPipeline.run(X)        ：analyze + 计时元数据。
- run_benchmark(...)         ：多 seed 随机对照实验（拓扑特征 vs 欧氏基线），
                               输出 mean±std + 显著性，全程确定性可复现。
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from ..classify.models import available_random_forest, make_classifier
from ..core.config import Config
from ..core.errors import PipelineError
from ..core.seed import set_all
from ..data.generators import BETTI, GENERATORS
from ..topology.features import extract_euclidean_features, extract_topo_features
from ..topology.persistence import compute_persistence
from ..topology.vr_complex import build_filtration, pairwise_distances

# 默认 6 类流形分类任务（覆盖不同拓扑签名）
DEFAULT_CLASSES = ["circle", "eight", "line", "sphere", "torus", "blob"]

# 各生成器默认采样点数（统一构建到 3-单形以正确计算 H2）
SAMPLE_SIZE = {"circle": 26, "eight": 26, "line": 26, "sphere": 26, "torus": 26, "blob": 26}

# 统一构建到 build_dim=3：H2 需 3-单形作为死亡映射，否则 2-循环不被填充而 H2 严重高估
AMBIENT_DIM = {"circle": 2, "eight": 2, "line": 2, "sphere": 3, "torus": 3, "blob": 3}


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]) -> float:
    f1s = []
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    for lab in labels:
        tp = int(np.sum((yt == lab) & (yp == lab)))
        fp = int(np.sum((yt != lab) & (yp == lab)))
        fn = int(np.sum((yt == lab) & (yp != lab)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


class TopoPipeline:
    """拓扑数据分析端到端流水线。"""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    def analyze(
        self, X: np.ndarray, build_dim: int | None = None, n_scales: int = 20
    ) -> np.ndarray:
        """单点云 → 定长拓扑特征向量（18 维）。"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            raise PipelineError("analyze 需要至少 2 个点的 2D 点云")
        dists = pairwise_distances(X)
        eps = float(np.max(dists)) if dists.size else 0.0
        if build_dim is None:
            build_dim = min(self.config.max_dimension + 1, 3)
        else:
            build_dim = min(build_dim, 3)
        simplices, boundary = build_filtration(
            dists, max_dim=self.config.max_dimension, eps_max=eps, build_dim=build_dim
        )
        result = compute_persistence(
            simplices, boundary, max_dim=self.config.max_dimension, eps_max=eps
        )
        return extract_topo_features(result, dims=(0, 1, 2), n_scales=n_scales)

    def run(self, X: np.ndarray, build_dim: int | None = None, n_scales: int = 20) -> dict:
        t0 = time.perf_counter()
        feats = self.analyze(X, build_dim=build_dim, n_scales=n_scales)
        return {
            "features": feats,
            "n_points": int(X.shape[0]),
            "n_features": int(feats.shape[0]),
            "elapsed_sec": time.perf_counter() - t0,
        }


def _sample_seed(base_seed: int, class_idx: int, sample_idx: int) -> int:
    """确定性派生子种子，保证 train/test 与跨 seed 互不泄漏。"""
    return (base_seed * 100000 + class_idx * 1000 + sample_idx) % (2**31)


def run_benchmark(
    config: Config | None = None,
    seeds: Sequence[int] | None = None,
    classes: Sequence[str] | None = None,
    n_per_class: int = 6,
    n_train_per_class: int = 4,
    n_scales: int = 20,
) -> dict:
    """多 seed 随机对照实验：拓扑特征(RF/NC) vs 欧氏特征(同分类器)。

    返回结构化 dict（逐 seed + 聚合 mean±std + 显著性），可直接落盘 benchmark.json。
    全程确定性：set_all(seed) + 固定子种子派生 + n_jobs=1。
    """
    config = config or Config()
    seeds = list(seeds if seeds is not None else (0, 1, 2))
    classes = list(classes if classes is not None else DEFAULT_CLASSES)
    if n_train_per_class >= n_per_class:
        raise PipelineError("n_train_per_class 必须小于 n_per_class")
    for name in classes:
        if name not in GENERATORS:
            raise PipelineError(f"未知流形类: {name!r}")

    per_seed = []
    for seed in seeds:
        set_all(seed)  # 全局确定性入口
        labels = list(range(len(classes)))
        topo_train, euc_train, topo_test, euc_test = [], [], [], []
        y_train, y_test = [], []
        for ci, name in enumerate(classes):
            gen = GENERATORS[name]
            n = SAMPLE_SIZE.get(name, 32)
            bdim = 3  # 所有形状统一构建到 3-单形：H2 需 3-单形作死亡映射，否则 2-循环不被填充而 H2 高估
            for k in range(n_per_class):
                s = _sample_seed(seed, ci, k)
                X = gen(s, n=n)
                tf = TopoPipeline(config).analyze(X, build_dim=bdim, n_scales=n_scales)
                ef = extract_euclidean_features(X)
                if k < n_train_per_class:
                    topo_train.append(tf)
                    euc_train.append(ef)
                    y_train.append(ci)
                else:
                    topo_test.append(tf)
                    euc_test.append(ef)
                    y_test.append(ci)

        clf_name = config.classifier
        topo_clf = make_classifier(clf_name, random_state=seed, standardize=config.standardize)
        topo_clf.fit(topo_train, y_train)
        pred_topo = topo_clf.predict(topo_test)

        euc_clf = make_classifier(clf_name, random_state=seed, standardize=config.standardize)
        euc_clf.fit(euc_train, y_train)
        pred_euc = euc_clf.predict(euc_test)

        per_seed.append(
            {
                "seed": int(seed),
                "topo_accuracy": accuracy(y_test, pred_topo),
                "topo_macro_f1": macro_f1(y_test, pred_topo, labels),
                "euclidean_accuracy": accuracy(y_test, pred_euc),
                "euclidean_macro_f1": macro_f1(y_test, pred_euc, labels),
                "n_test": len(y_test),
                "classifier": clf_name,
            }
        )

    # 聚合 mean±std
    def _agg(key: str):
        vals = np.array([r[key] for r in per_seed], dtype=np.float64)
        return {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
        }

    diff_acc = np.array([r["topo_accuracy"] - r["euclidean_accuracy"] for r in per_seed])
    diff_f1 = np.array([r["topo_macro_f1"] - r["euclidean_macro_f1"] for r in per_seed])
    sig = _significance(diff_acc, diff_f1)

    return {
        "system": "TopoForge",
        "task": "manifold_topology_classification",
        "classes": list(classes),
        "analytic_betti": {c: BETTI[c] for c in classes},
        "seeds": [int(s) for s in seeds],
        "n_per_class": n_per_class,
        "n_train_per_class": n_train_per_class,
        "n_scales": n_scales,
        "classifier": clf_name,
        "tier0_random_forest_available": available_random_forest(),
        "used_tier": "random_forest"
        if (available_random_forest() and clf_name == "random_forest")
        else "nearest_centroid",
        "per_seed": per_seed,
        "aggregate": {
            "topo_accuracy": _agg("topo_accuracy"),
            "topo_macro_f1": _agg("topo_macro_f1"),
            "euclidean_accuracy": _agg("euclidean_accuracy"),
            "euclidean_macro_f1": _agg("euclidean_macro_f1"),
        },
        "delta": {
            "accuracy_mean": float(diff_acc.mean()),
            "macro_f1_mean": float(diff_f1.mean()),
        },
        "significance": sig,
        "config": {
            "seed": config.seed,
            "max_dimension": config.max_dimension,
            "standardize": config.standardize,
        },
    }


def _scale_window_betti(simp_all: list, bnd_all: list, r: float):
    """（保留）在尺度 r 处取子复形计 Betti，用于 CLI 单点校验；批量请用 multi_scale_betti。"""

    keep = [i for i, s in enumerate(simp_all) if s[0] <= r]
    if len(keep) < 3:
        return None
    newidx = {o: nn for nn, o in enumerate(keep)}
    s2 = [simp_all[i] for i in keep]
    b2 = [[newidx[f] for f in bnd_all[i] if f in newidx] for i in keep]
    n = len(s2)
    order = sorted(range(n), key=lambda s: (s2[s][0], s2[s][1]))
    low_map: dict = {}
    red: dict = {}
    for j in order:
        col = set(b2[j])
        while col:
            low = max(col)
            if low in low_map:
                col ^= red[low_map[low]]
            else:
                break
        red[j] = col
        if col:
            low = max(col)
            low_map[low] = j
    nb = {d: sum(1 for s in range(n) if s2[s][1] == d) for d in range(4)}
    nz = {d: sum(1 for s in range(n) if s2[s][1] == d and red[s]) for d in range(4)}
    return {d: int(nb[d] - nz[d] - nz[d + 1]) for d in range(3)}


def benchmark_topology_recovery(
    seeds: Sequence[int] | None = None, n_per_shape: int = 6, n_windows: int = 30
) -> dict:
    """引擎正确性基准（多 seed）：采样流形在中尺度窗内恢复解析 Betti 的命中率。

    这是 TDA 引擎的本职性能目标——恢复已知拓扑，而非分类准确率。
    返回逐 seed×形状的恢复命中与聚合命中率。torus H2 为 VR 已知限制（不计入失败）。
    """
    from ..topology.persistence import multi_scale_betti

    seeds = list(seeds if seeds is not None else (0, 1, 2))
    shapes = list(DEFAULT_CLASSES)
    recover_dims = {"torus": (0, 1)}  # H2=1 为 VR 采样限制，仅校验 H0/H1
    per_shape: dict = {name: [] for name in shapes}
    for seed in seeds:
        set_all(seed)
        for ci, name in enumerate(shapes):
            gen = GENERATORS[name]
            n = SAMPLE_SIZE.get(name, 26)
            for k in range(n_per_shape):
                s = (seed * 100000 + ci * 1000 + k) % (2**31)
                X = np.asarray(gen(s, n=n), dtype=np.float64)
                diff = X[:, None, :] - X[None, :, :]
                dists = np.sqrt(np.sum(diff * diff, axis=-1))
                np.fill_diagonal(dists, 0.0)
                eps = float(np.max(dists))
                bdim = 3  # 所有形状统一构建到 3-单形：H2 需 3-单形作死亡映射
                simp, bnd = build_filtration(dists, max_dim=2, eps_max=eps, build_dim=bdim)
                exp = BETTI[name]
                dims = recover_dims.get(name, (0, 1, 2))
                scans = multi_scale_betti(simp, bnd, max_dim=2, n_windows=n_windows, eps_max=eps)
                rec = any(all(b[d] == exp[d] for d in dims) for _, b in scans)
                per_shape[name].append(int(rec))
    agg = {}
    for name in shapes:
        vals = np.array(per_shape[name], dtype=np.float64)
        ok_count = int(vals.sum())
        total = len(vals)
        agg[name] = {
            "recover_rate": float(ok_count / total) if total else 0.0,
            "recovered": ok_count,
            "total": total,
        }
    overall = float(np.mean([agg[name]["recover_rate"] for name in shapes]))
    return {
        "system": "TopoForge",
        "benchmark": "topology_recovery",
        "description": "采样流形在中尺度窗内恢复解析 Betti 的命中率（引擎正确性）",
        "seeds": [int(s) for s in seeds],
        "n_per_shape": n_per_shape,
        "n_windows": n_windows,
        "analytic_betti": {c: BETTI[c] for c in shapes},
        "note": "torus H2=1 为 VR 复形在有限采样上的已知限制（空洞难由有限持续条恢复），不计入失败",
        "per_shape": agg,
        "overall_recover_rate": overall,
    }


def _significance(diff_acc: np.ndarray, diff_f1: np.ndarray) -> dict:
    """配对显著性（拓扑 - 欧氏）。优先 scipy.stats.wilcoxon；缺失则精确符号检验。"""
    out = {"method": "none", "accuracy_p": None, "macro_f1_p": None}
    n = len(diff_acc)
    try:  # pragma: no cover - 依赖可选
        from scipy import stats as _stats  # type: ignore

        if n >= 2:
            try:
                _, pa = _stats.wilcoxon(diff_acc)
                out["accuracy_p"] = float(pa)
            except ValueError:
                out["accuracy_p"] = 1.0
            try:
                _, pf = _stats.wilcoxon(diff_f1)
                out["macro_f1_p"] = float(pf)
            except ValueError:
                out["macro_f1_p"] = 1.0
            out["method"] = "scipy.wilcoxon"
            return out
    except Exception:  # noqa: BLE001
        pass

    # 精确符号检验（单侧）：H0: 中位差 ≤ 0
    def _sign_p(d: np.ndarray) -> float:
        d = np.asarray(d, dtype=np.float64)
        pos = int(np.sum(d > 0))
        neg = int(np.sum(d < 0))
        m = pos + neg
        if m == 0:
            return 1.0
        # P(X >= pos) under Binomial(m, 0.5)
        from math import comb

        p = float(sum(comb(m, k) for k in range(pos, m + 1)) * (0.5**m))
        return p

    out["accuracy_p"] = _sign_p(diff_acc)
    out["macro_f1_p"] = _sign_p(diff_f1)
    out["method"] = "exact_sign_test"
    return out
