"""点云文件载入（.npy / .csv）。作者：晨星。"""

from __future__ import annotations

import csv
import os

import numpy as np

from ..core.errors import DataError


def load_point_cloud(path: str) -> np.ndarray:
    """载入点云，支持 .npy 与 .csv（无表头，每行一个样本）。"""
    if not os.path.exists(path):
        raise DataError(f"文件不存在: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".csv":
        rows: list[list[float]] = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                rows.append([float(v) for v in row])
        arr = np.array(rows, dtype=np.float64)
    else:
        raise DataError(f"不支持的格式: {ext}（仅 .npy/.csv）")
    if arr.ndim != 2:
        raise DataError(f"点云必须为 2D (n, d)，收到 {arr.ndim}D")
    if not np.all(np.isfinite(arr)):
        raise DataError("点云含非有限坐标（NaN/inf）")
    return arr
