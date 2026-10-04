"""TopoForge 错误码体系（E100~E500）。作者：晨星。"""

from __future__ import annotations


class TopoForgeError(Exception):
    """所有 TopoForge 错误的基类。"""

    code = "E000"


class ConfigError(TopoForgeError):
    code = "E100"


class DataError(TopoForgeError):
    code = "E200"


class TopologyError(TopoForgeError):
    code = "E300"


class ClassifyError(TopoForgeError):
    code = "E400"


class PipelineError(TopoForgeError):
    code = "E500"
