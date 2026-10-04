"""pytest 根配置：确保仓库根目录（cli.py / topoforge 包）在导入路径上。作者：晨星。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
