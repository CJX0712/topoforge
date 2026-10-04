# TopoForge 运行时镜像（确定性、零缓存外部下载）。作者：晨星。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 依赖先装（利用层缓存）
COPY requirements.txt requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 源码
COPY . .

# 安装为可编辑包 + 开发依赖（ruff/pytest 供 CI 使用）
RUN pip install --no-cache-dir -e ".[dev]"

# 默认运行端到端演示（可覆盖为 topoforge <subcmd>）
CMD ["python", "examples/run_demo.py"]
