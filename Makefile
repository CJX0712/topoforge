# TopoForge 开发任务入口。作者：晨星。
# 用法：make <target>

PY := .venv/Scripts/python.exe
PIP := .venv/Scripts/pip.exe

.PHONY: help venv install dev test lint fmt ci demo clean

help:
	@echo "TopoForge 可用目标:"
	@echo "  make venv     创建隔离虚拟环境(.venv)"
	@echo "  make install  安装运行时依赖 + 本包"
	@echo "  make dev      安装开发依赖(pytest, ruff)"
	@echo "  make test     运行 pytest"
	@echo "  make lint     ruff check + format --check"
	@echo "  make fmt      ruff format（自动格式化）"
	@echo "  make ci       本地等价 CI：lint + test"
	@echo "  make demo     运行端到端演示"
	@echo "  make clean    删除 .venv 与缓存"

venv:
	python -m venv .venv

install: venv
	$(PIP) install -U pip
	$(PIP) install -e .

dev: install
	$(PIP) install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

fmt:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix . || true

ci: lint test

demo:
	$(PY) examples/run_demo.py

clean:
	rm -rf .venv .pytest_cache .ruff_cache __pycache__
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
