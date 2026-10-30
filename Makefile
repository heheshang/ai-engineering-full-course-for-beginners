.PHONY: help install fmt fmt-check lint lint-fix check test test-offline logs-check clean

PY := uv run
RUFF := $(PY) ruff

help:  ## 显示所有可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## 安装依赖（含 dev 组）
	uv sync

fmt:  ## 统一格式化（全项目）
	$(RUFF) format .

fmt-check:  ## 检查格式化是否符合规范，不修改文件
	$(RUFF) format --check --diff .

lint:  ## 静态检查
	$(RUFF) check .

lint-fix:  ## 静态检查并自动修复可修复项
	$(RUFF) check --fix .

check: fmt-check lint  ## CI 用：格式 + 静态检查

test:  ## 运行全部评测（需要裁判模型凭据）
	$(PY) pytest

test-offline:  ## 只跑离线用例，无需任何 API Key
	EVAL_SKIP_LLM=1 $(PY) pytest -q

logs-check:  ## 静态检查各演示脚本的日志接入
	$(PY) python tools/check_logging.py

clean:  ## 清理缓存与构建产物
	rm -rf .ruff_cache .pytest_cache dist build
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
