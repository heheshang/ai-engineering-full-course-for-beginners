"""pytest 共享 fixtures。

这里只放**测试框架层面**的装配（fixture、标记、路径），
具体的裁判模型配置与评测数据集放在 ``eval_config.py``，
以便非 pytest 场景（比如独立评测脚本）也能复用。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 让同目录的 eval_config / sut 可被直接 import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_config import (  # noqa: E402
    build_judge_model,
    llm_judge_available,
    threshold,
)

# 召回条数，与 RAGDemo.py 的 TOP_K 保持一致
TOP_K = 3


@pytest.fixture(scope="session")
def judge():
    """构造裁判模型，整个测试会话共用一个实例。

    构造开销不小（可能触发模型初始化），因此用 session 作用域。
    """
    return build_judge_model()


@pytest.fixture(scope="session")
def eval_threshold() -> float:
    """指标默认通过阈值。"""
    return threshold()


@pytest.fixture(scope="session")
def retriever():
    """向量库检索器。

    使用内存向量库：评测不写磁盘，不污染项目的 ``chroma_db/`` 目录。

    ``search_kwargs={"k": TOP_K}`` 与 ``RAGDemo.py`` 保持一致（TOP_K=3）。
    """
    from sut import build_rag_pipeline

    vectorstore = build_rag_pipeline()
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


@pytest.fixture(scope="session")
def rag_agent(retriever):
    """RAG Agent（对应 ``src/01-rag-demo/RAGDemo.py``）。"""
    from sut import build_rag_agent

    return build_rag_agent(retriever)


def pytest_configure(config: pytest.Config) -> None:
    """注册自定义标记。"""
    config.addinivalue_line(
        "markers", "llm_judge: 需要 LLM 裁判模型的用例（跳过 API Key 检查）"
    )
    config.addinivalue_line("markers", "slow: 耗时较长的用例")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """没有裁判模型凭据时，自动跳过标记了 ``llm_judge`` 的用例。

    这样在 CI 或本地没配 Key 的环境里，
    离线用例仍能正常跑完，而不会因为 LLM 用例报错中断。
    """
    if llm_judge_available():
        return

    skip = pytest.mark.skip(reason="未配置裁判模型凭据（设 DEEPSEEK_API_KEY 或 EVAL_JUDGE_PROVIDER=ollama）")
    for item in items:
        if "llm_judge" in item.keywords:
            item.add_marker(skip)
