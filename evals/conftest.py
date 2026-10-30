"""pytest 共享 fixtures 与报告采集。

这里放**测试框架层面**的装配（fixture、标记、路径、结果采集），
具体的裁判模型配置与评测数据集放在 ``eval_config.py``，
以便非 pytest 场景（比如独立评测脚本）也能复用；
报告渲染逻辑放在 ``report.py``。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

# 让同目录的 eval_config / sut 可被直接 import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_config import (
    build_judge_model,
    judge_model_name,
    judge_provider,
    llm_judge_available,
    threshold,
)
from report import REGISTRY, REPORT_DIR, configure_reporting, finalize_reports, record_report

# 召回条数，与 RAGDemo.py 的 TOP_K 保持一致
TOP_K = 3

# 本次 pytest 会话的起始时间（用于计算总耗时）
_SESSION_STARTED_AT = time.time()


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


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    """采集每个用例的执行结果，用于生成评测报告。

    用 ``hookwrapper`` 而不是直接实现 ``pytest_runtest_makereport``，
    是因为 setup / call / teardown 三个阶段都会触发这个钩子，
    需要拿到「完整报告」才能判断用例最终状态。
    """
    outcome = yield
    report = outcome.get_result()
    record_report(item, report)


def pytest_configure(config: pytest.Config) -> None:
    """注册自定义标记，并让 DeepEval 进入「运行中」状态。

    DeepEval 的 pytest 插件在 ``pytest_sessionstart`` 阶段会检查
    ``DEEPEVAL=1`` 来决定是否创建 test run——必须在这里（sessionstart 之前）
    设置，否则 LLM 评判用例的指标不会进入 test run，报告里就没有明细。
    """
    config.addinivalue_line("markers", "llm_judge: 需要 LLM 裁判模型的用例（跳过 API Key 检查）")
    config.addinivalue_line("markers", "slow: 耗时较长的用例")

    # DeepEval 插件读的是 ``config.getoption("file_or_dir")[0]``，
    # 而这个选项保存的是**原始命令行位置参数**——直接跑 ``pytest``
    # （不带路径）时它是空列表，会触发 IndexError。
    # 这里用 pytest 已解析好的 ``config.args`` 补齐。
    if not config.getoption("file_or_dir", None) and config.args:
        config.option.file_or_dir = list(config.args)

    identifier = config.getoption("eval_identifier", None)
    if identifier:
        # DeepEval 的 pytest 插件在 sessionstart 时用 ``--identifier``
        # 给 test run 打标签；这里直接写进 option，插件就能读到。
        config.option.identifier = identifier

    configure_reporting(REPORT_DIR, identifier=identifier)


def pytest_addoption(parser: pytest.Parser) -> None:
    """注册命令行选项。"""
    parser.addoption(
        "--eval-identifier",
        action="store",
        default=None,
        help="本次评测运行的标识，会写进报告标题与 DeepEval test run",
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """会话结束时生成报告。

    这样无论用 ``pytest`` 还是 ``python evals/run_eval.py`` 跑，
    ``evals/reports/`` 里都会有一份完整报告。
    """
    if not REGISTRY.outcomes:
        return

    duration = time.time() - _SESSION_STARTED_AT
    finalize_reports(
        REPORT_DIR,
        identifier=session.config.getoption("eval_identifier", None) or "course",
        duration=duration,
        extra={
            "judge_model": f"{judge_provider()}/{judge_model_name()}",
            "threshold": f"{threshold():.2f}",
            "exit_code": int(exitstatus),
        },
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """没有裁判模型凭据时，自动跳过标记了 ``llm_judge`` 的用例。

    这样在 CI 或本地没配 Key 的环境里，
    离线用例仍能正常跑完，而不会因为 LLM 用例报错中断。
    """
    if llm_judge_available():
        return

    skip = pytest.mark.skip(
        reason="未配置裁判模型凭据（设 DEEPSEEK_API_KEY 或 EVAL_JUDGE_PROVIDER=ollama）"
    )
    for item in items:
        if "llm_judge" in item.keywords:
            item.add_marker(skip)
