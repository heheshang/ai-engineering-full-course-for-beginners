"""Agent 与工具调用评测。

覆盖的指标
----------
============================  ==============================================
指标                           评估什么
============================  ==============================================
``ToolCorrectnessMetric``     工具选得对不对、参数填得对不对
``TaskCompletionMetric``      任务是否真的完成了
``ToolUseMetric``             工具使用是否恰当（该用的时候用了）
``GEval``                     自定义标准：答案精确正确、流程是否合理
============================  ==============================================

对应被测系统
------------
- ``src/07-tools-and-agents-demo/MathTools_Agents_Usecase.py``
  （``calculator`` + ``language_model`` 两个工具）
- ``src/07-tools-and-agents-demo/SQLDatabaseTool_SqlAgent.py``
  （``SQLDatabaseToolkit`` 的 SQL 工具）

工具调用评测的核心是**「工具选择」**而非「答案对错」：
模型算出正确答案有两条路——自己算对，或调工具算对。
后者更可靠（不会口算失误），所以要单独评「该不该调、调对没有」。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# 把 src 加入模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_config import build_g_eval, threshold  # noqa: E402


# --------------------------------------------------------------------------- #
# 数学工具 Agent（对应 MathTools_Agents_Usecase.py）
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def math_agent():
    """数学工具 Agent。

    与 ``MathTools_Agents_Usecase.py`` 保持一致：
    同样的 ``calculator``（基于 numexpr）、同样的 ``language_model``、
    同样的 ``ModelRetryMiddleware(max_retries=3)``。
    """
    import math

    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelRetryMiddleware
    from langchain.chat_models import init_chat_model
    from langchain_core.tools import tool

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        pytest.skip("未配置 DEEPSEEK_API_KEY")
    os.environ["DEEPSEEK_API_KEY"] = api_key

    model = init_chat_model("deepseek:deepseek-chat", temperature=0.0, api_key=api_key)

    @tool
    def calculator(expression: str) -> str:
        """Calculate a mathematical expression and return the numeric result.

        The expression must be a single line mathematical expression that solves
        the problem, using Python arithmetic syntax.

        Examples:
            "100 * 4" for "100 times 4"
            "37593 * 67" for "37593 times 67"
            "17**11" for "17 raised to the power of 11"
        """
        import numexpr

        expr = expression.strip().strip("`")
        if expr.lower().startswith("answer:"):
            expr = expr[len("answer:") :].strip()
        return str(
            numexpr.evaluate(
                expr,
                global_dict={},
                local_dict={"pi": math.pi, "e": math.e},
            )
        )

    @tool
    def language_model(query: str) -> str:
        """Use this tool for general purpose queries and logic."""
        response = model.invoke(query)
        return response.content

    agent = create_agent(
        model=model,
        tools=[calculator, language_model],
        system_prompt=(
            "You are a mathematical assistant that helps solve math problems. "
            "For any question that requires arithmetic, first work out a single-line "
            "Python expression that solves it, then call the `calculator` tool with "
            "that expression to obtain the exact numeric answer. "
            "Always report the final numeric answer."
        ),
        middleware=[ModelRetryMiddleware(max_retries=3)],
    )
    return agent


def _extract_tool_calls(result) -> list:
    """从 Agent 返回结果里抽取工具调用记录。

    Args:
        result: ``agent.invoke()`` 的返回值（状态字典）。

    Returns:
        ``List[ToolCall]``；没有工具调用时返回空列表。
    """
    from deepeval.test_case import ToolCall

    calls: list[ToolCall] = []
    for message in result.get("messages", []):
        for call in getattr(message, "tool_calls", None) or []:
            calls.append(
                ToolCall(
                    name=call["name"],
                    description=None,
                    reasoning=None,
                    output=None,
                    input_parameters=call.get("args", {}),
                )
            )
    return calls


# 需要真正调用模型的用例
MATH_CASES = [
    ("multiply", "What is 100 times 4?", 400),
    ("power", "What is 17 raised to the power of 11?", 17**11),
    ("mixed", "What is (4.5 * 2.1) squared?", (4.5 * 2.1) ** 2),
]


@pytest.mark.llm_judge
@pytest.mark.slow
@pytest.mark.parametrize("name,question,expected", MATH_CASES, ids=[c[0] for c in MATH_CASES])
def test_math_agent_tool_correctness(judge, math_agent, name, question, expected):
    """Agent 应当调用 ``calculator`` 工具，并且算出正确结果。

    ``ToolCorrectnessMetric`` 会评估：

    - 工具名是否正确（该调 calculator 而不是 language_model）；
    - 参数是否正确（表达式填对没）。

    ``expected_tools`` 进一步声明「必须调用了这些工具」，
    工具完全没被调用时该指标直接判失败。
    """
    from deepeval import assert_test
    from deepeval.metrics import ToolCorrectnessMetric
    from deepeval.test_case import LLMTestCase, ToolCall

    from langchain_core.messages import HumanMessage

    result = math_agent.invoke({"messages": [HumanMessage(content=question)]})
    answer = result["messages"][-1].content
    tool_calls = _extract_tool_calls(result)

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        tools_called=tool_calls,
        # expected_tools 必须是 ToolCall 对象列表，不能直接传工具名字符串
        expected_tools=[
            ToolCall(
                name="calculator",
                description=None,
                reasoning=None,
                output=None,
                input_parameters=None,
            )
        ],
        expected_output=str(expected),
    )

    assert_test(
        test_case=test_case,
        metrics=[
            ToolCorrectnessMetric(
                model=judge,
                threshold=threshold(),
                should_consider_ordering=False,
            )
        ],
    )


@pytest.mark.llm_judge
@pytest.mark.slow
def test_math_agent_does_not_call_calculator_for_factual_question(judge, math_agent):
    """知识型问题不该调计算器——工具使用要「恰当」。

    很多 Agent 会养成「什么都调一下」的习惯，既慢又贵，
    还会把简单问题复杂化。

    **指标选择说明**
    ----------------
    这里用 ``TaskCompletionMetric`` 而不是 ``ToolUseMetric``：

    - ``ToolUseMetric`` 继承自 ``BaseConversationalMetric``，只能配
      ``ConversationalTestCase``，配 ``LLMTestCase`` 会报
      「All 'metrics' for an 'LLMTestCase' must be instances of 'BaseMetric' only」；
    - ``ToolCorrectnessMetric`` 则强制要求 ``expected_tools`` 非空，
      而本题「不该调任何工具」，没有合适的 expected_tools 可填。

    ``TaskCompletionMetric`` 正好合适：它评「任务有没有完成」，
    不强制要求声明工具。

    判定分两层：

    1. **确定性断言**：本题根本不该出现 calculator 调用
       （这是硬性规则，不需要裁判）；
    2. **LLM 评判**：任务是否被正确完成。
    """
    from deepeval import assert_test
    from deepeval.metrics import TaskCompletionMetric
    from deepeval.test_case import LLMTestCase

    from langchain_core.messages import HumanMessage

    question = "What is the capital of the United States of America?"
    result = math_agent.invoke({"messages": [HumanMessage(content=question)]})
    answer = result["messages"][-1].content
    tool_calls = _extract_tool_calls(result)

    # 确定性断言：知识型问题不该调计算器
    called_names = {call.name for call in tool_calls}
    assert "calculator" not in called_names, (
        f"知识型问题不应调用 calculator，实际调用：{called_names}"
    )

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        tools_called=tool_calls,
        expected_output="Washington, D.C.",
    )

    assert_test(
        test_case=test_case,
        metrics=[TaskCompletionMetric(model=judge, threshold=threshold())],
    )


@pytest.mark.llm_judge
@pytest.mark.slow
def test_math_agent_task_completion(judge, math_agent):
    """Agent 是否真正完成了任务：给出可验证的最终数值。

    ``TaskCompletionMetric`` 关注的是「有没有把事办完」，
    而不是「过程好不好」。
    """
    from deepeval import assert_test
    from deepeval.metrics import TaskCompletionMetric
    from deepeval.test_case import LLMTestCase

    from langchain_core.messages import HumanMessage

    question = "What is 37593 times 67?"
    result = math_agent.invoke({"messages": [HumanMessage(content=question)]})
    answer = result["messages"][-1].content
    tool_calls = _extract_tool_calls(result)

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        tools_called=tool_calls,
        expected_output=str(37593 * 67),
    )

    assert_test(
        test_case=test_case,
        metrics=[TaskCompletionMetric(model=judge, threshold=threshold())],
    )


@pytest.mark.llm_judge
@pytest.mark.slow
def test_math_agent_exact_numeric_answer(judge, math_agent):
    """数值类问题必须给出与标准答案完全一致的数字。

    用 GEval 做「答案正确性」的严格核对：
    明确告诉裁判标准答案，让它判断回答里的数字是否等于该值。
    这比 TaskCompletion 更严格——后者允许「意思对就行」。
    """
    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase

    from langchain_core.messages import HumanMessage

    question = "What is 37593 times 67?"
    expected = 37593 * 67
    result = math_agent.invoke({"messages": [HumanMessage(content=question)]})
    answer = result["messages"][-1].content

    metric = build_g_eval(
        name="数值答案精确正确",
        evaluation_steps=[
            f"标准答案是 {expected}。",
            "检查回答中给出的最终数值是否等于标准答案。",
            "若回答给出的数值与标准答案不符，判为不符合。",
            "若回答数值正确，判为符合。",
        ],
    )

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=str(expected),
    )
    assert_test(test_case=test_case, metrics=[metric])


# --------------------------------------------------------------------------- #
# SQL Agent（对应 SQLDatabaseTool_SqlAgent.py）
# --------------------------------------------------------------------------- #
@pytest.mark.llm_judge
@pytest.mark.slow
def test_sql_agent_tool_selection(judge):
    """SQL Agent 应当先看表结构、再执行查询，而不是瞎猜 SQL。

    对应 ``SQLDatabaseTool_SqlAgent.py`` 里模型的工作流：
    list_tables → schema → query → 总结。

    这里先做**确定性断言**（必须调用过 sql_db_query），
    再用 GEval 检查回答是否给出了具体数字。
    """
    from langchain.agents import create_agent
    from langchain.chat_models import init_chat_model
    from langchain_community.tools.sql_database.tool import (
        InfoSQLDatabaseTool,
        ListSQLDatabaseTool,
        QuerySQLDatabaseTool,
    )
    from langchain_community.utilities import SQLDatabase
    from langchain_core.messages import HumanMessage
    from sqlalchemy import create_engine

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        pytest.skip("未配置 DEEPSEEK_API_KEY")
    os.environ["DEEPSEEK_API_KEY"] = api_key

    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "chinook.db"

    engine = create_engine(f"sqlite:///{db_path}")
    db = SQLDatabase(engine)
    model = init_chat_model("deepseek:deepseek-chat", temperature=0.0, api_key=api_key)

    tools = [
        ListSQLDatabaseTool(db=db),
        InfoSQLDatabaseTool(db=db),
        QuerySQLDatabaseTool(db=db),
    ]
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=(
            "You are a SQLite expert. Use the provided SQL tools to answer the question. "
            "First list the tables, then inspect the relevant schema, then run a SELECT "
            "query with `sql_db_query`. Never modify data. "
            "Finally, answer in natural language with the concrete numbers."
        ),
    )

    question = "How many employees are there?"
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    answer = result["messages"][-1].content
    tool_calls = _extract_tool_calls(result)

    # 确定性断言：必须调用过查询工具
    tool_names = {call.name for call in tool_calls}
    assert "sql_db_query" in tool_names, f"未调用 sql_db_query，实际调用：{tool_names}"

    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        tools_called=tool_calls,
        expected_output="5",
    )

    metric = build_g_eval(
        name="SQL 回答包含正确的数字",
        evaluation_steps=[
            "数据库中共有 5 名员工。",
            "检查回答是否明确给出 5 这个数字（或明确的等价表述）。",
            "若回答给出了不同的数字或没有给出数字，判为不符合。",
        ],
    )

    assert_test(test_case=test_case, metrics=[metric])
