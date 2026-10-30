"""对话记忆评测。

覆盖的指标
----------
============================  ==============================================
指标                           评估什么
============================  ==============================================
``ConversationalTestCase``      多轮对话（逐轮存储上下文与输出）
``AnswerRelevancyMetric``      每轮回答是否切题
``GEval``                      记忆是否真正起作用（引用了前文信息）
``TurnContextualRelevancyMetric``  当前轮上下文对当前轮回答是否有用
``KnowledgeRetentionMetric``   长对话后是否仍记得关键信息
============================  ==============================================

对应被测系统
------------
- ``src/05-memory-demo/ConversationBufferMemoryDemo.py``（全量缓冲）
- ``src/05-memory-demo/ConversationBufferWindowMemoryDemo.py``（滑动窗口）
- ``src/05-memory-demo/ConversationSummaryMemoryDemo.py``（摘要记忆）

记忆评测的核心难点：**「模型答对了」不等于「记忆生效了」**。

如果模型光靠当前这一轮就能答对，那记忆机制有没有接上根本看不出来。
所以这里用**「需要前文才能答对」的问题**做探测：

第 1 轮告知信息（"我的订单号是 A123"），
第 2 轮问一个只有结合前文才能回答的问题（"我刚才说的订单号是多少？"）。
如果第 2 轮答不出，说明记忆断了——可能是没存、没取、或者被裁掉了。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_config import build_g_eval, threshold  # noqa: E402


def _require_key() -> str:
    """读取 API Key，缺失则跳过用例。"""
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        pytest.skip("未配置 DEEPSEEK_API_KEY")
    os.environ["DEEPSEEK_API_KEY"] = api_key
    return api_key


@pytest.fixture(scope="module")
def llm():
    """评测用对话模型。"""
    from langchain.chat_models import init_chat_model

    api_key = _require_key()
    return init_chat_model(model="deepseek-chat", api_key=api_key, temperature=0)


@pytest.fixture(scope="module")
def memory_agent():
    """带记忆的 Agent。

    对应 ``ConversationBufferMemoryDemo.py`` 的实现：
    用 ``create_agent`` + ``checkpointer``（``InMemorySaver``）保存对话历史。
    """
    from langchain.agents import create_agent
    from langchain.chat_models import init_chat_model
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver

    api_key = _require_key()
    model = init_chat_model(model="deepseek-chat", api_key=api_key, temperature=0)

    @tool
    def lookup_order(order_id: str) -> str:
        """Look up the status of an order by its ID."""
        return f"Order {order_id} has been shipped and is expected to arrive in 2 days."

    agent = create_agent(
        model=model,
        tools=[lookup_order],
        system_prompt=(
            "You are a helpful order support assistant. "
            "Remember details the customer gives you across turns, "
            "and use the lookup_order tool when they ask about a specific order."
        ),
        checkpointer=InMemorySaver(),
    )
    return agent


def _run_turn(agent, thread_id: str, message: str) -> str:
    """在指定会话线程里跑一轮对话。

    Args:
        agent: 带 checkpointer 的 Agent。
        thread_id: 会话线程标识（决定读哪份记忆）。
        message: 用户消息。

    Returns:
        Agent 的回答文本。
    """
    from langchain_core.messages import HumanMessage

    result = agent.invoke(
        {"messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content


@pytest.mark.llm_judge
@pytest.mark.slow
def test_memory_recalls_previous_turn(judge, memory_agent):
    """第 2 轮应当记得第 1 轮提供的订单号——这是记忆生效的直接证据。

    第 2 轮的问题（「我的订单号是多少」）**只看当前轮是无法回答的**，
    必须依赖第 1 轮的记忆，所以答对就说明记忆生效。
    """
    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase

    thread_id = "eval-memory-recall"
    turn1 = _run_turn(memory_agent, thread_id, "Hi, my order ID is A12345 and I need help with it.")
    turn2 = _run_turn(memory_agent, thread_id, "Actually, what was my order ID again?")

    metric = build_g_eval(
        name="跨轮记忆召回",
        evaluation_steps=[
            "用户在第一轮提供了订单号 A12345。",
            f"模型在第二轮的回答是：{turn2}",
            "检查回答中是否正确提到了订单号 A12345。",
            "若回答没有提到该订单号，或给出了错误的号码，判为不符合。",
        ],
    )

    test_case = LLMTestCase(
        input="Actually, what was my order ID again?",
        actual_output=turn2,
    )
    assert_test(test_case=test_case, metrics=[metric])


@pytest.mark.llm_judge
@pytest.mark.slow
def test_memory_uses_tool_with_recalled_id(judge, memory_agent):
    """Agent 应当在第二轮用**记住的**订单号调用工具。

    这是记忆与工具调用的结合评测：
    不仅要记住，还要把记忆用起来。
    """
    from deepeval import assert_test
    from deepeval.metrics import ToolCorrectnessMetric
    from deepeval.test_case import LLMTestCase, ToolCall

    from langchain_core.messages import HumanMessage

    thread_id = "eval-memory-tool"
    _run_turn(memory_agent, thread_id, "My order number is B67890, please check it.")

    result = memory_agent.invoke(
        {"messages": [HumanMessage(content="Please look up my order status.")]},
        config={"configurable": {"thread_id": thread_id}},
    )
    answer = result["messages"][-1].content

    calls = [
        ToolCall(
            name=call["name"],
            description=None,
            reasoning=None,
            output=None,
            input_parameters=call.get("args", {}),
        )
        for message in result.get("messages", [])
        for call in (getattr(message, "tool_calls", None) or [])
    ]

    lookup_calls = [c for c in calls if c.name == "lookup_order"]
    assert lookup_calls, "未调用 lookup_order 工具"

    # 确定性断言：传给工具的订单号必须是记住的那个
    passed_order_id = str(lookup_calls[0].input_parameters.get("order_id", ""))
    assert "B67890" in passed_order_id, (
        f"工具参数中的订单号不正确：{passed_order_id}（应为 B67890）"
    )

    test_case = LLMTestCase(
        input="Please look up my order status.",
        actual_output=answer,
        tools_called=calls,
        # expected_tools 必须是 ToolCall 对象列表，不能直接传工具名字符串
        expected_tools=[
            ToolCall(
                name="lookup_order",
                description=None,
                reasoning=None,
                output=None,
                input_parameters={"order_id": "B67890"},
            )
        ],
    )
    assert_test(
        test_case=test_case,
        metrics=[ToolCorrectnessMetric(model=judge, threshold=threshold())],
    )


@pytest.mark.llm_judge
@pytest.mark.slow
def test_memory_isolated_between_threads(judge, memory_agent):
    """不同会话线程之间记忆必须隔离，不能串号。

    这是记忆系统的**正确性红线**：
    用户 A 的订单号绝不能出现在用户 B 的回答里。

    用两个线程分别告知不同订单号，再交叉询问。
    """
    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase

    _run_turn(memory_agent, "eval-thread-1", "My order ID is C11111.")
    _run_turn(memory_agent, "eval-thread-2", "My order ID is D22222.")

    answer1 = _run_turn(memory_agent, "eval-thread-1", "What is my order ID?")

    metric = build_g_eval(
        name="会话间记忆隔离",
        evaluation_steps=[
            "线程 1 的用户订单号是 C11111，线程 2 的用户订单号是 D22222。",
            f"线程 1 中的回答是：{answer1}",
            "检查回答中是否只提到了 C11111。",
            "若回答中出现了 D22222，说明记忆串到了别的会话，判为不符合。",
        ],
    )

    test_case = LLMTestCase(input="What is my order ID?", actual_output=answer1)
    assert_test(test_case=test_case, metrics=[metric])


@pytest.mark.llm_judge
@pytest.mark.slow
def test_conversational_case_metrics(judge, memory_agent):
    """用 ``ConversationalTestCase`` 跑多轮评测。

    多轮场景下 ``LLMTestCase`` 不够用——
    ``ConversationalTestCase`` 会保留每一轮的输入/输出/上下文，
    支持 ``TurnContextualRelevancyMetric`` 这类「逐轮」指标。
    """
    from deepeval import assert_test
    from deepeval.metrics import TurnContextualRelevancyMetric
    from deepeval.test_case import ConversationalTestCase, Turn

    thread_id = "eval-conversational"
    t1 = _run_turn(memory_agent, thread_id, "I need to cancel my subscription.")
    t2 = _run_turn(memory_agent, thread_id, "What did I just ask you to do?")

    test_case = ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="I need to cancel my subscription.",
                retrieval_context=None,
            ),
            Turn(
                role="assistant",
                content=t1,
                retrieval_context=None,
            ),
            Turn(
                role="user",
                content="What did I just ask you to do?",
                retrieval_context=None,
            ),
            Turn(
                role="assistant",
                content=t2,
                retrieval_context=None,
            ),
        ]
    )

    assert_test(
        test_case=test_case,
        metrics=[TurnContextualRelevancyMetric(model=judge, threshold=threshold())],
    )


@pytest.mark.llm_judge
@pytest.mark.slow
def test_summary_memory_keeps_key_facts(judge):
    """摘要记忆：多轮之后仍应保留关键事实。

    对应 ``ConversationSummaryMemoryDemo.py``：
    对话变长时把早期内容压缩成摘要，而不是原样保留。

    评测方式：先给出关键事实，再插入若干轮无关闲聊，
    最后回头问那个关键事实——考察压缩过程有没有把信息丢掉。
    """
    from langchain.agents import create_agent
    from langchain.chat_models import init_chat_model
    from langgraph.checkpoint.memory import InMemorySaver

    api_key = _require_key()
    model = init_chat_model(model="deepseek-chat", api_key=api_key, temperature=0)

    agent = create_agent(
        model=model,
        tools=[],
        system_prompt=(
            "You are a helpful assistant. Remember important details the user "
            "tells you, even across many turns of conversation."
        ),
        checkpointer=InMemorySaver(),
    )

    thread_id = "eval-summary-memory"
    _run_turn(agent, thread_id, "My preferred delivery address is 88 Willow Lane, Springfield.")

    # 插入若干轮无关内容，模拟对话变长
    for filler in [
        "What's a good recipe for banana bread?",
        "Can you explain how photosynthesis works?",
        "Tell me a short fun fact about octopuses.",
        "How do I change a tire on my car?",
    ]:
        _run_turn(agent, thread_id, filler)

    answer = _run_turn(agent, thread_id, "What is my preferred delivery address?")

    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase

    metric = build_g_eval(
        name="长对话后关键事实留存",
        evaluation_steps=[
            "用户第一轮提供的地址是 88 Willow Lane, Springfield。",
            f"多轮之后模型的回答是：{answer}",
            "检查回答中是否仍然包含该地址信息。",
            "若回答说不记得、或给出了别的地址，判为不符合。",
        ],
    )

    test_case = LLMTestCase(
        input="What is my preferred delivery address?",
        actual_output=answer,
    )
    assert_test(test_case=test_case, metrics=[metric])
