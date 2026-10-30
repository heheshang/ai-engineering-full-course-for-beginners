"""链式编排评测。

覆盖的指标
----------
============================  ==============================================
指标                           评估什么
============================  ==============================================
确定性断言                       路由是否选对分支、SQL 是否能执行
``AnswerRelevancyMetric``      最终输出是否切题
``GEval``                      中间结果质量、总结是否到位
============================  ==============================================

对应被测系统
------------
- ``src/06-chains-demo/RouterChainDemo.py``（路由链）
- ``src/06-chains-demo/LLMRouterChainDemo.py``（多提示路由链）
- ``src/06-chains-demo/SequentialChainDemo.py`` / ``SequentialChainDemo2.py``（顺序链）
- ``src/06-chains-demo/SQLChainDemo.py``（SQL 链）
- ``src/06-chains-demo/SimpleSequentialChainDemo.py`（简单顺序链）

链式编排的评测**重点不在最终答案**，而在「中间环节」：

一条链错了，可能是提示词错、也可能是路由选错、也可能是字段传递断了。
所以这里的设计是：**每一段单独评**，而不是只看终态。
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


# --------------------------------------------------------------------------- #
# 路由链（对应 RouterChainDemo.py）
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def router_chain():
    """路由链：把物理问题按难度分发给「初学者」或「专家」两条分支。

    与 ``RouterChainDemo.py`` 的实现保持一致：同样的分支提示词、
    同样的结构化输出 schema（``RouteQuery``）。
    """
    from typing import Any, Literal

    from langchain.chat_models import init_chat_model
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    from pydantic import BaseModel, Field

    api_key = _require_key()
    llm = init_chat_model(model="deepseek-chat", api_key=api_key, temperature=0)

    beginner_template = """You are a physics teacher who is really
focused on beginners and explaining complex topics in simple to understand terms.
You assume no prior knowledge. Here is the question\n{input}"""

    expert_template = """You are a world expert physics professor who explains physics topics
to advanced audience members. You can assume anyone you answer has a
PhD level understanding of Physics. Here is the question\n{input}"""

    prompt_infos = [
        {
            "name": "advanced physics",
            "description": "Answers advanced physics questions",
            "prompt_template": expert_template,
        },
        {
            "name": "beginner physics",
            "description": "Answers basic beginner physics questions",
            "prompt_template": beginner_template,
        },
    ]

    destination_chains: dict[str, Any] = {}
    for info in prompt_infos:
        prompt = ChatPromptTemplate.from_template(info["prompt_template"])
        destination_chains[info["name"]] = prompt | llm | StrOutputParser()

    DestinationName = Literal["advanced physics", "beginner physics"]

    class RouteQuery(BaseModel):
        """路由结果：选哪个分支 + 传给分支的问题。"""

        destination: DestinationName = Field(description="选中的分支名称")
        next_inputs: str = Field(description="传给该分支的问题，可对原问题做改写")

    destinations_str = "\n".join(
        f"{info['name']}: {info['description']}" for info in prompt_infos
    )

    router_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Given a raw text input to a language model, select the prompt best suited for the input.
You will be given the names of the available prompts and a description of what the prompt is best suited for.
You may also revise the original input if you think revising it will lead to a better response.

Available prompts:
{destinations}

Only respond with the destination name and the (possibly revised) input.""",
            ),
            ("human", "{input}"),
        ]
    ).partial(destinations=destinations_str)

    router_chain = router_prompt | llm.with_structured_output(RouteQuery)

    return (
        RunnablePassthrough.assign(route=router_chain)
        | RunnableLambda(lambda x: (x["route"].destination, x["route"].next_inputs))
        | RunnableLambda(
            lambda x: destination_chains[x[0]].invoke({"input": x[1]})
        )
    ), router_chain


@pytest.mark.llm_judge
@pytest.mark.slow
def test_router_selects_beginner_for_simple_question(router_chain):
    """简单问题应当路由到「beginner physics」分支。

    这是**确定性断言**：路由结果必须是指定分支之一，
    简单的磁铁问题明显属于初学者分支。
    """
    chain, router = router_chain
    question = "How do magnets work explain?"

    route = router.invoke({"input": question})
    assert route.destination == "beginner physics", (
        f"路由错误：期望 'beginner physics'，实际 '{route.destination}'"
    )
    # 路由后仍要能正常产出答案
    answer = chain.invoke({"input": question})
    assert answer.strip(), "路由后未产出答案"


@pytest.mark.llm_judge
@pytest.mark.slow
def test_router_selects_expert_for_advanced_question(router_chain):
    """专业问题应当路由到「advanced physics」分支。"""
    chain, router = router_chain
    question = "How do Feynman Diagrams work?"

    route = router.invoke({"input": question})
    assert route.destination == "advanced physics", (
        f"路由错误：期望 'advanced physics'，实际 '{route.destination}'"
    )
    answer = chain.invoke({"input": question})
    assert answer.strip(), "路由后未产出答案"


@pytest.mark.llm_judge
@pytest.mark.slow
def test_router_output_relevancy(judge, router_chain):
    """路由后的最终回答必须切题。"""
    from deepeval import assert_test
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase

    chain, _ = router_chain
    question = "How do magnets work explain?"
    answer = chain.invoke({"input": question})

    test_case = LLMTestCase(input=question, actual_output=answer)
    assert_test(
        test_case=test_case,
        metrics=[AnswerRelevancyMetric(model=judge, threshold=threshold())],
    )


@pytest.mark.llm_judge
@pytest.mark.slow
def test_router_rewrites_question(judge, router_chain):
    """路由模型允许改写问题，改写后仍应保留原问题核心。

    这是多提示路由链的关键能力：把模糊问题补全后再交给分支。
    """
    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase

    _, router = router_chain
    question = "How do magnets work explain?"

    route = router.invoke({"input": question})

    metric = build_g_eval(
        name="路由改写保留原问题意图",
        evaluation_steps=[
            f"原始问题是：{question}",
            f"路由模型改写后的问题是：{route.next_inputs}",
            "判断改写后的问题是否仍然在问同一件事（关于磁铁如何工作）。",
            "若改写后偏离了原问题意图，判为不符合。",
        ],
    )

    test_case = LLMTestCase(input=question, actual_output=route.next_inputs)
    assert_test(test_case=test_case, metrics=[metric])


# --------------------------------------------------------------------------- #
# 顺序链（对应 SequentialChainDemo.py / SequentialChainDemo2.py）
# --------------------------------------------------------------------------- #
@pytest.mark.llm_judge
@pytest.mark.slow
def test_sequential_chain_preserves_topic(judge):
    """顺序链：第二段生成的问题必须围绕第一段生成的主题。

    对应 ``SequentialChainDemo.py``：先总结绩效评语，再识别弱点，
    再制定改进计划——每一段都依赖上一段的输出。

    这里验证最核心的一点：**上下文传递没有断**。
    """
    from langchain.chat_models import init_chat_model
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    api_key = _require_key()
    llm = init_chat_model(model="deepseek-chat", api_key=api_key, temperature=0)

    topic_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant who generates interesting topics."),
            ("user", "Generate a random academic topic for discussion."),
        ]
    )
    question_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant who generates thought-provoking topics.",
            ),
            (
                "user",
                "Generate 3 thought-provoking questions about this topic: {topic}.",
            ),
        ]
    )

    topic_chain = topic_prompt | llm | StrOutputParser()
    question_chain = question_prompt | llm | StrOutputParser()

    # 与 SequentialChainDemo2.py 一致的并行分支写法。
    # 注意：这里必须用 RunnableParallel 显式组合。
    # 如果直接写 {"topic": topic_chain} | {"questions": question_chain}，
    # Python 会把两个 dict 当成普通 dict 做 | 合并（后者覆盖前者的键），
    # 而不是把它们组合成 RunnableParallel——那样就完全不是「并行分支」了。
    from langchain_core.runnables import RunnableParallel

    chain = RunnableParallel(
        topic=topic_chain,
    ) | RunnableParallel(
        topic=lambda x: x["topic"],
        questions=question_chain,
    )

    results = chain.invoke({})
    topic = results["topic"]
    questions = results["questions"]

    assert topic.strip(), "未生成主题"
    assert questions.strip(), "未生成问题"

    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase

    metric = build_g_eval(
        name="问题与主题一致",
        evaluation_steps=[
            f"主题是：{topic}",
            f"生成的问题是：{questions}",
            "检查这 3 个问题是否确实围绕该主题展开。",
            "若问题与主题无关，判为不符合。",
        ],
    )

    test_case = LLMTestCase(input="Generate 3 questions about a random topic", actual_output=questions)
    assert_test(test_case=test_case, metrics=[metric])


@pytest.mark.llm_judge
@pytest.mark.slow
def test_sequential_chain_two_stage(judge):
    """两段顺序链：第一段的输出必须真正影响第二段。

    对应 ``SequentialChainDemo.py`` 的「总结 → 识别弱点」：
    第二段的输出里应当能看到第一段总结过的内容。
    """
    from langchain.chat_models import init_chat_model
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    api_key = _require_key()
    llm = init_chat_model(model="deepseek-chat", api_key=api_key, temperature=0)

    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize this employee performance review in 2-3 sentences:\n{review}"
    )
    weakness_prompt = ChatPromptTemplate.from_template(
        "Identify key employee weaknesses based on this summary:\n{summary}"
    )

    review = (
        "Employee consistently missed code review deadlines, which caused two "
        "release delays. Customer feedback about responsiveness was also mixed. "
        "On the other hand, the employee was praised for thorough documentation "
        "and for mentoring two junior developers."
    )

    chain = (summary_prompt | llm | StrOutputParser()) | (
        weakness_prompt | llm | StrOutputParser()
    )
    weaknesses = chain.invoke({"review": review})

    assert weaknesses.strip(), "未生成弱点列表"

    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase

    metric = build_g_eval(
        name="弱点识别准确",
        evaluation_steps=[
            f"绩效评语原文：{review}",
            f"识别出的弱点：{weaknesses}",
            "检查识别出的弱点是否与原文反映的问题一致（如代码评审延期、响应速度）。",
            "检查是否把原文中的优点（如文档详尽、指导新人）误判为弱点。",
            "若弱点与原文不符，判为不符合。",
        ],
    )

    test_case = LLMTestCase(input="Identify key weaknesses from this review", actual_output=weaknesses)
    assert_test(test_case=test_case, metrics=[metric])


# --------------------------------------------------------------------------- #
# SQL 链（对应 SQLChainDemo.py）
# --------------------------------------------------------------------------- #
@pytest.mark.llm_judge
@pytest.mark.slow
def test_sql_chain_generates_executable_sql(judge):
    """SQL 链生成的 SQL 必须可执行，且结果正确。

    对应 ``SQLChainDemo.py`` 的路线 A（LCEL 固定链）：
    生成 SQL → 执行 → 总结。

    这里做**双重校验**：
    1. 确定性：SQL 真的能跑，且结果行数符合预期；
    2. LLM：生成的 SQL 语法与逻辑是否正确。
    """
    from langchain.chat_models import init_chat_model
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_community.utilities import SQLDatabase
    from sqlalchemy import create_engine

    api_key = _require_key()
    llm = init_chat_model(model="deepseek-chat", api_key=api_key, temperature=0)

    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "chinook.db"
    engine = create_engine(f"sqlite:///{db_path}")
    db = SQLDatabase(engine)

    sql_prompt = ChatPromptTemplate.from_template(
        """Given the following SQL database schema:
{schema}

Question: {question}

Write a single SQLite query that answers the question.
Return ONLY the SQL, with no explanation and no markdown fences."""
    )

    answer_prompt = ChatPromptTemplate.from_template(
        """Given the question and the SQL query result, answer the question
in natural language with the concrete numbers.

Question: {question}
SQL result: {context}
Answer:"""
    )

    sql_chain = RunnablePassthrough.assign(
        schema=lambda _: db.get_table_info()
    ) | sql_prompt | llm | StrOutputParser()

    full_chain = (
        RunnablePassthrough.assign(sql=sql_chain)
        | RunnablePassthrough.assign(context=lambda x: db.run_no_throw(x["sql"]))
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    question = "How many employees are there?"
    generated_sql = sql_chain.invoke({"question": question})
    logger_output = f"生成的 SQL：{generated_sql}"

    # 确定性校验：SQL 必须能执行，且结果包含 5
    sql_result = db.run_no_throw(generated_sql)
    assert "error" not in sql_result.lower(), (
        f"生成的 SQL 无法执行：{logger_output}\n错误：{sql_result}"
    )
    assert "5" in sql_result, f"SQL 结果不符合预期（应为 5）：{sql_result}\n{logger_output}"

    answer = full_chain.invoke({"question": question})

    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase

    metric = build_g_eval(
        name="SQL 链最终回答正确",
        evaluation_steps=[
            "数据库中共有 5 名员工。",
            f"最终回答是：{answer}",
            "检查回答是否给出了 5 这个数字。",
            "若回答给出其它数字或没有给出数字，判为不符合。",
        ],
    )

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=[sql_result],
        expected_output="5",
    )
    assert_test(test_case=test_case, metrics=[metric])
