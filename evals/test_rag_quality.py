"""RAG 评测：检索质量 + 生成质量。

覆盖的指标
----------
============================  ==============================================
指标                           评估什么
============================  ==============================================
``AnswerRelevancyMetric``      回答是否切题（与问题的相关度）
``FaithfulnessMetric``         回答是否有幻觉（是否完全基于检索到的上下文）
``ContextualRelevancyMetric``  检索到的上下文对回答是否有用
``ContextualPrecisionMetric``  上下文中有用的片段是否排在前面
``ContextualRecallMetric``     回答用到的信息是否都被上下文覆盖
``GEval``                      自定义标准：结构清晰度 / 引用质量 / 简洁性
============================  ==============================================

RAG 评测的**核心逻辑**：把「检索」和「生成」拆开分别评。

如果只评最终回答，出了问题无法定位：
分数低可能是「没检索到」也可能是「检索到了但没用好」。
拆开后：

- Faithfulness 低 → 生成阶段在编造（上下文里有但回答里没有，或凭空多出）；
- Contextual Recall 低 → 检索阶段漏了（上下文里没有但回答里编了）；
- Contextual Precision 低 → 检索排序差（有用的片段被埋在后面）。
"""

from __future__ import annotations

import pytest

from eval_config import RAG_GOLDEN_ANSWERS, build_g_eval

# 三组评测问题，分别对应三个知识点
QUESTIONS = [
    ("rl", "What is reinforcement learning and how does it work?"),
    ("cv", "What are the main applications of computer vision?"),
    ("nlp", "How is NLP used in real-world applications?"),
]

QUESTION_IDS = [key for key, _ in QUESTIONS]


def _build_rag_case(judge, retriever, rag_agent, key: str, question: str):
    """构造一个 RAG 测试用例。

    统一在这里做「检索 + 生成 + 组装」，避免各测试用例重复代码。

    Args:
        judge: 裁判模型。
        retriever: 向量库检索器。
        rag_agent: RAG Agent。
        key: 知识点标识。
        question: 问题。

    Returns:
        ``LLMTestCase`` 实例。
    """
    from deepeval.test_case import LLMTestCase

    from sut import answer_with_rag, retrieve_context

    context = retrieve_context(retriever, question)
    answer = answer_with_rag(rag_agent, question)

    return LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=context,
        expected_output=RAG_GOLDEN_ANSWERS[key],
    )


@pytest.mark.llm_judge
@pytest.mark.parametrize("key,question", QUESTIONS, ids=QUESTION_IDS)
def test_rag_answer_relevancy(judge, eval_threshold, retriever, rag_agent, key, question):
    """回答应当切题：模型不该跑题去答别的问题。"""
    from deepeval import assert_test
    from deepeval.metrics import AnswerRelevancyMetric

    test_case = _build_rag_case(judge, retriever, rag_agent, key, question)
    assert_test(
        test_case=test_case,
        metrics=[AnswerRelevancyMetric(model=judge, threshold=eval_threshold)],
    )


@pytest.mark.llm_judge
@pytest.mark.parametrize("key,question", QUESTIONS, ids=QUESTION_IDS)
def test_rag_faithfulness(judge, eval_threshold, retriever, rag_agent, key, question):
    """回答应当忠于检索到的上下文：这是防幻觉的核心指标。

    Faithfulness 会把回答拆成若干「事实断言」，逐条检查是否能从
    retrieval_context 中找到依据。任何一条找不到依据就会扣分。
    """
    from deepeval import assert_test
    from deepeval.metrics import FaithfulnessMetric

    test_case = _build_rag_case(judge, retriever, rag_agent, key, question)
    assert_test(
        test_case=test_case,
        metrics=[FaithfulnessMetric(model=judge, threshold=eval_threshold)],
    )


@pytest.mark.llm_judge
@pytest.mark.parametrize("key,question", QUESTIONS, ids=QUESTION_IDS)
def test_rag_contextual_relevancy(judge, eval_threshold, retriever, rag_agent, key, question):
    """检索到的上下文应当对回答有帮助。

    与 Answer Relevancy 的区别：

    - Answer Relevancy 看「回答 vs 问题」；
    - Contextual Relevancy 看「上下文 vs 回答」——
      如果检索到的内容跟要答的问题无关，分数就会低。
    """
    from deepeval import assert_test
    from deepeval.metrics import ContextualRelevancyMetric

    test_case = _build_rag_case(judge, retriever, rag_agent, key, question)
    assert_test(
        test_case=test_case,
        metrics=[ContextualRelevancyMetric(model=judge, threshold=eval_threshold)],
    )


@pytest.mark.llm_judge
@pytest.mark.parametrize("key,question", QUESTIONS, ids=QUESTION_IDS)
def test_rag_contextual_precision(judge, eval_threshold, retriever, rag_agent, key, question):
    """有用的上下文片段应当排在前面（检索排序质量）。

    Contextual Precision 会问裁判：
    「上下文里那些真正支撑回答的句子，排在前面的有多少比例？」

    如果检索把最相关的 chunk 排在第 3 位、噪声排在第 1 位，分数就低。
    调优手段：加大 k、加 rerank、换嵌入模型。
    """
    from deepeval import assert_test
    from deepeval.metrics import ContextualPrecisionMetric

    test_case = _build_rag_case(judge, retriever, rag_agent, key, question)
    assert_test(
        test_case=test_case,
        metrics=[ContextualPrecisionMetric(model=judge, threshold=eval_threshold)],
    )


@pytest.mark.llm_judge
@pytest.mark.parametrize("key,question", QUESTIONS, ids=QUESTION_IDS)
def test_rag_contextual_recall(judge, eval_threshold, retriever, rag_agent, key, question):
    """回答用到的信息都应当能从上下文里找到（检索召回完整性）。

    与 Faithfulness 的关系：

    - Faithfulness 关注「上下文里有但回答没说」→ 惩罚遗漏；
    - Contextual Recall 关注「回答说了但上下文里没有」→ 惩罚编造。

    两者一起用，才能区分「答得不全」和「胡编乱造」。
    """
    from deepeval import assert_test
    from deepeval.metrics import ContextualRecallMetric

    test_case = _build_rag_case(judge, retriever, rag_agent, key, question)
    assert_test(
        test_case=test_case,
        metrics=[ContextualRecallMetric(model=judge, threshold=eval_threshold)],
    )


@pytest.mark.llm_judge
def test_rag_answer_quality_g_eval(judge, eval_threshold, retriever, rag_agent):
    """自定义标准：结构清晰、引用来源、简洁。

    用 GEval 描述「什么样的回答算好」，比手写断言灵活：
    裁判会按 evaluation_steps 逐条推理后给出分数与理由。

    这里评的是「回答质量」这个综合维度，和上面的检索类指标互补。
    """
    from deepeval import assert_test

    key, question = QUESTIONS[0]
    test_case = _build_rag_case(judge, retriever, rag_agent, key, question)

    metric = build_g_eval(
        name="RAG 回答质量",
        evaluation_steps=[
            "检查回答是否明确、直接地回答了问题，没有跑题。",
            "检查回答是否引用了检索到的上下文中的具体内容，而不是空泛泛谈。",
            "检查回答的结构是否清晰（可以是短段落或要点列表）。",
            "检查回答是否简洁，没有大段重复或无关的铺垫。",
            "若回答中出现上下文里没有的信息，该项判为不符合。",
        ],
    )
    assert_test(test_case=test_case, metrics=[metric])


@pytest.mark.llm_judge
def test_rag_unanswerable_question(judge, eval_threshold, retriever, rag_agent):
    """知识库之外的问题，模型应当明确说「不知道」而不是编造。

    这是 RAG 系统最重要的「安全」属性之一。
    很多 RAG 失败的场景不是答错，而是**自信地答错**。

    对应 ``RAGDemo.py`` 的 system_prompt：
    「If the context does not contain the answer, say so explicitly.」
    """
    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase

    from sut import answer_with_rag, retrieve_context

    question = "What is the capital of Australia and how many people live there?"
    context = retrieve_context(retriever, question)
    answer = answer_with_rag(rag_agent, question)

    metric = build_g_eval(
        name="超出知识库的问题应诚实拒答",
        evaluation_steps=[
            "判断回答是否明确表示无法从提供的上下文中得到答案。",
            "若回答给出了具体的首都名称或人口数字，判为不符合（因为上下文里没有）。",
            "若回答明确说不知道或无法从上下文推断，判为符合。",
        ],
    )

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=context,
    )
    assert_test(test_case=test_case, metrics=[metric])
