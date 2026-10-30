"""提示工程评测。

覆盖的指标
----------
============================  ==============================================
指标                           评估什么
============================  ==============================================
确定性断言                       硬性格式约束是否满足（无需 LLM）
``AnswerRelevancyMetric``      回答是否切题
``GEval``                      语气、说服力、角色一致性
``PromptAlignmentMetric``      回答是否遵循提示词的意图
``ToxicityMetric``             是否有攻击性/不当内容（客服场景必备）
``BiasMetric``                 是否存在偏见
============================  ==============================================

对应被测系统
------------
- ``src/03-prompt-engineering-demo/ConstrainedOutputPromptingDemo.py``
- ``src/03-prompt-engineering-demo/FewShotDemo.py``
- ``src/03-prompt-engineering-demo/RoleTaskContextPromptDemo.py``
- ``src/03-prompt-engineering-demo/ChainOfThoughtDemo.py``

提示工程评测的关键设计：**格式约束用确定性断言，不用 LLM 评判**。

「有没有感叹号」「道歉出现几次」这类事实，交给正则判断比交给 LLM
既快又准，还省 token。LLM 评判留给真正主观的维度（语气、说服力）。
"""

from __future__ import annotations

import pytest
from eval_config import CUSTOMER_MESSAGE, build_g_eval, threshold
from sut import answer_customer_reply, check_constrained_output


# --------------------------------------------------------------------------- #
# 硬性格式约束（确定性，可离线跑）
# --------------------------------------------------------------------------- #
@pytest.mark.llm_judge
@pytest.mark.slow
def test_constrained_output_format_rules():
    """约束式输出必须满足全部硬性规则。

    这条断言完全由正则/字符串统计判定，不依赖 LLM，
    因此结果稳定、可重复，适合放进 CI 做回归。
    """
    answer = answer_customer_reply(CUSTOMER_MESSAGE)
    checks = check_constrained_output(answer)

    failed = [name for name, ok in checks.items() if not ok]
    assert not failed, (
        f"约束校验未通过，失败项：{failed}\n--- 实际输出 ---\n{answer}\n--- 校验明细 ---\n{checks}"
    )


@pytest.mark.llm_judge
@pytest.mark.slow
def test_constrained_output_no_exclamation_mark():
    """客服回复里不能有感叹号——这正是提示词里显式禁止的规则。"""
    answer = answer_customer_reply(CUSTOMER_MESSAGE)
    assert "!" not in answer, f"输出中出现了感叹号：\n{answer}"


@pytest.mark.llm_judge
@pytest.mark.slow
def test_constrained_output_three_paragraphs():
    """必须是恰好 3 个段落。"""
    answer = answer_customer_reply(CUSTOMER_MESSAGE)
    paragraphs = [p for p in answer.split("\n\n") if p.strip()]
    assert len(paragraphs) == 3, f"期望 3 段，实际 {len(paragraphs)} 段：\n{answer}"


@pytest.mark.llm_judge
@pytest.mark.slow
def test_constrained_output_ends_with_question():
    """第三段必须以问句结尾（引导用户继续对话）。"""
    answer = answer_customer_reply(CUSTOMER_MESSAGE)
    assert answer.strip().endswith("?"), f"输出未以问号结尾：\n{answer}"


# --------------------------------------------------------------------------- #
# 语义质量（需要 LLM 评判）
# --------------------------------------------------------------------------- #
@pytest.mark.llm_judge
@pytest.mark.slow
def test_customer_reply_relevancy(judge):
    """回复必须针对客户的实际问题（订单延误 + 要求退款）。"""
    from deepeval import assert_test
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase

    answer = answer_customer_reply(CUSTOMER_MESSAGE)

    test_case = LLMTestCase(
        input=CUSTOMER_MESSAGE,
        actual_output=answer,
    )
    assert_test(
        test_case=test_case,
        metrics=[AnswerRelevancyMetric(model=judge, threshold=threshold())],
    )


@pytest.mark.llm_judge
@pytest.mark.slow
def test_customer_reply_empathy_and_tone(judge):
    """回复应当共情、克制、有同理心，同时保持专业。

    这是「语气」类的主观维度，用 GEval 描述标准比硬编码断言更合适。
    """
    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase

    answer = answer_customer_reply(CUSTOMER_MESSAGE)

    metric = build_g_eval(
        name="客服回复的共情与专业度",
        evaluation_steps=[
            "检查回复是否表达了对客户处境的理解与共情。",
            "检查回复是否保持了克制、专业的语气，没有指责客户。",
            "检查回复是否针对订单延误和退款诉求给出了实质回应。",
            "检查回复是否提出了明确的下一步（如提供订单号以便查询）。",
            "若回复冷淡、指责客户或完全回避诉求，判为不符合。",
        ],
    )

    test_case = LLMTestCase(
        input=CUSTOMER_MESSAGE,
        actual_output=answer,
    )
    assert_test(test_case=test_case, metrics=[metric])


@pytest.mark.llm_judge
@pytest.mark.slow
def test_customer_reply_not_toxic(judge):
    """客服回复不应含有攻击性、贬损或不当内容。

    客服场景里这是硬性合规要求，
    即使模型整体表现好，也要单独检查这一项。
    """
    from deepeval import assert_test
    from deepeval.metrics import ToxicityMetric
    from deepeval.test_case import LLMTestCase

    answer = answer_customer_reply(CUSTOMER_MESSAGE)

    test_case = LLMTestCase(
        input=CUSTOMER_MESSAGE,
        actual_output=answer,
    )
    assert_test(
        test_case=test_case,
        metrics=[ToxicityMetric(model=judge, threshold=threshold())],
    )


@pytest.mark.llm_judge
@pytest.mark.slow
def test_customer_reply_no_bias(judge):
    """回复不应包含针对特定人群的偏见表述。"""
    from deepeval import assert_test
    from deepeval.metrics import BiasMetric
    from deepeval.test_case import LLMTestCase

    answer = answer_customer_reply(CUSTOMER_MESSAGE)

    test_case = LLMTestCase(
        input=CUSTOMER_MESSAGE,
        actual_output=answer,
    )
    assert_test(
        test_case=test_case,
        metrics=[BiasMetric(model=judge, threshold=threshold())],
    )


# --------------------------------------------------------------------------- #
# Few-shot 风格一致性
# --------------------------------------------------------------------------- #
@pytest.mark.llm_judge
@pytest.mark.slow
def test_few_shot_style_consistency(judge):
    """Few-shot 示例应当把输出风格「带」出来。

    对应 ``FewShotDemo.py``：给模型看几个示例，
    期望它的输出风格与示例一致。

    评测方式：把「示例风格」和「实际输出」一起给裁判，
    让它判断两者风格是否一致。
    """
    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase
    from sut import answer_customer_reply

    answer = answer_customer_reply(CUSTOMER_MESSAGE)

    metric = build_g_eval(
        name="Few-shot 风格一致性",
        evaluation_steps=[
            "参考的示例风格是：冷静、专业、共情、使用分点结构。",
            "检查实际回答是否也呈现出冷静、专业、共情的风格。",
            "检查实际回答是否没有使用感叹号、没有过度承诺、没有指责。",
            "若实际回答与示例风格明显不符，判为不符合。",
        ],
    )

    test_case = LLMTestCase(
        input=CUSTOMER_MESSAGE,
        actual_output=answer,
    )
    assert_test(test_case=test_case, metrics=[metric])
