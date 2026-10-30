"""DeepEval 评测套件的公共配置层。

职责
----
1. **裁判模型（judge model）配置**：把 deepeval 的裁判模型统一收敛到一处，
   并支持通过环境变量切换（DeepSeek / OpenAI / 本地代理 / Ollama）。
2. **测试数据与基线**：集中管理评测数据集，避免各测试文件重复定义。
3. **跳过机制**：没有配置裁判模型 API Key 时，自动跳过需要 LLM 评判的用例，
   而不是直接报错中断整个测试套件。

环境变量
--------
``EVAL_JUDGE_PROVIDER``
    裁判模型提供方，可选值：

    - ``deepeval``（默认）：使用 deepeval 内置的 DeepSeek 集成；
    - ``openai``：直连 OpenAI；
    - ``proxy``：走本项目的 OpenAI 兼容代理（``src/utils/deepseek_proxy.py``）；
    - ``ollama``：本地 Ollama，无需任何 API Key。

``EVAL_JUDGE_MODEL``
    裁判模型名，缺省按 provider 取合理默认值。
``DEEPSEEK_API_KEY`` / ``OPENAI_API_KEY``
    对应 provider 的密钥。
``EVAL_THRESHOLD``
    指标默认通过阈值，默认 ``0.5``。
``EVAL_SKIP_LLM``
    设为 ``1`` 时跳过所有需要 LLM 评判的用例，只跑离线可测的用例。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 关闭 deepeval 遥测上报：评测内容可能包含业务数据，不应外传。
# 必须在 import deepeval 之前设置。
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# 项目根目录（evals/ 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 默认通过阈值
DEFAULT_THRESHOLD = 0.5

# 是否强制跳过需要 LLM 评判的用例
_SKIP_LLM = os.getenv("EVAL_SKIP_LLM", "").strip().lower() in {"1", "true", "yes", "on"}


def judge_provider() -> str:
    """返回当前配置的裁判模型提供方。

    Returns:
        ``deepeval`` / ``openai`` / ``proxy`` / ``ollama`` 之一。
    """
    return os.getenv("EVAL_JUDGE_PROVIDER", "deepeval").strip().lower()


def judge_model_name() -> str:
    """返回裁判模型名。

    Returns:
        模型名；未显式配置时按 provider 返回默认值。
    """
    explicit = os.getenv("EVAL_JUDGE_MODEL")
    if explicit:
        return explicit

    provider = judge_provider()
    if provider == "deepeval":
        return os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
    if provider == "openai":
        return os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    if provider == "proxy":
        return os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
    if provider == "ollama":
        return os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:7b")
    return "deepseek-chat"


def threshold() -> float:
    """返回指标默认通过阈值。

    Returns:
        0.0 ~ 1.0 之间的浮点数；配置非法时回退到 ``0.5``。
    """
    raw = os.getenv("EVAL_THRESHOLD")
    if not raw:
        return DEFAULT_THRESHOLD
    try:
        value = float(raw)
    except ValueError:
        logger.warning("EVAL_THRESHOLD=%r 不是合法数字，回退到 %.2f", raw, DEFAULT_THRESHOLD)
        return DEFAULT_THRESHOLD
    if not 0.0 <= value <= 1.0:
        logger.warning("EVAL_THRESHOLD=%s 超出 [0, 1] 范围，回退到 %.2f", value, DEFAULT_THRESHOLD)
        return DEFAULT_THRESHOLD
    return value


def build_judge_model():
    """构造裁判模型实例。

    这是所有需要 LLM 评判指标的共同依赖。不同 provider 的构造方式：

    - ``deepeval``：使用内置 DeepSeek 集成，直连 DeepSeek 官方 API；
    - ``openai``：直连 OpenAI；
    - ``proxy``：以 OpenAI 兼容协议访问本项目的本地代理，
      由代理把 ``json_schema`` 结构化输出改写成 DeepSeek 支持的 tool-calling；
    - ``ollama``：本地 Ollama，完全离线。

    Returns:
        ``DeepEvalBaseLLM`` 实例。

    Raises:
        RuntimeError: 缺少必要的 API Key，或 provider 不受支持。
    """
    provider = judge_provider()
    model_name = judge_model_name()

    if provider == "deepeval":
        from deepeval.models import DeepSeekModel

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "未找到 DEEPSEEK_API_KEY，请在 .env 中设置；"
                "或设置 EVAL_JUDGE_PROVIDER=ollama 使用本地模型"
            )
        logger.info("裁判模型：deepeval DeepSeekModel，model=%s", model_name)
        return DeepSeekModel(model=model_name, api_key=api_key, temperature=0.0)

    if provider == "openai":
        from deepeval.models import OpenAIModel

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("未找到 OPENAI_API_KEY，请在 .env 中设置")
        logger.info("裁判模型：OpenAIModel，model=%s", model_name)
        return OpenAIModel(model=model_name, api_key=api_key, temperature=0.0)

    if provider == "proxy":
        from deepeval.models import OpenAIModel

        base_url = os.getenv("DEEPSEEK_PROXY_URL", "http://127.0.0.1:8765/v1")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("未找到 DEEPSEEK_API_KEY，请在 .env 中设置")
        logger.info("裁判模型：经本地代理 OpenAIModel，base_url=%s, model=%s", base_url, model_name)
        return OpenAIModel(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
        )

    if provider == "ollama":
        from deepeval.models import OllamaModel

        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        logger.info("裁判模型：本地 OllamaModel，base_url=%s, model=%s", base_url, model_name)
        return OllamaModel(model=model_name, base_url=base_url)

    raise RuntimeError(
        f"不支持的 EVAL_JUDGE_PROVIDER={provider!r}，"
        "可选值：deepeval / openai / proxy / ollama"
    )


def llm_judge_available() -> bool:
    """判断当前环境是否具备跑 LLM 评判用例的条件。

    Returns:
        True 表示可以跑；False 表示应当跳过相关用例。
    """
    if _SKIP_LLM:
        logger.info("检测到 EVAL_SKIP_LLM=1，跳过所有需要 LLM 评判的用例")
        return False

    provider = judge_provider()
    if provider == "ollama":
        return True

    if provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))

    return bool(os.getenv("DEEPSEEK_API_KEY"))


def require_llm_judge() -> bool:
    """给 pytest 用的 skip 标记。

    Returns:
        True 表示继续执行；False 表示应跳过该用例。
    """
    return llm_judge_available()


def build_g_eval(
    name: str,
    criteria: str | None = None,
    evaluation_steps: list[str] | None = None,
    params: list | None = None,
    model=None,
):
    """构造一个配置正确的 ``GEval`` 指标。

    **为什么需要这个包装函数**
    --------------------------
    deepeval 4.x 的 ``GEval`` 在与「纯 test case」（而非 trace）配合使用时，
    必须显式告诉它「要评判哪些字段」，否则会抛
    ``MissingTestCaseParamsError``：

        GEval(name="x", evaluation_steps=[...])
        → MissingTestCaseParamsError: has no evaluation_params

    也就是说，只有 ``criteria`` 或 ``evaluation_steps`` 是不够的，
    还得给 ``evaluation_params``。

    本函数统一处理这个细节，并默认评判「输入 + 实际输出」这两个最常用字段。

    Args:
        name: 指标名称。
        criteria: 评判标准（自然语言描述）。
        evaluation_steps: 逐步评判步骤；与 criteria 二选一或同时给出。
        params: 要评判的字段列表，默认 ``[INPUT, ACTUAL_OUTPUT]``。
        model: 裁判模型；None 表示使用 deepeval 默认模型。

    Returns:
        配置好的 ``GEval`` 实例。

    Raises:
        RuntimeError: 未配置裁判模型且未显式传入 model。
    """
    from deepeval.metrics import GEval
    from deepeval.test_case import SingleTurnParams

    if model is None:
        model = build_judge_model()

    if params is None:
        params = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]

    return GEval(
        name=name,
        model=model,
        threshold=threshold(),
        criteria=criteria,
        evaluation_steps=evaluation_steps,
        evaluation_params=params,
    )


# --------------------------------------------------------------------------- #
# 评测数据集
# --------------------------------------------------------------------------- #
# 说明：这里只放「与实现无关」的稳定数据。
# 需要真实模型输出的用例，会在各自测试文件里现场调用被测系统再构造 test case。

# RAG 评测用的知识片段（与 src/01-rag-demo/RAGDemo.py 的示例文档同源）
RAG_CONTEXT = [
    "Reinforcement learning is a type of machine learning where an agent learns "
    "by receiving rewards or penalties for its actions in an environment.",
    "Computer vision is a field of AI that enables computers to interpret and "
    "understand visual information such as images and video.",
    "Natural language processing (NLP) allows computers to understand, interpret, "
    "and generate human language. Applications include chatbots, sentiment "
    "analysis, translation, and text summarization.",
]

# RAG 的「标准答案」，用于 Faithfulness / GEval 等指标做参考答案
RAG_GOLDEN_ANSWERS = {
    "rl": (
        "Reinforcement learning is a machine learning paradigm where an agent "
        "learns a policy by interacting with an environment and receiving "
        "rewards or penalties for its actions."
    ),
    "cv": (
        "Computer vision is the field of AI concerned with enabling computers "
        "to interpret and understand images and video. Common applications "
        "include image recognition, object detection, and medical imaging."
    ),
    "nlp": (
        "NLP is the subfield of AI focused on understanding and generating "
        "human language. Real-world uses include chatbots, sentiment analysis, "
        "machine translation, and summarization."
    ),
}

# 提示工程评测用的客户消息（与 src/03-prompt-engineering-demo/ 各演示同源）
CUSTOMER_MESSAGE = (
    "Hi, I'm really frustrated. My order #12345 hasn't arrived after two weeks "
    "and I want a refund right now. This is unacceptable!"
)

# 提示工程：约束式输出的硬性要求
CONSTRAINED_OUTPUT_RULES = {
    "paragraphs": 3,
    "no_exclamation_marks": True,
    "apology_count": 1,
    "i_understand_count": 1,
    "ends_with_question": True,
}


__all__ = [
    "DEFAULT_THRESHOLD",
    "PROJECT_ROOT",
    "CUSTOMER_MESSAGE",
    "CONSTRAINED_OUTPUT_RULES",
    "RAG_CONTEXT",
    "RAG_GOLDEN_ANSWERS",
    "build_g_eval",
    "build_judge_model",
    "judge_model_name",
    "judge_provider",
    "llm_judge_available",
    "require_llm_judge",
    "threshold",
]
