"""链（Chain）演示 4：路由链（Router Chain）。

核心概念
--------
路由链解决的是「**一个问题该交给哪条链处理**」的问题。

最简单的路由链由三部分组成：

  1. **router（路由器）**：一个 LLM 调用，只负责「分类」。
     它不回答问题，只输出一个类别名（如 Science / History / Math）。
  2. **destination chains（目标链）**：每类问题对应一条独立的处理链，
     可以有不同的提示词、不同的模型、不同的输出格式。
  3. **路由函数**：拿到类别名后，用 if/else 把控制权交给对应的链。

数据流：

    输入问题 ──> router_chain ──> "math" ──> route_chain() ──> math_chain ──> 答案

适用场景
--------
  - **多领域客服**：技术问题走技术链、账单问题走账单链；
  - **多模型调度**：简单问题走便宜的小模型，复杂问题走昂贵的大模型；
  - **意图识别**：识别用户意图后调用不同的工具/流程。

与 Agent 的关系
--------------
路由链是「**硬编码的路由**」——类别和目标链的对应关系写死在代码里。
而 Agent 是「**模型自主决策**」——模型自己决定调用哪个工具。
路由链更可控、更省 token、延迟更低；Agent 更灵活但不确定性更高。
实际项目里两者常常混用：先用路由快速分流，再对复杂分支启用 Agent。

关于 LangChain 1.x 的 API 变化（重要）
--------------------------------------
    ``langchain.chains`` 模块**已完全移除**（``ModuleNotFoundError``），
    ``LLMChain`` / ``RouterChain`` 等全部不存在；
    官方推荐统一使用 **LCEL（LangChain Expression Language）**：
    用 ``|`` 把 Runnable 组合起来，取代过去各类 Chain 类。

    本文件用 ``PromptTemplate | llm | StrOutputParser()`` 这条管道
    同时扮演「路由器」和「目标链」的角色，语义完全一致。

依赖：pip install langchain langchain-deepseek python-dotenv
"""

import os
import sys
from pathlib import Path

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from utils.logging_utils import get_logger

# 初始化 logger：日志同时输出到控制台和 logs/course.log
logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# 0. 加载环境变量
# --------------------------------------------------------------------------- #
# load_dotenv() 自动读取当前目录（或逐级向上）目录中的 .env 文件，
# 把键值对写入 os.environ，无需手动 export。
load_dotenv()

# 优先读 DEEPSEEK_API_KEY，其次兼容通用命名 API_KEY；
# 都没有就立刻报错，避免等到请求接口时才失败、难以排查。
api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY")
if not api_key:
    raise RuntimeError("未找到 API Key，请在 .env 中设置 DEEPSEEK_API_KEY（或 API_KEY）")
# 再显式写回环境变量，供 init_chat_model 按 provider 自动读取。
os.environ["DEEPSEEK_API_KEY"] = api_key
logger.info("已加载 API Key（来源：DEEPSEEK_API_KEY 或 API_KEY）")

# --------------------------------------------------------------------------- #
# 1. 初始化 LLM
# --------------------------------------------------------------------------- #
# init_chat_model 是统一的模型工厂函数，签名形如 init_chat_model("provider:model", ...)
#   - provider 决定用哪家的客户端（deepseek / openai / anthropic ...）
#   - model    是具体模型名
# temperature=0 让输出尽量确定，便于复现路由结果（分类任务尤其需要确定性）。
llm = init_chat_model(
    model="deepseek:deepseek-chat",
    api_key=api_key,
    temperature=0,
)
logger.info("LLM 初始化完成：provider=deepseek, model=deepseek-chat, temperature=0")

# --------------------------------------------------------------------------- #
# 2. 为不同类型的问题定义不同的提示词（= 不同的目标链）
# --------------------------------------------------------------------------- #
# 每条目标链的提示词会注入不同的「人设」：
#   - 科学问题 -> 科学家视角
#   - 历史问题 -> 历史学家视角
#   - 数学问题 -> 数学家视角
# 这样同一个模型在不同分支上会给出更专业、更贴题的答案。
science_prompt = PromptTemplate.from_template(
    "You are a scientific expert. Answer the following question: {input}"
)

history_prompt = PromptTemplate.from_template(
    "You are a historical expert. Answer the following question: {input}"
)

math_prompt = PromptTemplate.from_template(
    "You are a mathematics expert. Answer the following question: {input}"
)

# --------------------------------------------------------------------------- #
# 3. 组装目标链
# --------------------------------------------------------------------------- #
# ``|`` 是 Runnable 的重载运算符，返回 RunnableSequence，运行时依次执行：
#   1) PromptTemplate.invoke({...}) -> PromptValue（渲染好的提示词）
#   2) llm.invoke(PromptValue)           -> AIMessage（模型回答）
#   3) StrOutputParser.invoke(AIMessage) -> str（纯文本，去掉对象包装）
#
# 最后一步 StrOutputParser 很关键：没有它，链的输出是 AIMessage 对象，
# 后面做字符串判断（如 "science" in category）会失败。
science_chain = science_prompt | llm | StrOutputParser()
history_chain = history_prompt | llm | StrOutputParser()
math_chain = math_prompt | llm | StrOutputParser()
logger.info("已构建 3 条目标链：science / history / math")

# --------------------------------------------------------------------------- #
# 4. 定义路由器
# --------------------------------------------------------------------------- #
# 路由器提示词的关键设计：
#   1. **限定输出集合**：明确写出 "Science, History, or Math"，
#      避免模型自造类别名（如 "Physics"）导致后面 if/else 匹配失败；
#   2. **强调只回类别名**："Only respond with the category name."，
#      防止模型输出一堆解释文字；
#   3. **以 "Category:" 结尾**：利用预训练模型「续写」的倾向，
#      让它自然地补上类别名，效果通常比纯指令更稳。
router_prompt = PromptTemplate.from_template(
    """Given a question, determine which category it belongs to: Science, History, or Math.
    Only respond with the category name.

    Question: {input}
    Category:"""
)

# 路由器链：结构与目标链一致，只是提示词不同
router_chain = router_prompt | llm | StrOutputParser()
logger.info("路由器链构建完成")


# --------------------------------------------------------------------------- #
# 5. 路由逻辑
# --------------------------------------------------------------------------- #
def route_chain(input_text: str):
    """判断问题类型，并返回对应的目标链。

    这是路由链的「决策中枢」：调用路由器拿到类别名，
    再用 if/else 把类别映射到具体的 Runnable。

    注意本实现用「子串包含」而非「相等」来判断，
    这是为了容忍模型输出 "Category: Science" 或 "This is a science question." 这类
    带额外前后缀的回答（更鲁棒）。代价是 "history" 这种词若出现在别的类别描述里
    可能误判，类别名相近时建议改用精确匹配。

    Args:
        input_text: 用户的原始问题。

    Returns:
        应该用来回答该问题的 Runnable（目标链）。
    """
    # 调用路由器，只做分类不做回答
    raw_category = router_chain.invoke({"input": input_text})

    # 归一化：去掉首尾空白、转小写，便于做子串匹配
    category = raw_category.strip().lower()
    logger.info("路由器判定类别：原始输出=%r，归一化后=%r", raw_category, category)

    # 按类别分发到对应链
    if "science" in category:
        logger.info("路由结果 -> science_chain（科学专家）")
        return science_chain
    if "history" in category:
        logger.info("路由结果 -> history_chain（历史专家）")
        return history_chain
    # 兜底分支：模型输出无法识别时一律走数学链。
    # 生产环境更稳妥的做法是再加一个 "general"（通用回答）分支，
    # 避免把非数学问题硬塞给数学专家。
    logger.info("路由结果 -> math_chain（数学专家，兜底分支）")
    return math_chain


# --------------------------------------------------------------------------- #
# 6. 运行演示
# --------------------------------------------------------------------------- #
# 三个测试问题分别覆盖三条分支，方便观察路由是否生效。
questions = [
    "What causes photosynthesis?",  # 期望 -> Science
    "Who was the first president of the United States?",  # 期望 -> History
    "What is the square root of 100?",  # 期望 -> Math
]

logger.info("开始路由链演示，共 %d 个问题", len(questions))
for idx, question in enumerate(questions, start=1):
    logger.info("[%d/%d] 处理问题：%s", idx, len(questions), question)

    # 第一步：路由，选出应该使用的链
    chain = route_chain(question)

    # 第二步：用选中的链真正回答问题
    answer = chain.invoke({"input": question})
    logger.info("[%d/%d] 回答完成，长度=%d 字符", idx, len(questions), len(answer))

    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")

logger.info("路由链演示结束")
