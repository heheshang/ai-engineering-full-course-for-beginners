"""链（Chain）演示 1：单链（Single Chain）。

旧版写法（LangChain < 0.3）
---------------------------
    from langchain.chains import LLMChain

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(product="plumbing fixtures")

或用 prompt 模板 + 管道：

    chain = chat_prompt_template | llm
    response = chain.invoke({"product": "plumbing fixtures"})

LangChain 1.x 中的现状
----------------------
  - ``langchain.chains`` 模块**已完全移除**（``ModuleNotFoundError``），
    ``LLMChain``、``SimpleChain``、``ConversationChain`` 等全部不存在；
  - 官方推荐统一使用 **LCEL（LangChain Expression Language）**：
    用 ``|`` 把 Runnable 组合起来，取代过去各类 Chain 类；
  - 模型初始化推荐用统一的 ``init_chat_model`` 工厂函数，
    不用再为每个厂商 import 不同的包。

本文件演示两种现代写法：

  写法 A（推荐）：``init_chat_model`` + ``ChatPromptTemplate`` + LCEL 管道
  写法 B：``create_agent``（需要多轮 / 工具调用时才用）

依赖：pip install langchain langchain-deepseek python-dotenv
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.logging_utils import get_logger

# 初始化 logger：日志同时输出到控制台和 logs/course.log
logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# 0. 加载环境变量
# --------------------------------------------------------------------------- #
load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY")
if not api_key:
    raise RuntimeError("未找到 API Key，请在 .env 中设置 DEEPSEEK_API_KEY（或 API_KEY）")
os.environ["DEEPSEEK_API_KEY"] = api_key

# --------------------------------------------------------------------------- #
# 1. 初始化 LLM
# --------------------------------------------------------------------------- #
# 统一工厂函数 init_chat_model("provider:model", ...)：
#   - provider 取值如 deepseek / openai / anthropic / google_vertexai ...
#   - 它会按需从对应集成包（如 langchain-deepseek）加载实现
#   - 好处是换模型只改字符串，业务代码不用改 import
llm = init_chat_model(
    "deepseek:deepseek-chat",
    api_key=api_key,
    temperature=0,
)
logger.info("LLM 初始化完成：provider=deepseek, model=deepseek-chat, temperature=0")

# 也可以继续用具体集成类（完全等价）：
#
# from langchain_deepseek import ChatDeepSeek
#
# llm = ChatDeepSeek(model="deepseek-chat", api_key=api_key, temperature=0)

# --------------------------------------------------------------------------- #
# 2. 定义提示词模板
# --------------------------------------------------------------------------- #
# ChatPromptTemplate.from_messages 支持「元组简写」：
#   ("human", "模板")   等价于 HumanMessagePromptTemplate.from_template("模板")
#   ("system", "模板")  等价于 SystemMessagePromptTemplate.from_template("模板")
#   ("ai", "模板")      等价于 AIMessagePromptTemplate.from_template("模板")
# 用简写更简洁，也少一次 import。
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that invents catchy company names.",
        ),
        (
            "human",
            "Make up a funny company name for a company that produces {product}",
        ),
    ]
)

# 模板里的占位符可以从这里查看
print("模板输入变量：", chat_prompt_template.input_variables)
logger.info("提示词模板构建完成，输入变量：%s", chat_prompt_template.input_variables)

# --------------------------------------------------------------------------- #
# 3. 组装链（LCEL）并调用
# --------------------------------------------------------------------------- #
# ``|`` 是 Runnable 的重载运算符，返回 RunnableSequence：
#   1) chat_prompt_template.invoke({...}) -> PromptValue
#   2) llm.invoke(PromptValue)            -> AIMessage
# 这就是 LangChain 1.x 里取代各种 Chain 类的标准做法。
chain = chat_prompt_template | llm

# invoke 传字典，键名对应模板里的占位符
logger.info("开始调用单链，输入：product=plumbing fixtures")
response = chain.invoke({"product": "plumbing fixtures"})

# response 是 AIMessage，正文在 .content 里
logger.info("单链调用完成，输出长度=%d 字符", len(response.content))
print("公司名：", response.content)

# --------------------------------------------------------------------------- #
# 4. 其他常用调用方式
# --------------------------------------------------------------------------- #
# 4.1 批量调用：传一组输入
logger.info("开始批量调用（2 条输入）")
batch_inputs = [{"product": "plumbing fixtures"}, {"product": "coffee cups"}]
responses = chain.batch(batch_inputs)
logger.info("批量调用完成，共返回 %d 条结果", len(responses))
for item in responses:
    print("批量 →", item.content)

# 4.2 流式输出
print("\n流式输出：")
logger.info("开始流式调用（product=rain boots）")
stream_parts = []
for chunk in chain.stream({"product": "rain boots"}):
    # AIMessageChunk，累加 .content 就是完整回答
    stream_parts.append(chunk.content)
    print(chunk.content, end="", flush=True)
print()
logger.info(
    "流式调用完成，共 %d 个片段，拼接后 %d 字符", len(stream_parts), len("".join(stream_parts))
)

# 4.3 想要结构化输出（返回 Pydantic 对象而不是字符串）
#
# from pydantic import BaseModel, Field
#
#
# class CompanyName(BaseModel):
#     """抽取出公司名及其解释。"""
#
#     name: str = Field(description="生成的公司名")
#     rationale: str = Field(description="为什么这个名字有趣")
#
#
# structured_llm = llm.with_structured_output(CompanyName)
# structured_chain = chat_prompt_template | structured_llm
# result = structured_chain.invoke({"product": "plumbing fixtures"})
# print(result.name, result.rationale)

# --------------------------------------------------------------------------- #
# 5. 什么时候才需要 create_agent？
# --------------------------------------------------------------------------- #
# 单链（一次输入 -> 一次输出）用 LCEL 管道就够了，不需要 Agent。
# 只有出现下面这些需求时才升级到 ``create_agent``：
#   - 需要多轮对话记忆（配合 checkpointer）；
#   - 需要模型自主决定调用哪些工具（tool calling 循环）；
#   - 需要人工审批（human-in-the-loop）、中断恢复等。
#
# from langchain.agents import create_agent
# from langgraph.checkpoint.memory import InMemorySaver
#
# agent = create_agent(
#     model=llm,
#     tools=[],
#     system_prompt="You are a helpful assistant that invents catchy company names.",
#     checkpointer=InMemorySaver(),
# )
#
# result = agent.invoke(
#     {"messages": [("user", "Make up a funny company name for a company that produces plumbing fixtures")]},
#     {"configurable": {"thread_id": "chain-1"}},
# )
# print(result["messages"][-1].content)
