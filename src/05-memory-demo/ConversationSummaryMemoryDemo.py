"""对话记忆（Memory）演示 3：ConversationSummaryMemory 的 LangChain 1.x 等价实现。

旧版写法（LangChain < 0.3）
---------------------------
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationSummaryMemory
    from langchain.memory import ConversationSummaryBufferMemory

    # 逐轮总结版（每轮对话后更新一次 summary）
    memory = ConversationSummaryMemory(llm=llm)
    # 按 token 上限总结版（历史超出 max_token_limit 时触发总结）
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)

    conversation = ConversationChain(memory=memory, llm=llm)

核心语义
--------
**摘要记忆（Summary Memory）**：不丢弃历史，而是把旧的历史压缩成一段摘要，
再把摘要（+ 最近几轮原文）一起送给模型。

  - ``ConversationSummaryMemory``：每轮对话结束后，用 LLM 把「已有摘要 + 最新两条
    消息」更新成新摘要；送进 prompt 的只有摘要本身（history = summary）。
  - ``ConversationSummaryBufferMemory``：历史 token 超过 ``max_token_limit`` 时，
    把超出部分总结掉，剩余部分保留原文（history = summary + 剩余消息）。

旧版默认提示词是 ``langchain_classic.memory.prompt.SUMMARY_PROMPT``：
"Progressively summarize the lines of conversation provided, adding onto the
previous summary returning a new summary."

LangChain 1.x 中的现状
----------------------
  - ``langchain.memory`` 模块已移除；
  - ``langchain.chains.ConversationChain`` 已移除；
  - ``langchain_classic.memory.summary.ConversationSummaryMemory`` 仍在，但
    ``deprecated(removal="2.0.0")``，官方替代方案写明为
    ``create_agent`` + summarization middleware；
  - ``ConversationSummaryBufferMemory``（带 ``max_token_limit`` 的那个）在本环境
    已**完全不可用**：``langchain_community`` 正在 sunset，相关模块已被移除；
  - ``RunnableWithMessageHistory`` / ``InMemoryChatMessageHistory`` 自 1.6.4 起
    也已 deprecated。

因此本文件给出两种现代写法：

  路线 A（官方推荐）：``create_agent`` + ``SummarizationMiddleware``
      - 官方内置中间件，trigger 控制何时触发总结，keep 控制保留多少历史；
      - 触发后用 ``RemoveMessage(REMOVE_ALL_MESSAGES)`` 清空原历史，
        再写入「摘要 + 保留的消息」，语义与旧版摘要记忆一致。

  路线 B（完全自定义，最贴近旧版）：``RunnableWithMessageHistory`` +
      自定义 ``SummaryBufferChatMessageHistory``
      - 自己实现 token 计数、摘要 prompt、增量总结；
      - 可以精确保留旧版 ``max_token_limit`` 的行为。

依赖：pip install langchain langchain-anthropic langgraph python-dotenv
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.logging_utils import get_logger

# 初始化 logger：日志同时输出到控制台和 logs/course.log
logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# 0. 加载环境变量
# --------------------------------------------------------------------------- #
load_dotenv()

# 兼容两种常见的环境变量命名：优先 ANTHROPIC_API_KEY，其次回落到 API_KEY。
# api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
# if not api_key:
#     raise RuntimeError(
#         "未找到 API Key，请在 .env 中设置 ANTHROPIC_API_KEY（或 API_KEY）"
#     )
# # Anthropic 客户端默认从环境变量 ANTHROPIC_API_KEY 读取密钥
# os.environ["ANTHROPIC_API_KEY"] = api_key
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")

# --------------------------------------------------------------------------- #
# 1. 初始化 LLM
# --------------------------------------------------------------------------- #
# LangChain 1.x 中 ChatAnthropic 的参数已统一：
#   model / api_key / base_url（旧的 anthropic_api_key 已重命名为 api_key）。
# llm = ChatAnthropic(
#     model="claude-sonnet-4-5-20250929",  # 可换成 claude-sonnet-4-5 或更新的模型
#     api_key=api_key,
#     temperature=0,
#     max_tokens=1024,
# )

# 也可以换成 DeepSeek（与本仓库其他 demo 保持一致）：
#
from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0,
)
logger.info("LLM 初始化完成：model=deepseek-chat, temperature=0")

# =========================================================================== #
# 路线 A（推荐）：create_agent + SummarizationMiddleware
# =========================================================================== #
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

# 触发阈值：历史 token 数超过 MAX_TOKEN_LIMIT 就触发总结
MAX_TOKEN_LIMIT = 100
# 总结后保留最近 KEEP_MESSAGES 条原文（不能为 0，最小值为 1）
KEEP_MESSAGES = 4

checkpointer = InMemorySaver()

# SummarizationMiddleware 是官方现成的摘要中间件：
#   trigger：何时触发（tokens / messages / fraction 三种维度）
#   keep   ：总结后保留多少上下文
#   model  ：用哪个模型生成摘要（可以用比主模型更便宜的小模型）
summarization = SummarizationMiddleware(
    model=llm,
    trigger=("tokens", MAX_TOKEN_LIMIT),
    keep=("messages", KEEP_MESSAGES),
)
logger.info(
    "摘要中间件配置完成：触发条件=token 数 > %d，总结后保留最近 %d 条原文",
    MAX_TOKEN_LIMIT,
    KEEP_MESSAGES,
)

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a friendly, helpful AI assistant.",
    checkpointer=checkpointer,
    middleware=[summarization],
    debug=True,
)

# 同一个 thread_id = 同一段会话
CONFIG = {"configurable": {"thread_id": "mala-summary-1"}}
logger.info("会话配置就绪：thread_id=%s", CONFIG["configurable"]["thread_id"])


def ask(text: str) -> None:
    """发起一轮对话并打印回答。

    Args:
        text: 用户本轮输入。
    """
    logger.info("用户输入：%s", text)
    result = agent.invoke({"messages": [("user", text)]}, CONFIG)
    answer = result["messages"][-1].content
    print(f"\n👤 用户: {text}")
    print(f"🤖 AI  : {answer}")
    logger.info("本轮回答完成，当前状态消息数=%d", len(result["messages"]))


print("=" * 70)
print(f"路线 A：create_agent + SummarizationMiddleware（trigger=tokens>{MAX_TOKEN_LIMIT}）")
print("=" * 70)

ask("Hi, my name is Mala")
ask("I'm doing well! Just having a conversation with an AI.")
ask("what is AI?")
ask("what is my name?")

print("\n" + "-" * 70)
print("会话状态（历史已被压缩为摘要 + 最近几条原文）")
print("-" * 70)
state = agent.get_state(CONFIG)
state_messages = state.values["messages"]
logger.info("会话状态共 %d 条消息（历史已被压缩，不再是原始条数）", len(state_messages))
for message in state_messages:
    role = message.type
    print(f"[{role}] {str(message.content)[:200]}")
    logger.debug("消息类型=%s, 长度=%d", role, len(str(message.content)))
logger.info(
    "摘要记忆的关键点：旧消息被压缩进摘要而非丢弃，因此语义仍然保留，"
    "但细节会丢失；最后一句提问中的名字依然能被模型想起来"
)

# --------------------------------------------------------------------------- #
# 进阶：摘要 + 窗口 组合
# --------------------------------------------------------------------------- #
# 摘要 + 滑动窗口是互补的两种策略：
#   - 窗口：硬丢弃旧消息，token 可控但会「失忆」；
#   - 摘要：压缩旧消息，保留语义但会丢失细节。
# 常见做法是「先窗口裁剪，再在超限时摘要」，二者可以叠加：
#
# from langchain.agents.middleware import before_model
# from langchain_core.messages import RemoveMessage
#
# @before_model
# def trim_window(state, runtime):
#     keep = 6
#     messages = state["messages"]
#     if len(messages) <= keep:
#         return None
#     return {"messages": [RemoveMessage(id=m.id) for m in messages[:-keep]]}
#
# agent = create_agent(
#     model=llm,
#     tools=[],
#     checkpointer=checkpointer,
#     middleware=[trim_window, summarization],
# )
#
# --------------------------------------------------------------------------- #
# 进阶：用更便宜的小模型做摘要
# --------------------------------------------------------------------------- #
# 摘要任务对模型要求不高，可以用小模型降低成本：
#
# from langchain_deepseek import ChatDeepSeek
#
# cheap_llm = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))
#
# summarization = SummarizationMiddleware(
#     model=cheap_llm,
#     trigger=("tokens", 2000),
#     keep=("messages", 20),
# )
#
# 还可以用 summarizer 参数接入自定义总结逻辑（函数/Runnable）：
#
# def my_summarizer(messages) -> str:
#     return "用户叫 Mala，主题是 AI"
#
# summarization = SummarizationMiddleware(
#     model=llm,
#     trigger=("tokens", 100),
#     summarizer=my_summarizer,
# )
#
# --------------------------------------------------------------------------- #
# 进阶：持久化
# --------------------------------------------------------------------------- #
# 换 checkpointer 即可，重启进程后摘要仍在：
#
# from langgraph.checkpoint.sqlite import SqliteSaver
#
# with SqliteSaver.from_conn_string("checkpoints.sqlite") as saver:
#     agent = create_agent(model=llm, tools=[], checkpointer=saver, middleware=[summarization])

# =========================================================================== #
# 路线 B（完全自定义，最贴近旧版 max_token_limit 语义）
# =========================================================================== #
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
#
#
# def count_tokens_approx(messages: list[BaseMessage]) -> int:
#     """粗略 token 估算：字符数 / 4。真实项目请用 tiktoken 或模型自带计数。"""
#     return sum(len(str(m.content)) for m in messages) // 4
#
#
# # 旧版 ConversationSummaryBufferMemory 用的就是这段提示词
# SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "Progressively summarize the lines of conversation provided, "
#             "adding onto the previous summary returning a new summary.",
#         ),
#         (
#             "human",
#             "Current summary:\n{summary}\n\n"
#             "New lines of conversation:\n{new_lines}\n\nNew summary:",
#         ),
#     ]
# )
#
#
# class SummaryBufferChatMessageHistory(BaseChatMessageHistory):
#     """超 token 上限时把旧消息压缩成摘要（= ConversationSummaryBufferMemory）。"""
#
#     def __init__(self, llm: Any, max_token_limit: int = 100, keep_messages: int = 2) -> None:
#         self.llm = llm
#         self.max_token_limit = max_token_limit
#         self.keep_messages = keep_messages
#         self._messages: list[BaseMessage] = []
#         self._summary: str = ""
#
#     @property
#     def messages(self) -> list[BaseMessage]:
#         """返回「摘要 + 最近若干条原文」，作为 history 注入 prompt。"""
#         prefix = (
#             [SystemMessage(content=f"Summary of earlier conversation:\n{self._summary}")]
#             if self._summary
#             else []
#         )
#         return prefix + self._messages[-self.keep_messages :]
#
#     def add_messages(self, messages: Sequence[BaseMessage]) -> None:
#         self._messages.extend(messages)
#         self._summarize_if_needed()
#
#     def _summarize_if_needed(self) -> None:
#         """历史超限时，把超出部分总结进 summary，并从原文中移除。"""
#         while (
#             count_tokens_approx(self._messages) > self.max_token_limit
#             and len(self._messages) > self.keep_messages
#         ):
#             overflow = self._messages[: len(self._messages) - self.keep_messages]
#             self._summary = self.llm.invoke(
#                 SUMMARY_PROMPT.format(
#                     summary=self._summary,
#                     new_lines=get_buffer_string(overflow),
#                 )
#             ).content
#             self._messages = self._messages[len(overflow) :]
#
#     def clear(self) -> None:
#         self._messages = []
#         self._summary = ""
#
#
# history = SummaryBufferChatMessageHistory(llm, max_token_limit=100, keep_messages=2)
#
# # 需要一个 prompt 模板来承接「历史消息 + 新输入」
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a friendly, helpful AI assistant."),
#         MessagesPlaceholder("history"),  # 历史消息占位符（摘要 + 最近原文）
#         ("human", "{input}"),             # 本轮新输入
#     ]
# )
#
# chain = RunnableWithMessageHistory(
#     prompt | llm,
#     get_session_history=lambda session_id: history,
#     input_messages_key="input",
#     history_messages_key="history",
# )
#
# cfg = {"configurable": {"session_id": "mala-summary-session"}}
# for text in [
#     "Hi, my name is Mala",
#     "I'm doing well! Just having a conversation with an AI.",
#     "what is AI?",
#     "what is my name?",   # 名字可能已被压缩进摘要里
# ]:
#     response = chain.invoke({"input": HumanMessage(content=text)}, cfg)
#     print(f"\n👤 用户: {text}\n🤖 AI  : {response.content}")
#
# print("\n摘要：")
# print(history._summary)
# print("窗口内消息：")
# for message in history.messages:
#     print(f"[{message.type}] {message.content}")

# =========================================================================== #
# 长期记忆（跨会话）：LangGraph store
# =========================================================================== #
# 摘要记忆仍属于「短期记忆」，只覆盖当前会话。若要跨会话记住用户偏好，
# 用 LangGraph 的 store（长期记忆）：
#
# from langgraph.store.memory import InMemoryStore
#
# store = InMemoryStore()
# agent = create_agent(model=llm, tools=[], store=store, checkpointer=checkpointer)
#
# # namespace 相当于命名空间/用户 id，key 是条目名
# store.put(("users", "mala"), "profile", {"name": "Mala", "interest": "AI"})
#
# # 检索长期记忆
# print(store.search(("users", "mala")))
