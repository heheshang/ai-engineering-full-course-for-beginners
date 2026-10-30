"""对话记忆（Memory）演示 2：ConversationBufferWindowMemory 的 LangChain 1.x 等价实现。

旧版写法（LangChain < 0.3）
---------------------------
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferWindowMemory

    memory = ConversationBufferWindowMemory(k=2)          # 只保留最近 2 轮
    conversation = ConversationChain(memory=memory, llm=llm)

它的核心语义是 **滑动窗口（sliding window）**：
只保留最近 k 轮对话（一轮 = 1 条 Human + 1 条 AI），更早的消息直接丢弃，
以此控制 token 消耗。底层实现就是 ``chat_memory.messages[-k * 2:]``。

LangChain 1.x 中的现状
----------------------
  - ``langchain.memory`` 模块已移除；
  - ``langchain.chains.ConversationChain`` 已移除；
  - ``langchain_classic.memory.buffer_window.ConversationBufferWindowMemory`` 仍在，
    但已标记 deprecated（removal="2.0.0"），官方替代方案是
    ``langchain.agents.create_agent`` + checkpointing / ``Store`` API；
  - ``RunnableWithMessageHistory`` 与 ``InMemoryChatMessageHistory`` 自 1.6.4 起
    也已 deprecated。

因此本文件给出两种现代写法：

  路线 A（官方推荐）：``create_agent`` + ``InMemorySaver`` + ``before_model`` 裁剪中间件
      - checkpointer 负责保存会话状态；
      - 自定义 middleware 在每次调用模型前，把窗口之外的历史消息从状态里删掉；
      - 效果等价于 ConversationBufferWindowMemory，且天然支持 thread_id 隔离。

  路线 B（语义最贴近旧类）：``RunnableWithMessageHistory`` + 自定义窗口历史类
      - 自己实现一个只暴露最近 k 轮的 ChatMessageHistory。

依赖：pip install langchain langchain-anthropic langgraph python-dotenv
"""

import os
from collections.abc import Sequence
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, RemoveMessage, trim_messages

import sys
from pathlib import Path

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
# Anthropic 客户端默认从环境变量 ANTHROPIC_API_KEY 读取密钥
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
# 路线 A（推荐）：create_agent + checkpointer + 窗口裁剪中间件
# =========================================================================== #
from langchain.agents import create_agent  # noqa: E402
from langchain.agents.middleware import before_model  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

# 窗口大小：保留最近 k 轮对话（一轮 = Human + AI 两条消息）
K = 2

# checkpointer 按 thread_id 保存完整会话状态
checkpointer = InMemorySaver()


@before_model
def trim_message_window(state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
    """在调用模型前，把窗口之外的历史消息从状态中删除。

    这是 ConversationBufferWindowMemory 在 LangChain 1.x 中的等价实现：
    保留最近 K 轮（K * 2 条消息），更早的消息用 RemoveMessage 标记删除。

    注意：
      - 返回 None 表示不修改状态（历史还没超过窗口）；
      - RemoveMessage 需要消息带 id，Agent 生成的消息自带 id；
      - 被裁剪的消息会真正从 checkpointer 的状态里移除，
        所以窗口是「滑动」的，而不是仅在送给模型前做截断。
    """
    messages: list[BaseMessage] = state["messages"]
    keep = K * 2
    if len(messages) <= keep:
        logger.debug("消息数 %d 未超过窗口上限 %d，无需裁剪", len(messages), keep)
        return None

    # 只保留最后 keep 条，更早的用 RemoveMessage 删除
    removed = messages[:-keep]
    logger.info(
        "触发窗口裁剪：当前 %d 条消息超过上限 %d，删除最早的 %d 条",
        len(messages), keep, len(removed),
    )
    return {"messages": [RemoveMessage(id=m.id) for m in removed]}


agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a friendly, helpful AI assistant.",
    checkpointer=checkpointer,
    middleware=[trim_message_window],
)

# 同一个 thread_id = 同一段会话
CONFIG = {"configurable": {"thread_id": "mala-window-1"}}
logger.info("会话配置就绪：thread_id=%s，窗口大小 K=%d 轮（最多 %d 条消息）",
             CONFIG["configurable"]["thread_id"], K, K * 2)


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
    logger.info("本轮回答完成，返回消息数=%d（裁剪后）", len(result["messages"]))


print("=" * 70)
print(f"路线 A：create_agent + InMemorySaver + 窗口裁剪（k={K}）")
print("=" * 70)

ask("Hi, my name is Mala")
ask("I'm doing well! Just having a conversation with an AI.")
ask("what is AI?")
ask("what is my name?")

print("\n" + "-" * 70)
print(f"当前窗口内的消息（最多 {K} 轮 / {K * 2} 条）")
print("-" * 70)
state = agent.get_state(CONFIG)
window_messages = state.values["messages"]
logger.info("窗口内当前保留 %d 条消息（上限 %d 条）", len(window_messages), K * 2)
if len(window_messages) > K * 2:
    logger.warning("消息数超过窗口上限，裁剪中间件可能未生效")
for message in window_messages:
    role = "Human" if message.type == "human" else "AI"
    print(f"[{role}] {message.content}")
logger.info("注意：最后的提问「what is my name?」发生在裁剪之后，"
            "名字所在的第 1 轮已被移出窗口，模型可能已经「忘记」")

# --------------------------------------------------------------------------- #
# 进阶：按 token 数而不是轮数裁剪
# --------------------------------------------------------------------------- #
# 窗口也可以用 token 预算控制，更贴近真实场景（长回复占比差异很大）：
#
# @before_model
# def trim_by_tokens(state, runtime):
#     trimmed = trim_messages(
#         state["messages"],
#         max_tokens=500,
#         token_counter="approximate",
#         strategy="last",
#         include_system=True,
#     )
#     removed = [m for m in state["messages"] if m not in trimmed]
#     if not removed:
#         return None
#     return {"messages": [RemoveMessage(id=m.id) for m in removed]}
#
# 或者直接用官方现成的 SummarizationMiddleware（超限时自动总结历史）：
#
from langchain.agents.middleware import SummarizationMiddleware
#
# agent = create_agent(
#     model=llm,
#     tools=[],
#     checkpointer=checkpointer,
#     middleware=[SummarizationMiddleware(model=llm, trigger=("messages", 20))],
# )

# --------------------------------------------------------------------------- #
# 进阶：把窗口状态持久化到磁盘
# --------------------------------------------------------------------------- #
# 换 checkpointer 即可，重启进程后仍能记住历史：
#
# from langgraph.checkpoint.sqlite import SqliteSaver
#
# with SqliteSaver.from_conn_string("checkpoints.sqlite") as saver:
#     agent = create_agent(model=llm, tools=[], checkpointer=saver, middleware=[trim_message_window])
#
# 生产环境用 PostgresSaver（pip install langgraph-checkpoint-postgres）。

# =========================================================================== #
# 路线 B（语义最贴近旧类，已 deprecated）
# =========================================================================== #
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.messages import HumanMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables.history import RunnableWithMessageHistory
#
#
# class WindowChatMessageHistory(BaseChatMessageHistory):
#     """只暴露最近 k 轮的聊天历史（= ConversationBufferWindowMemory 的核心）。"""
#
#     def __init__(self, k: int = 2) -> None:
#         self.k = k
#         self._messages: list[BaseMessage] = []
#
#     @property
#     def messages(self) -> list[BaseMessage]:
#         # 旧版实现就是 messages[-k * 2:]
#         return self._messages[-self.k * 2 :] if self.k > 0 else []
#
#     def add_messages(self, messages: Sequence[BaseMessage]) -> None:
#         self._messages.extend(messages)
#
#     def clear(self) -> None:
#         self._messages = []
#
#
# history = WindowChatMessageHistory(k=2)
#
# # 需要一个 prompt 模板来承接「历史消息 + 新输入」
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a friendly, helpful AI assistant."),
#         MessagesPlaceholder("history"),  # 历史消息占位符
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
# cfg = {"configurable": {"session_id": "mala-window-session"}}
# for text in [
#     "Hi, my name is Mala",
#     "I'm doing well! Just having a conversation with an AI.",
#     "what is AI?",
#     "what is my name?",   # 因为窗口只留 2 轮，这里模型可能已经忘了名字
# ]:
#     response = chain.invoke({"input": HumanMessage(content=text)}, cfg)
#     print(f"\n👤 用户: {text}\n🤖 AI  : {response.content}")
#
# print("\n窗口内消息：")
# for message in history.messages:
#     role = "Human" if isinstance(message, HumanMessage) else "AI"
#     print(f"[{role}] {message.content}")

# =========================================================================== #
# 长期记忆（跨会话）：LangGraph store
# =========================================================================== #
# 窗口记忆是「短期记忆」，只覆盖当前会话。若要跨会话记住用户偏好，
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
