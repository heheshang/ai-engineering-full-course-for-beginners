"""对话记忆（Memory）演示：旧版 ConversationBufferMemory 的 LangChain 1.x 等价实现。

背景 / 迁移说明
--------------
旧版 LangChain（< 0.3）示例：

    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory
    conversation = ConversationChain(memory=memory, llm=llm)

在 LangChain 1.x 中：

  - ``langchain.memory`` 模块已移除；
  - ``langchain.chains.ConversationChain`` 已移除；
  - ``RunnableWithMessageHistory`` / ``InMemoryChatMessageHistory``
    也已在 1.6.4 标记为 deprecated，官方推荐改用 **LangGraph 的 checkpointer**。

所以本文件给出两种现代写法：

  路线 A（官方推荐）：``langchain.agents.create_agent`` + ``InMemorySaver``
      - checkpointer 按 ``thread_id`` 保存完整会话状态，天然支持多轮记忆与会话隔离；
      - 落盘持久化只需把 ``InMemorySaver`` 换成 ``SqliteSaver`` / ``PostgresSaver``。

  路线 B（仍可用，已 deprecated）：``RunnableWithMessageHistory``
      - 语义上最接近旧的 ConversationBufferMemory（内存里存全部消息缓冲区）；
      - 需要自己用 ChatPromptTemplate + MessagesPlaceholder 组装 prompt。

另外还有第三种记忆：LangGraph ``store``（长期记忆，跨会话的用户画像），
见文件末尾注释。

依赖：pip install langchain langchain-anthropic langgraph python-dotenv
"""

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek

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
api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY")
if not api_key:
    raise RuntimeError(
        "未找到 API Key，请在 .env 中设置 ANTHROPIC_API_KEY（或 API_KEY）"
    )
# Anthropic 客户端默认从环境变量 ANTHROPIC_API_KEY 读取密钥
# os.environ["ANTHROPIC_API_KEY"] = api_key
os.environ["DEEPSEEK_API_KEY"] = api_key

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
llm = ChatDeepSeek(
    model="deepseek-chat",  # 可换成 claude-sonnet-4-5 或更新的模型
    api_key=api_key,
    temperature=0,
    max_tokens=1024,
)
logger.info("LLM 初始化完成：model=deepseek-chat, temperature=0, max_tokens=1024")


# =========================================================================== #
# 路线 A（推荐）：LangGraph create_agent + checkpointer（短期记忆）
# =========================================================================== #
from langchain.agents import create_agent  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

# checkpointer 负责「记住」每个 thread_id 下的全部消息与状态
checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a friendly, helpful AI assistant.",
    checkpointer=checkpointer,
)

# 同一个 thread_id = 同一段会话，模型就能记住前面说过的话
CONFIG = {"configurable": {"thread_id": "mala-thread-1"}}
logger.info("会话配置就绪：thread_id=%s，checkpointer=InMemorySaver", CONFIG["configurable"]["thread_id"])


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
    # 记录本轮的消息累计数，用于观察记忆是如何累积的
    logger.info("本轮回答完成，当前会话累计消息数=%d", len(result["messages"]))


ask("Hi, my name is Mala")
ask("I'm doing well! Just having a conversation with an AI.")
ask("what is AI?")
ask("what is my name?")

print("\n" + "=" * 70)
print("会话状态（等价于旧版 memory.buffer 的内容）")
print("=" * 70)
state = agent.get_state(CONFIG)
messages = state.values["messages"]
logger.info("会话状态共 %d 条消息（缓冲区记忆不做任何裁剪，全部保留）", len(messages))
for message in messages:
    role = "Human" if message.type == "human" else "AI"
    print(f"[{role}] {message.content}")

# --------------------------------------------------------------------------- #
# 路线 A 的进阶：持久化到磁盘
# --------------------------------------------------------------------------- #
# 把 InMemorySaver 换成 SqliteSaver 即可重启后仍记得历史：
#
# from langgraph.checkpoint.sqlite import SqliteSaver
#
# with SqliteSaver.from_conn_string("checkpoints.sqlite") as saver:
#     agent = create_agent(model=llm, tools=[], checkpointer=saver)
#     agent.invoke({"messages": [("user", "Hi, my name is Mala")]},
#                  {"configurable": {"thread_id": "t-1"}})
#
# 生产环境用 PostgresSaver（配合 pip install langgraph-checkpoint-postgres）。

# =========================================================================== #
# 路线 B（语义最接近旧版 ConversationBufferMemory，已 deprecated）
# =========================================================================== #
# from langchain_core.chat_history import InMemoryChatMessageHistory
# from langchain_core.messages import HumanMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables.history import RunnableWithMessageHistory
#
# # 内存消息缓冲区：保存全部 Human/AI 消息，无窗口裁剪（buffer 语义）
# chat_history = InMemoryChatMessageHistory()
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
#     get_session_history=lambda session_id: chat_history,
#     input_messages_key="input",
#     history_messages_key="history",
# )
#
# cfg = {"configurable": {"session_id": "mala-session"}}
# for text in [
#     "Hi, my name is Mala",
#     "I'm doing well! Just having a conversation with an AI.",
#     "what is AI?",
#     "what is my name?",
# ]:
#     response = chain.invoke({"input": HumanMessage(content=text)}, cfg)
#     print(f"\n👤 用户: {text}\n🤖 AI  : {response.content}")
#
# for message in chat_history.messages:
#     role = "Human" if isinstance(message, HumanMessage) else "AI"
#     print(f"[{role}] {message.content}")

# =========================================================================== #
# 第三种记忆：长期记忆 store（跨会话的用户画像 / 知识）
# =========================================================================== #
# 若需要「跨会话记住用户偏好」，用 LangGraph 的 store（长期记忆）：
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
