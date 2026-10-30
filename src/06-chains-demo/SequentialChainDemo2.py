"""链（Chain）演示 3-2：带并行分支的顺序链（Sequential Chain + RunnableParallel）。

与「演示 3」的关系
------------------
``SequentialChainDemo.py``（演示 3）演示的是「多段串联」：
上一段的输出整体喂给下一段，靠 LCEL 的隐式包装规则完成数据传递。

本文件（演示 3-2）补上的是「**并行分支 + 字段映射**」：
用字典形式精确控制每一段能拿到哪些数据，而不是依赖隐式规则。

编号说明：本文件是演示 3 的进阶版，所以编号为 3-2；
下一个文件 ``RouterChainDemo.py`` 是演示 4。

核心概念
--------
本文件演示 LangChain LCEL 中两个非常关键的能力：

  1. **输出解析器（Output Parser）**：把 AIMessage 转成纯字符串；
  2. **并行/字典映射（RunnableParallel 的字典简写）**：一次链中同时
     保留上游已有的字段，并新增由某个分支计算出的字段。

业务流程（先生成主题，再围绕主题生成 3 个问题）：

    {}（空输入）
       │
       ▼
    ┌─ topic_chain ──────────────────────┐
    │  topic_prompt → llm → StrOutputParser │  →  "topic": "随机学术主题"
    └──────────────────────────────────────┘
       │
       ▼  RunnablePassthrough()（原样透传所有字段）
    ┌───────────────────────────┬──────────────────────────────┐
    │ "topic": itemgetter("topic")│ "questions": question_chain    │
    │  → 直接取出上游的 topic    │  → 模板用 {topic} 生成 3 个问题 │
    └───────────────────────────┴──────────────────────────────┘
       │
       ▼
    结果字典：{"topic": ..., "questions": ...}

与前面几个 demo 的差别
---------------------
  - ``SingleChainDemo.py``：一段，prompt | llm；
  - ``SequentialChainDemo.py``：多段串联，上一段输出整体喂给下一段；
  - **本文件**：串联 + 并行 + 字段映射，用「字典」精确控制每一段
    能拿到哪些数据，而不是依赖隐式包装规则。
依赖：pip install langchain langchain-deepseek python-dotenv
"""

import os
import sys
from operator import itemgetter
from pathlib import Path

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

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

# --------------------------------------------------------------------------- #
# 1. 初始化 LLM
# --------------------------------------------------------------------------- #
# init_chat_model 是统一的模型工厂函数：
#   - provider 决定用哪家的客户端（deepseek / openai / anthropic ...）
#   - model    是具体模型名
# temperature=0 让输出尽量确定，便于复现。
llm = init_chat_model(model="deepseek-chat", api_key=api_key, temperature=0)
logger.info("LLM 初始化完成：model=deepseek-chat, temperature=0")

# --------------------------------------------------------------------------- #
# 2. 第一条链：随机生成一个学术主题
# --------------------------------------------------------------------------- #
# from_messages 接收「消息列表」，元组简写形式为 (角色, 模板内容)：
#   ("system", ...) → 系统消息，设定角色与行为
#   ("user",   ...) → 用户消息
#   ("ai",     ...) → 助手消息（模拟已有对话历史）
#
# 注意这里的提示词里 **没有任何占位符**，
# 因为这一段不需要外部输入，只要模型自由发挥即可。
topic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant who generates interesting topics."),
        ("user", "Generate a random academic topic for discussion."),
    ]
)

# 组装：提示词 → 模型 → 输出解析器
#
# StrOutputParser 的作用：把模型的 AIMessage 直接转成 str。
# 没有它的话，链路输出的是 AIMessage 对象，
# 后面用字符串的 .format()、字符串拼接等操作就会出问题。
#
# topic_chain 的输入是任意值（这里是空字典 {}），输出是 str。
topic_chain = topic_prompt | llm | StrOutputParser()
logger.info(
    "第一条链构建完成：生成学术主题，输入变量=%s（为空，说明不依赖外部输入）",
    topic_prompt.input_variables,
)

# --------------------------------------------------------------------------- #
# 3. 第二条链：围绕主题生成 3 个发人深省的问题
# --------------------------------------------------------------------------- #
# 这里出现了占位符 {topic}，表示需要外部传入主题。
question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant who generates thought-provoking topics.",
        ),
        ("user", "Generate 3 thought-provoking questions about this topic: {topic}."),
    ]
)

# question_chain 的输入是 {"topic": "..."}，输出是 str。
question_chain = question_prompt | llm | StrOutputParser()
logger.info("第二条链构建完成：围绕主题生成问题，输入变量=%s", question_prompt.input_variables)

# --------------------------------------------------------------------------- #
# 4. 组装完整链：串联 + 并行
# --------------------------------------------------------------------------- #
# sequential_chain = (
#     {"topic": topic_chain}          # ← 第 ① 步：并行分支
#     | RunnablePassthrough()        # ← 第 ② 步：原样透传
#     | {"topic": itemgetter("topic"), "questions": question_chain}  # ← 第 ③ 步
# )
#
# 逐段解释：
#
# ① {"topic": topic_chain}
#    字典形式是 RunnableParallel 的简写：字典的每个键对应一个分支，
#    键名就是输出字段名，值是一个 Runnable。
#    topic_chain 没有输入变量，所以传什么给它都无所谓，
#    它的输出被放进字段 "topic"。
#
# ② RunnablePassthrough()
#    默认的 RunnablePassthrough 会把上一步的整个输出字典原封不动传下去。
#    这一步的作用是「保留 topic 字段」，让第 ③ 步能同时用到它。
#    （也可以省略，但显式写出来可读性更好；
#      另一种常见写法是 RunnablePassthrough.assign(...)，
#      用于在保留原字段的同时追加新字段。）
#
# ③ {"topic": itemgetter("topic"), "questions": question_chain}
#    这是一个「并行」步骤，两个分支同时基于上游的 {"topic": ...} 执行：
#      - "topic": itemgetter("topic")
#        itemgetter 是 Python 标准库函数，取字典里的某个键。
#        放在这里相当于「把上游的 topic 字段原样抄到结果的 topic 字段」，
#        这样最终结果里既能拿到主题，也能拿到问题。
#      - "questions": question_chain
#        真正发起 LLM 调用，把 {"topic": ...} 填进 {topic} 占位符，
#        生成 3 个问题，输出放进 "questions" 字段。
#
# 关键点：③ 中的两个分支是「并行」的，但只有 question_chain 需要调用 API。
# 而 ③ 依赖 ② 传来的 topic，② 又依赖 ① 生成的 topic，
# 因此整体数据流仍是「先生成主题，再基于主题生成问题」。
sequential_chain = (
    {"topic": topic_chain}
    | RunnablePassthrough()
    | {"topic": itemgetter("topic"), "questions": question_chain}
)
logger.info("完整链组装完成：3 步（并行生成主题 → 透传 → 并行取主题+生成问题）")

# --------------------------------------------------------------------------- #
# 5. 调用并打印结果
# --------------------------------------------------------------------------- #
# 初始输入是空字典 {}：
# 第一步的 topic_chain 不需要任何输入变量，所以空字典完全够用。
logger.info("开始执行链（初始输入为空字典）")
results = sequential_chain.invoke({})
logger.info("链执行完成，返回字段：%s", list(results.keys()))
logger.info("生成的主题：%s", results["topic"])
logger.info("生成的问题，长度=%d 字符", len(results["questions"]))

# 最终输出是一个字典：
#   {"topic": "生成的学术主题", "questions": "3 个问题"}
# 注意取值要用 results["questions"]，而不是 results.content。

# \033[1m = 加粗，\033[0m = 重置样式；用于给控制台输出加层次感
print("\n\033[1m Generate 3 thought-provoking questions about a random academic topic")
print("-" * 50)
print("\033[0m" + results["questions"])

# --------------------------------------------------------------------------- #
# 6. 补充说明与常见坑
# --------------------------------------------------------------------------- #
# 6.1 为什么中间一定要 RunnablePassthrough()？
#     第 ③ 步需要从上游字典里取 topic。字典简写的并行步骤
#     默认会把 **整个上游输出** 作为各分支的输入，
#     若上游是 {"topic": ...}，直接写
#         {"topic": itemgetter("topic"), "questions": question_chain}
#     也是可行的（itemgetter 能取到、question_chain 的 {topic} 也已被填充）；
#     保留 RunnablePassthrough() 是为了结构更清晰，也方便在此处插入
#     额外的字段加工逻辑。
#
# 6.2 itemgetter vs RunnableLambda
#     - itemgetter("topic")：只取字典某个键，最轻量；
#     - RunnableLambda(lambda d: {"topic": d["topic"]})：可以做任意变换，
#       但 Lambda 默认不支持流式（stream），需要流式时要自定义 Runnable。
#
# 6.3 常见错误
#     - 忘记 StrOutputParser：链路输出是 AIMessage，
#       做字符串拼接会报类型错误；
#     - 字典键名与模板占位符不匹配：会报 "missing variables"；
#       例如模板写 {topic}，但上游字典里是 "subject"；
#     - 模板里想输出字面量花括号：要写成 {{topic}} 转义。
#
# 6.4 调试技巧
#     - print(topic_chain.input_variables)   查看该链需要哪些输入
#     - print(sequential_chain)              打印整条链的结构
#     - 分段 invoke，逐段打印中间结果
#
# 6.5 旧版 API（已移除，仅供理解历史写法）：
#         from langchain.chains import LLMChain
#         chain = LLMChain(llm=llm, prompt=topic_prompt)
#         chain.run()
