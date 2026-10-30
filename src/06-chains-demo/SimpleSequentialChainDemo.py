"""链（Chain）演示 2：简单顺序链（SimpleSequentialChain）。

核心概念
--------
「顺序链」就是把 **多段 LLM 调用按固定顺序串起来**，前一段的输出自动成为
后一段的输入。本文件演示一个经典的二段式流程：

    主题(topic)
       │
       ▼
    ① 第一段提示词：生成博客文章的大纲（bullet point outline）
       │  输出：outline 字符串
       ▼
    ② 第二段提示词：按这个大纲写完整博文（blog post）
       │
       ▼
    最终结果：一篇完整文章

关键点：第二段模板里的占位符名必须与第一段的「输出形态」对齐。
这里用 `{abc}` 作为占位符名，配合 LCEL 的 **输入字典整体透传** 特性，
第一段返回的 AIMessage 会被转成字符串塞进 `{abc}`。

实现方式对比
------------
  - 旧版 LangChain 的 ``SimpleSequentialChain`` 类已经随 ``langchain.chains``
    一起被移除（LangChain 1.x 中 ``langchain.chains`` 模块已不存在）；
  - 现在用 **LCEL（LangChain Expression Language）** 直接表达同样的逻辑：
        full_chain = chain_one | chain_two
    管道会自动完成「把上一步的输出当作下一步的输入」这件事。

依赖：pip install langchain langchain-deepseek python-dotenv
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# --------------------------------------------------------------------------- #
# 0. 加载环境变量
# --------------------------------------------------------------------------- #
# load_dotenv() 会自动读取当前目录（或逐级向上）的 .env 文件，
# 把里面的键值对塞进 os.environ，不需要手动 export。
load_dotenv()

# 优先读取 DEEPSEEK_API_KEY，其次兼容通用命名 API_KEY；
# 两者都没有就直接报错，避免后面请求时才失败、不好排查。
api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY")
if not api_key:
    raise RuntimeError(
        "未找到 API Key，请在 .env 中设置 DEEPSEEK_API_KEY（或 API_KEY）"
    )
# 再显式写回环境变量：init_chat_model 内部会按 provider 自动读取对应变量名。
os.environ["DEEPSEEK_API_KEY"] = api_key

# --------------------------------------------------------------------------- #
# 1. 初始化 LLM
# --------------------------------------------------------------------------- #
# init_chat_model 是统一工厂函数，第一个参数是 "provider:model"：
#   - provider 决定走哪家的客户端（deepseek / openai / anthropic ...）
#   - model    是具体的模型名
# temperature=0 表示尽量确定性的输出，适合需要稳定复现的流水线。
llm = init_chat_model(model="deepseek-chat", api_key=api_key, temperature=0)

# --------------------------------------------------------------------------- #
# 2. 第一段链：生成文章大纲
# --------------------------------------------------------------------------- #
# 提示词模板：用 {topic} 作为占位符，调用时传入实际主题。
template = "Give me a simple bullet point outline for a blog post on {topic}"

# 演示主题
topic = "AI"

# 由模板创建提示词对象。
# ChatPromptTemplate 会把模板编译成「消息列表 + 输入变量声明」，
# 后面可以直接用 | 管道接到模型上。
first_prompt = ChatPromptTemplate.from_template(template)

# 第一段链 = 提示词 | 模型
# 运行时：first_prompt.invoke({"topic": topic}) -> PromptValue
#         llm.invoke(PromptValue)                   -> AIMessage
chain_one = first_prompt | llm

# 先单独跑一遍第一段，看看大纲长什么样（便于观察两段之间传递的内容）
chain_one_result = chain_one.invoke(topic)

# \033[1m = 加粗，\033[0m = 重置样式；用 ANSI 转义给控制台输出加一点层次感
print("\n\033[1m Give me a simple bullet point outline for a blog post")
print("-" * 50)
print("\033[0m" + chain_one_result.content)

# --------------------------------------------------------------------------- #
# 3. 第二段链：按大纲写文章
# --------------------------------------------------------------------------- #
# 占位符名这里叫 {abc}（随意命名，只要和后续传给 invoke 的字典键一致）。
# 注意：接成 full_chain 之后，上游输出会「整体」作为输入字典传给第二段模板，
# 所以这里可以拿到上一段生成的大纲文本。
template = "Write a blog post using this outline: {abc}"

# 创建第二段提示词
second_prompt = ChatPromptTemplate.from_template(template)

# 第二段链 = 提示词 | 模型
chain_two = second_prompt | llm

# print('CHAIN TWO ==========>', chain_two)   # 调试用：可打印出链的结构

# --------------------------------------------------------------------------- #
# 4. 串成完整顺序链并调用
# --------------------------------------------------------------------------- #
# full_chain = chain_one | chain_two
#   运行时数据流：
#     topic（字符串）
#       → chain_one：生成 AIMessage(大纲)
#       → 隐式转换：非字典输出会被包成 {"output": ...} 之类的输入字典
#       → chain_two：模板取到 {abc}，生成 AIMessage(完整文章)
#
# 也就是说，LCEL 自动完成了「上一步输出 → 下一步输入」的传递，
# 这正是旧版 SimpleSequentialChain 的核心行为。
full_chain = chain_one | chain_two

# invoke 传入主题；后续段的输入由管道自动衔接。
result = full_chain.invoke(topic)

print("\n\033[1m Write a blog post using the outline")
print("-" * 50)
print("\033[0m" + result.content)

# --------------------------------------------------------------------------- #
# 5. 补充说明
# --------------------------------------------------------------------------- #
# 5.1 如果想让占位符名更直观，可以把第二段模板写成：
#        template = "Write a blog post using this outline: {abc}"
#     并在两段之间插入一个显式的映射函数，例如：
#        from langchain_core.runnables import RunnableLambda
#        def to_outline(ai_message):
#            return {"abc": ai_message.content}
#        full_chain = chain_one | RunnableLambda(to_outline) | chain_two
#     好处是字段名可控，不依赖 LCEL 的隐式包装规则。
#
# 5.2 三段及以上也是同理，继续往后 | 即可：
#        full_chain = chain_one | chain_two | chain_three
#
# 5.3 旧版 API（已移除，仅供理解历史写法）：
#        from langchain.chains import SimpleSequentialChain
#        chain = SimpleSequentialChain(chains=[chain_one, chain_two])
#        chain.invoke(topic)
