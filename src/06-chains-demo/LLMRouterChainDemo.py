"""链（Chain）演示 5：多提示路由链（MultiPromptChain 的现代写法）。

背景：MultiPromptChain 还能用吗？
---------------------------------
``MultiPromptChain`` 原属于 ``langchain.chains`` 模块，该模块在 LangChain 1.x
中已被移除（从 ``langchain_classic`` 迁出），且类本身标注了：

    @deprecated(since="0.2.12", removal="2.0.0",
                alternative="langchain.agents.create_agent")

也就是说：**新项目不应该再用它**。但它的思路（LLM 先判断该用哪个提示词，
再交给对应分支）依然是「路由」的标准范式，值得用 LCEL 重新实现一遍。

本文件给出两种现代写法
----------------------
  写法 A（推荐，纯 LCEL）：用 ``with_structured_output`` 让路由模型输出
          严格的 ``{destination, next_inputs}``，再动态分发到对应分支。
          与旧版 MultiPromptChain 行为几乎一一对应，且支持流式 / 批量。

  写法 B（可选）：用 ``create_agent`` + 工具调用 / 子代理实现路由，
          更适合「分支之间也需要工具」或「需要多轮记忆」的场景。

旧版 MultiPromptChain 到底做了什么
----------------------------------
  1. 把所有 prompt_infos 拼成一段「候选提示词清单」交给 LLM；
  2. 要求 LLM 输出 JSON：``{"destination": "<名称或 DEFAULT>", "next_inputs": "<改写后的问题>"}``；
  3. 解析 JSON，按 destination 找到对应分支，把 next_inputs 喂进去；
  4. 若 destination == "DEFAULT"，走默认分支（默认是对话链）。

下面的写法 A 完整复刻了 1~4 步。

依赖：pip install langchain langchain-deepseek python-dotenv
"""

import os
import sys
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
llm = init_chat_model(model="deepseek-chat", api_key=api_key, temperature=0)
logger.info("LLM 初始化完成：model=deepseek-chat, temperature=0")

# --------------------------------------------------------------------------- #
# 2. 定义各个分支的提示词模板
# --------------------------------------------------------------------------- #
# 三个模板都使用同一个输入变量名 {input}，
# 这样路由链把「改写后的问题」传给任意分支时都能直接填进去。
beginner_template = """You are a physics teacher who is really
focused on beginners and explaining complex topics in simple to understand terms.
You assume no prior knowledge. Here is the question\n{input}"""

expert_template = """You are a world expert physics professor who explains physics topics
to advanced audience members. You can assume anyone you answer has a
PhD level understanding of Physics. Here is the question\n{input}"""

# 「空问题」分支：原版这里是一个空模板，这里换成一句可用的兜底回答，
# 避免模型拿到的提示词毫无意义。
empty_template = "The question is empty, please ask me a physics question."

# --------------------------------------------------------------------------- #
# 3. 构造「候选提示词清单」（对应旧版 prompt_infos）
# --------------------------------------------------------------------------- #
# 每个条目包含：
#   name              —— 路由时 LLM 返回的目标名（必须与后续分支名一致）
#   description       —— 给路由模型看的用途描述
#   prompt_template   —— 该分支使用的提示词模板
prompt_infos = [
    {
        "name": "empty",
        "description": "Replies to empty questions",
        "prompt_template": empty_template,
    },
    {
        "name": "advanced physics",
        "description": "Answers advanced physics questions",
        "prompt_template": expert_template,
    },
    {
        "name": "beginner physics",
        "description": "Answers basic beginner physics questions",
        "prompt_template": beginner_template,
    },
]

# --------------------------------------------------------------------------- #
# 写法 A：纯 LCEL 实现（推荐）
# --------------------------------------------------------------------------- #
# 4.1 先把每个分支组装成可运行的链：prompt | model | StrOutputParser
#     StrOutputParser 把 AIMessage 转成 str，后续分支的输出就是纯文本。
destination_chains: dict[str, Any] = {}
for info in prompt_infos:
    prompt = ChatPromptTemplate.from_template(info["prompt_template"])
    destination_chains[info["name"]] = prompt | llm | StrOutputParser()
logger.info("候选分支构建完成：%s", list(destination_chains.keys()))

# 4.2 定义路由模型的「结构化输出 schema」
#     这是与旧版 MultiPromptChain 最大的区别：
#     旧版靠模型输出 ```json ... ``` 再用正则/JSON 解析（容易解析失败），
#     新版用 with_structured_output 让模型直接返回符合 schema 的对象（稳定可靠）。
#
#     destination 用 Literal 限定只能是这几个名字之一，
#     模型选错名字时会被 schema 校验拦下，而不是静默走错分支。
#
#     next_inputs 复刻旧版语义：允许路由模型对原问题做「改写优化」
#     （例如把模糊问题补全），再交给分支处理。
DestinationName = Literal["empty", "advanced physics", "beginner physics"]


class RouteQuery(BaseModel):
    """路由结果：选哪个分支 + 传给分支的问题。"""

    destination: DestinationName = Field(description="选中的分支名称")
    next_inputs: str = Field(description="传给该分支的问题，可对原问题做改写")


# 4.3 构造「路由提示词」
#     注意这里是把候选清单动态拼进去的，逻辑与旧版 from_prompts 一致。
destinations_str = "\n".join(f"{info['name']}: {info['description']}" for info in prompt_infos)

router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Given a raw text input to a language model, select the prompt best suited for the input.
You will be given the names of the available prompts and a description of what the prompt is best suited for.
You may also revise the original input if you think revising it will lead to a better response.

Available prompts:
{destinations}

Only respond with the destination name and the (possibly revised) input.""",
        ),
        ("human", "{input}"),
    ]
).partial(destinations=destinations_str)

# 4.4 路由链：提示词 → 模型（结构化输出）
#     with_structured_output 会把 schema 转成工具/JSON 约束，
#     DeepSeek 的 function_calling 模式支持这个能力。
router_chain = router_prompt | llm.with_structured_output(RouteQuery)

# 4.5 组装完整的多提示路由链
#
#     数据流：
#       {"input": "问题"}
#          │
#          ▼  router_chain → RouteQuery(destination, next_inputs)
#          ▼
#     {"destination": ..., "next_inputs": ...}
#          │
#          ▼  RunnableLambda：按 destination 从 destination_chains 里选链
#          │
#          ▼  该分支用 next_inputs 作为 {input} 执行 → 最终答案（str）
#
#     注意这里用 next_inputs 作为分支输入，而不是最初的 input，
#     这样才和旧版 MultiPromptChain 的「可改写问题」语义一致。
multi_prompt_chain = (
    RunnablePassthrough.assign(
        route=router_chain,
    )
    | RunnableLambda(
        # 从路由结果里取出分支名与改写后的问题
        lambda x: (x["route"].destination, x["route"].next_inputs)
    )
    | RunnableLambda(
        # 按分支名挑选要执行的链，并喂入改写后的问题
        lambda x: destination_chains[x[0]].invoke({"input": x[1]})
    )
)


# --------------------------------------------------------------------------- #
# 写法 B：用 create_agent 实现路由（可选）
# --------------------------------------------------------------------------- #
# 当「分支本身也需要工具调用」或「需要多轮记忆 / 中断恢复」时，
# 用 Agent 更合适：把每个分支包装成一个「子代理」或「工具」，
# 让主 Agent 自己决定调用哪一个。
#
# from langchain.agents import create_agent
# from langchain_core.tools import tool
#
#
# @tool
# def beginner_physics(question: str) -> str:
#     """用初学者友好的方式回答物理问题。"""
#     return (ChatPromptTemplate.from_template(beginner_template)
#             | llm | StrOutputParser()).invoke({"input": question})
#
#
# @tool
# def advanced_physics(question: str) -> str:
#     """用专家级深度回答物理问题。"""
#     return (ChatPromptTemplate.from_template(expert_template)
#             | llm | StrOutputParser()).invoke({"input": question})
#
#
# agent = create_agent(
#     model=llm,
#     tools=[beginner_physics, advanced_physics],
#     system_prompt=(
#         "你是物理问题路由器。"
#         "问题简单/面向初学者时调用 beginner_physics；"
#         "问题专业/深入时调用 advanced_physics。"
#     ),
# )
#
# result = agent.invoke({"messages": [("user", "How do magnets work explain?")]})
# print(result["messages"][-1].content)

# --------------------------------------------------------------------------- #
# 6. 调用示例
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    questions = [
        "How do magnets work explain?",
        "How do Feynman Diagrams work?",
    ]

    logger.info("开始运行多提示路由链演示，共 %d 个问题", len(questions))
    for idx, q in enumerate(questions, start=1):
        logger.info("[%d/%d] 处理问题：%s", idx, len(questions), q)
        # \033[0m 重置样式；结果已经是 str，可直接打印
        answer = multi_prompt_chain.invoke({"input": q})
        logger.info("[%d/%d] 回答完成，长度=%d 字符", idx, len(questions), len(answer))

        print(f"\n\033[1m 问题：{q}")
        print("-" * 50)
        print("\033[0m" + answer)
    logger.info("多提示路由链演示结束")

# --------------------------------------------------------------------------- #
# 7. 旧版写法对照（已废弃，仅供理解历史实现）
# --------------------------------------------------------------------------- #
# from langchain_classic.chains.router import MultiPromptChain
#
# chain = MultiPromptChain.from_prompts(llm, prompt_infos, verbose=True)
# chain.invoke("How do magnets work explain?")
#
# 注意：
#   - 需要额外安装 langchain-classic 包；
#   - 类已标记 deprecated，官方推荐改用 create_agent；
#   - 依赖模型输出 ```json 代码块并解析，解析失败会抛 OutputParserException；
#   - 不支持流式与批量调用。
#
# 如果只是想「最小改动跑通旧代码」，也可以直接安装：
#     uv add langchain-classic
#
# --------------------------------------------------------------------------- #
# 8. 常见坑
# --------------------------------------------------------------------------- #
# 8.1 with_structured_output 的 method 选择：
#     - function_calling（默认）：DeepSeek/OpenAI 均支持，最稳；
#     - json_mode：要求模型支持 JSON 输出模式；
#     - json_schema：要求 provider 支持严格 JSON Schema。
# 8.2 Literal 里漏写某个分支名，路由模型就永远选不到它。
# 8.3 分支名含空格（"advanced physics"）时，用作字典 key 没问题，
#     但若用 with_structured_output 的 Literal，务必与字符串完全一致。
# 8.4 路由本身要消耗一次 API 调用，整条链的总调用次数 = 1（路由）+ 1（分支）。
