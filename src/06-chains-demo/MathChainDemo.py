"""链（Chain）演示 6：数学链（LangChain 1.x 现代写法）。

背景
----
课程里最早演示的写法是旧版 ``LLMMathChain``：

    from langchain.chains import LLMMathChain          # ❌ 已移除
    math_chain = LLMMathChain.from_llm(llm=llm, prompt=math_prompt)

它已被标记为 deprecated（``langchain_classic`` 里仍保留旧实现）：

    @deprecated(since="0.2.13", removal="2.0.0",
                alternative="langchain.agents.create_agent")

它的做法是「让模型输出一行 Python 表达式 → 再把表达式求值」，
这依赖模型配合输出特定格式的代码块，比较脆弱，
而且不支持流式、批量与多轮记忆。

本文件用官方推荐的新写法实现**同样的效果**：
**``create_agent`` + ``@tool`` 计算器工具**。
模型通过 **function calling** 调用计算器工具拿到精确数值，
不再依赖解析代码块，因此天然支持流式、批量与记忆。

    原始（旧链）                          现在（1.x Agent + 工具）
    ─────────────────────────             ─────────────────────────
    prompt → "Answer: 100 * 4"            prompt → tool_call: calculator("100 * 4")
           → 代码解析 → 求值                       → numexpr 求值
           → 返回字符串答案                        → 模型复述最终答案

依赖：pip install langchain langchain-openai python-dotenv numexpr
"""

import math
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

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
# load_dotenv() 自动读取当前目录（或逐级向上）目录中的 .env 文件，
# 把键值对写入 os.environ，无需手动 export。
load_dotenv()

# 读取 Key：优先 OPENAI_API_KEY，其次兼容通用命名 API_KEY；
# 都没有就立刻报错，避免等到请求接口时才失败、难以排查。
api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY")
if not api_key:
    raise RuntimeError("未找到 API Key，请在 .env 中设置 OPENAI_API_KEY（或 API_KEY）")
# 再显式写回环境变量，供 init_chat_model 按 provider 自动读取。
os.environ["DEEPSEEK_API_KEY"] = api_key

# --------------------------------------------------------------------------- #
# 1. 初始化 LLM
# --------------------------------------------------------------------------- #
# init_chat_model 是统一的模型工厂函数：
#   - provider 决定用哪家的客户端（openai / deepseek / anthropic ...）
#   - model    是具体模型名
#
# 想换成 DeepSeek 只需改这两行：
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
llm = init_chat_model(model="deepseek-chat", temperature=0)
# model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
# llm = init_chat_model(model=model_name, api_key=api_key, temperature=0)

# --------------------------------------------------------------------------- #
# 2. 定义计算器工具
# --------------------------------------------------------------------------- #
# 关键设计：用 numexpr 而不是 Python 内置 eval。
# numexpr 只能解析「数学表达式」，无法执行任意 Python 代码
# （没有 import、函数定义、属性访问等能力），因此更安全。
#
# @tool 装饰器会：
#   - 从函数签名和 docstring 自动生成给模型看的工具描述；
#   - 自动把返回值转成 ToolMessage 放进对话历史。
#
# docstring 非常重要——模型就是靠它决定「什么时候该调这个工具、
# 参数长什么样」，所以要写清输入格式并给例子。
try:
    import numexpr

    _HAS_NUMEXPR = True
except ImportError:  # pragma: no cover - 依赖缺失时的兜底
    _HAS_NUMEXPR = False


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression and return the numeric result.

    The expression must be a single line mathematical expression that solves
    the problem, using Python arithmetic syntax.

    Examples:
        "100 * 4" for "100 times 4"
        "37593 * 67" for "37593 times 67"
        "17**11" for "17 raised to the power of 11"
    """
    # 容错：模型可能带上前缀 "Answer: " 或用 ``` 包裹，这里统一清洗。
    expr = expression.strip().strip("`")
    if expr.lower().startswith("answer:"):
        expr = expr[len("answer:") :].strip()

    logger.info("计算器工具被调用：expression=%r", expr)

    if not _HAS_NUMEXPR:
        # 没装 numexpr 时退化为内置 eval（仅演示用，生产环境务必装 numexpr）
        logger.warning("未安装 numexpr，回退到内置 eval（安全性较低）")
        try:
            result = str(eval(expr, {"__builtins__": {}}, {}))
            logger.info("计算结果：%s", result)
            return result
        except Exception as exc:
            logger.error("计算失败：%s", exc)
            return f"Calculation error: {exc}"

    # global_dict={} 限制访问全局变量；
    # local_dict 注入常用数学常量，让模型可以写 pi / e。
    try:
        result = str(
            numexpr.evaluate(
                expr,
                global_dict={},
                local_dict={"pi": math.pi, "e": math.e},
            )
        )
    except Exception as exc:
        # numexpr 只支持数值运算，遇到变量赋值/循环/函数定义会报解析错误。
        # 把错误转成可读返回值，模型能读到后自行纠正格式。
        logger.error("numexpr 求值失败：expression=%r, error=%s", expr, exc)
        return f"Calculation error: {exc}"
    logger.info("计算结果：%s", result)
    return result


# --------------------------------------------------------------------------- #
# 3. 创建 Agent
# --------------------------------------------------------------------------- #
# create_agent 会自动搭好一个循环：
#   模型思考 → 若发出 tool_call 则执行工具 → 把结果回灌给模型 → 直到模型给出最终答案
#
# system_prompt 里明确要求「先转成表达式再调计算器」，
# 这是让 Agent 稳定走工具路线的关键。
math_agent = create_agent(
    model=llm,
    tools=[calculator],
    system_prompt=(
        "You are a mathematical assistant that helps solve math problems. "
        "For any question that requires arithmetic, first work out a single-line "
        "Python expression that solves it, then call the `calculator` tool with "
        "that expression to obtain the exact numeric answer. "
        "Always report the final numeric answer."
    ),
)

# --------------------------------------------------------------------------- #
# 4. 调用
# --------------------------------------------------------------------------- #
MATH_QUESTION = "What is 100 times 4?"

# Agent 的返回值是状态字典，最后一条消息（messages[-1]）就是最终答案。
logger.info("开始调用数学 Agent：%s", MATH_QUESTION)
agent_result = math_agent.invoke({"messages": [("user", MATH_QUESTION)]})
final_answer = agent_result["messages"][-1].content
logger.info("Agent 回答完成：%s", final_answer)

print(f"\033[1mQuestion: {MATH_QUESTION}")
print("-" * 50)
print(f"\033[0mAnswer: {final_answer}")

# --------------------------------------------------------------------------- #
# 5. 流式输出：实时看到「思考 → 调工具 → 得答案」的全过程
# --------------------------------------------------------------------------- #
# stream_mode="values" 表示每步产出一个完整状态快照；
# 若想逐 token 输出，可改为 stream_mode="messages"。
print("\n\033[1m[流式过程]")
print("-" * 50)
print("\033[0m")
for chunk in math_agent.stream(
    {"messages": [("user", MATH_QUESTION)]},
    stream_mode="values",
):
    last = chunk["messages"][-1]
    print(f"  · {last.type}: {last.content or getattr(last, 'tool_calls', '')}")
    logger.debug("流式节点：type=%s", last.type)

# --------------------------------------------------------------------------- #
# 6. 对照组：纯靠模型心算（不调用任何工具）
# --------------------------------------------------------------------------- #
# 这一步用来对比「有没有工具」的差别：
# 纯靠模型心算，大数幂运算很容易出错或答非所问。
logger.info("开始对照组：不使用任何工具，纯模型作答")
plain = llm.invoke([HumanMessage(content=MATH_QUESTION)])
logger.info("对照组回答完成：%s", plain.content)
print("\n\033[1m[无工具，纯模型作答]")
print("-" * 50)
print("\033[0m" + plain.content)

# --------------------------------------------------------------------------- #
# 7. 常见坑
# --------------------------------------------------------------------------- #
# 7.1 numexpr 只支持数值运算，不支持变量赋值、循环、函数定义；
#     遇到 "let x = ..." 这类写法会报解析错误。
# 7.2 numexpr 对超大整数会返回 float（精度可能损失）；
#     若需精确大整数，工具里应改用 Python 原生 int 运算。
# 7.3 旧版 ``LLMMathChain`` 的输出是 ``{"answer": ...}``；
#     Agent 版没有字典返回值，取 ``result["messages"][-1].content``。
# 7.4 若模型仍然自己心算不调工具，可以收紧 system_prompt，
#     或用 llm.bind_tools(tools, tool_choice="any") 强制走工具。
