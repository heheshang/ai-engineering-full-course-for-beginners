"""链（Chain）演示 6-2：数学链（旧版 LLMMathChain → LangChain 1.x 现代写法）。

对应「课程旧写法」：

    # pip install langchain-openai
    # pip install numexpr
    from langchain.chains import LLMMathChain          # ❌ 1.x 中已移除
    math_chain = LLMMathChain.from_llm(llm=llm, prompt=math_prompt)
    math_result = math_chain.invoke({"question": math_question})
    print(math_result["answer"])                        # ❌ 旧返回值字典

在 LangChain 1.x 里 ``langchain.chains`` 已被移除，旧实现搬到了
``langchain_classic`` 且带 deprecation 警告：

    @deprecated(since="0.2.13", removal="2.0.0",
                alternative="langchain.agents.create_agent")

**本文件用官方推荐的 ``create_agent`` + ``@tool`` 计算器工具重写同样的效果：**

    旧链路                                   新链路
    ────────────────────────────            ────────────────────────────
    prompt 要求输出 "Answer: 100 * 4"        prompt 要求先转表达式再调工具
    → 正则/代码块解析 → 求值                  → function calling: calculator("100 * 4")
    → 直接返回字符串答案                      → 工具返回数值 → 模型复述最终答案

好处：不再依赖脆弱的代码块解析，天然支持流式、批量与多轮记忆。

依赖：pip install langchain langchain-openai python-dotenv numexpr
"""

import math
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

# --------------------------------------------------------------------------- #
# 0. 加载环境变量
# --------------------------------------------------------------------------- #
load_dotenv()

# 优先 OPENAI_API_KEY，其次兼容通用命名 API_KEY；都没有就立刻报错。
api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY")
if not api_key:
    raise RuntimeError("未找到 API Key，请在 .env 中设置 OPENAI_API_KEY（或 API_KEY）")
os.environ["DEEPSEEK_API_KEY"] = api_key    

# --------------------------------------------------------------------------- #
# 1. 初始化 LLM
# --------------------------------------------------------------------------- #
# init_chat_model 是 1.x 的统一模型工厂函数（provider + model）。
# 也可以直接用 ChatOpenAI(model="gpt-4o-mini", temperature=0)。
# model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
llm = init_chat_model(model="deepseek-chat", temperature=0)

# llm = init_chat_model(model=model_name, api_key=api_key, temperature=0)

# --------------------------------------------------------------------------- #
# 2. 定义计算器工具
# --------------------------------------------------------------------------- #
# 用 numexpr 做安全求值：它只解析数学表达式，无法执行任意 Python 代码
# （没有 import / 函数定义 / 属性访问），比 eval 安全得多。
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
        "2 + 2" for "2 plus 2"
        "25 * 4" for "25 times 4"
    """
    # 容错：模型偶尔会带 "Answer: " 前缀或 ``` 包裹，统一清洗。
    expr = expression.strip().strip("`")
    if expr.lower().startswith("answer:"):
        expr = expr[len("answer:") :].strip()

    if not _HAS_NUMEXPR:
        return str(eval(expr, {"__builtins__": {}}, {}))
    return str(
        numexpr.evaluate(
            expr,
            global_dict={},
            local_dict={"pi": math.pi, "e": math.e},
        )
    )


# --------------------------------------------------------------------------- #
# 3. 创建 Agent（替代旧的 LLMMathChain）
# --------------------------------------------------------------------------- #
# system_prompt 明确要求「先转成一行表达式再调计算器」，
# 这是让 Agent 稳定走工具路线的关键。
math_agent = create_agent(
    model=llm,
    tools=[calculator],
    system_prompt=(
        "You are a mathematical assistant that helps solve math problems. "
        "Given a math problem, work out a single-line Python expression that "
        "solves it, then call the `calculator` tool with that expression to "
        "obtain the exact numeric answer. Do not do arithmetic yourself. "
        "Always report the final numeric answer."
    ),
)

# --------------------------------------------------------------------------- #
# 4. 调用
# --------------------------------------------------------------------------- #
# 旧版：math_result = math_chain.invoke({"question": math_question})
#       math_result["answer"]
# 1.x Agent：返回状态字典，最后一条消息即最终答案。
MATH_QUESTION = "What is 100 times 4?"
agent_result = math_agent.invoke({"messages": [("user", MATH_QUESTION)]})

print(f"\033[1mQuestion: {MATH_QUESTION}")
print("-" * 50)
print(f"\033[0mAnswer: {agent_result['messages'][-1].content}")

# --------------------------------------------------------------------------- #
# 5. 流式输出：观察 思考 → 调工具 → 得答案 的全过程
# --------------------------------------------------------------------------- #
print("\n\033[1m[流式过程]")
print("-" * 50)
print("\033[0m")
for chunk in math_agent.stream(
    {"messages": [("user", MATH_QUESTION)]},
    stream_mode="values",
):
    last = chunk["messages"][-1]
    print(f"  · {last.type}: {last.content or getattr(last, 'tool_calls', '')}")

# --------------------------------------------------------------------------- #
# 6. 常见坑
# --------------------------------------------------------------------------- #
# 6.1 旧版 LLMMathChain 输出是 ``{"answer": ...}`` 字典；
#     Agent 版没有字典返回值，取 ``result["messages"][-1].content``。
# 6.2 numexpr 只支持数值运算，不支持变量赋值、循环、函数定义。
# 6.3 numexpr 对超大整数会返回 float（精度可能损失），
#     若需精确大整数，工具里应改用 Python 原生 int 运算。
# 6.4 若模型不调工具就自己心算，可收紧 system_prompt，
#     或用 llm.bind_tools([calculator], tool_choice="any") 强制走工具。
