import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
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


load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY")
if not api_key:
    raise RuntimeError(
        "未找到 API Key，请在 .env 中设置 DEEPSEEK_API_KEY（或 API_KEY）"
    )
# 再显式写回环境变量，供 init_chat_model 按 provider 自动读取。
os.environ["DEEPSEEK_API_KEY"] = api_key
# 1. 初始化模型 (使用 init_chat_model 替代旧的 OpenAI 类)
# 这里以 OpenAI 为例，你也可以换成 deepseek 等支持的提供商
model = init_chat_model("deepseek:deepseek-chat", temperature=0.8, api_key=api_key)
logger.info("LLM 初始化完成：deepseek:deepseek-chat, temperature=0.8")


# 2. 用 @tool 装饰器定义工具
# 旧版的 llm-math 工具链已被移除，这里直接用 Python 的 eval 实现计算
@tool
def calculator(expression: str) -> str:
    """Useful for when you need to answer questions about math."""
    logger.info("计算器工具被调用：expression=%r", expression)
    try:
        # 注意：生产环境请使用更安全的计算库，此处仅为演示
        # eval 的 globals 里清空 __builtins__，禁止访问任何内置函数，
        # 大幅降低表达式注入的风险（但仍不是安全的沙箱）。
        result = str(eval(expression, {"__builtins__": {}}, {}))
        logger.info("计算器返回结果：%s", result)
        return result
    except Exception as e:
        # 工具内部不抛异常，而是把错误转成可读的返回值，
        # 这样模型能读到错误信息并自行纠正，而不是让整个 Agent 崩掉
        logger.error("计算失败：expression=%r, error=%s", expression, e)
        return f"Calculation error: {e}"


@tool
def language_model(query: str) -> str:
    """Use this tool for general purpose queries and logic."""
    logger.info("通用问答工具被调用：query=%r", query)
    response = model.invoke(query)
    logger.info("通用问答工具返回，长度=%d 字符", len(response.content))
    return response.content


# 3. 创建 Agent
# 旧版的 max_iterations 可以通过 middleware 或 agent 的递归限制来控制
# 这里我们配置 ModelRetryMiddleware 来处理模型调用的重试
agent = create_agent(
    model=model,
    tools=[calculator, language_model],
    system_prompt="You are a helpful assistant that solves math problems and answers questions.",
    middleware=[
        # 配置模型调用的最大重试次数为 3
        ModelRetryMiddleware(max_retries=3),
    ],
    debug=True,  # 开启调试，方便查看 Agent 的思考过程
)
logger.info("Agent 创建完成，已注册工具：calculator, language_model；重试次数上限=3")

# 4. 调用 Agent
# 注意：输入格式变为 {"messages": [...]}
print("--- 测试数学计算 ---")
logger.info("测试 1/2：数学计算")
result = agent.invoke({"messages": [HumanMessage(content="what is (4.5*2.1)^2.2?")]})
print(result["messages"][-1].content)
logger.info("测试 1/2 完成，答案长度=%d 字符", len(result["messages"][-1].content))

print("\n--- 测试语言模型工具 ---")
logger.info("测试 2/2：通用问答")
result = agent.invoke(
    {
        "messages": [
            HumanMessage(content="what is the capital of the United States of America?")
        ]
    }
)
print(result["messages"][-1].content)
logger.info("测试 2/2 完成，答案长度=%d 字符", len(result["messages"][-1].content))
logger.info("Agent 工具调用演示结束")
