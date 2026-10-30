import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

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


# 2. 用 @tool 装饰器定义工具
# 旧版的 llm-math 工具链已被移除，这里直接用 Python 的 eval 实现计算
@tool
def calculator(expression: str) -> str:
    """Useful for when you need to answer questions about math."""
    try:
        # 注意：生产环境请使用更安全的计算库，此处仅为演示
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def language_model(query: str) -> str:
    """Use this tool for general purpose queries and logic."""
    response = model.invoke(query)
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

# 4. 调用 Agent
# 注意：输入格式变为 {"messages": [...]}
print("--- 测试数学计算 ---")
result = agent.invoke({"messages": [HumanMessage(content="what is (4.5*2.1)^2.2?")]})
print(result["messages"][-1].content)

print("\n--- 测试语言模型工具 ---")
result = agent.invoke(
    {
        "messages": [
            HumanMessage(content="what is the capital of the United States of America?")
        ]
    }
)
print(result["messages"][-1].content)
