"""Prompt 工程演示 2：思维链提示（Chain-of-Thought Prompt）。

核心概念：
    思维链（Chain of Thought, CoT） prompting 让模型先"想"再"答"，
    在最终回答前显式给出推理步骤，通常能显著提升复杂任务（推理、规划、
    多约束满足）的准确率。

关于 LangChain 1.x 的 API 变化（重要）：
    ``create_agent()`` 的签名里**没有** ``prompt`` 参数，只有
    ``system_prompt: str | SystemMessage | None``。
    也就是说 Agent 的系统提示词只能是**字符串**，不能直接传
    ``PromptTemplate`` / ``ChatPromptTemplate`` 对象。

    因此正确做法是：
      1. 用 PromptTemplate（或 f-string）把变量填进模板，得到最终字符串；
      2. 把这个字符串传给 ``create_agent(model=..., system_prompt=...)``；
      3. 用户输入依然通过 ``{"messages": [HumanMessage(...)]}`` 传入。

    （若不传 system_prompt，Agent 就等价于"没有系统提示的单轮问答"。）

关于返回结构：
    ``agent.invoke(...)`` 返回 LangGraph 状态字典，``result["messages"][-1]``
    才是模型的最终回答（``result["messages"][0]`` 是用户输入）。
"""

import os
import sys
from pathlib import Path

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek

from utils.logging_utils import get_logger, log_exception

# 初始化 logger：日志同时输出到控制台和 logs/course.log
logger = get_logger(__name__)

# 加载 .env 中的 API Key，避免硬编码密钥。
load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

# 初始化 DeepSeek 聊天模型：
#   model：模型名，deepseek-flash 速度快、适合演示；
#   temperature：0.5 控制随机性，0 最确定，1 以上越发散；
#   api_key：从环境变量显式传入。
llm = ChatDeepSeek(model="deepseek-flash", temperature=0.5, api_key=api_key)
logger.info("LLM 初始化完成：model=deepseek-flash, temperature=0.5")

# --------------------------------------------------------------------------- #
# 1. 定义思维链提示词模板
# --------------------------------------------------------------------------- #
# {customer_message} 是变量占位符，模板化后可在运行时填充不同内容。
chain_of_thought_template = """
You are a highly professional customer support specialist at a premium tech company.
Your goal is to address the customer's concerns while maintaining a calm and helpful demeanor.

Let's approach this step by step:

1. First, analyze the customer's sentiment and identify the core issues in their message
2. Then, formulate an empathetic acknowledgment of their feelings
3. Next, address each identified issue with specific solutions
4. Finally, end with a positive and forward-looking statement

Here are some examples of good responses:
Angry Customer: "Your product is terrible! I've been trying to make it work for hours and nothing helps!"
Thought Process:
1. Sentiment: Customer is frustrated due to wasted time and product difficulties
2. Core issues: Product not working, time wasted, lack of proper guidance
3. Required: Immediate assistance, clear steps, acknowledgment of time value
Response: "I sincerely apologize for the frustration you're experiencing. I understand how valuable your time is, and it's unacceptable that you've spent hours trying to resolve this. Let's work together to fix this immediately. Could you please tell me what specific issues you're encountering? I'll guide you through the solution step by step."

Now, please analyze and respond to this customer message: {customer_message}
Please show your thought process before providing the response.
"""

chain_of_thought_prompt = PromptTemplate(
    input_variables=["customer_message"], template=chain_of_thought_template
)

customer_message = """
I've been waiting for 2 hours for your delivery and it still hasn't arrived!
This is absolutely unacceptable! I paid extra for express delivery and this
is what I get? I want my money back immediately!
"""

# --------------------------------------------------------------------------- #
# 2. 渲染模板 -> 系统提示词（必须是 str）
# --------------------------------------------------------------------------- #
# PromptTemplate.format(...) 把占位符替换成实际内容，返回纯字符串。
system_prompt = chain_of_thought_prompt.format(customer_message=customer_message)

# --------------------------------------------------------------------------- #
# 3. 创建 Agent
# --------------------------------------------------------------------------- #
# 注意参数名是 system_prompt（字符串），不是 prompt（模板对象）。
agent = create_agent(model=llm, system_prompt=system_prompt)
logger.info("Agent 创建完成，系统提示词长度=%d 字符", len(system_prompt))

# --------------------------------------------------------------------------- #
# 4. 调用
# --------------------------------------------------------------------------- #
logger.info("开始调用 Agent（思维链提示）")
try:
    result = agent.invoke({"messages": [HumanMessage(content="Please respond to my message.")]})
except Exception:
    log_exception(logger, "Agent 调用失败")
    raise

# messages[0] 是用户输入，模型真正的回答在最后一条。
answer = result["messages"][-1].content
logger.info("回答完成，长度=%d 字符", len(answer))

# 日志：记录 token 用量，便于评估思维链带来的额外成本
usage = getattr(result["messages"][-1], "usage_metadata", None)
if usage:
    logger.info(
        "token 用量：输入=%s，输出=%s，合计=%s",
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        usage.get("total_tokens"),
    )

print(result["messages"][-1].content)
