"""Prompt 工程演示 1：基础提示词（Basic Prompt）。

核心概念：
    提示词（Prompt）就是发给大模型的输入文本。这里用 LangChain 的
    ``ChatPromptTemplate`` 把提示词模板化：用 ``{customer_message}`` 占位符标记变量位置，
    运行时通过 ``.format(...)`` 或 ``.invoke(...)`` 填充实际内容。

    使用模板的好处：
      - 提示词与业务代码分离，改提示词不用改代码逻辑；
      - 变量可以被安全转义（模板会做 f-string 式的替换）；
      - 便于复用与 A/B 测试（同一模板换不同变量对比效果）。

关于 Agent 的返回结构：
    ``agent.invoke(...)`` 返回一个字典（LangGraph 的状态对象），本例中包含：
      - "messages": 消息列表，依次为 [HumanMessage(用户输入), AIMessage(模型回答)]
      - 可能还有 "structured_response" / "remaining_steps" 等键（取决于是否启用工具/结构化输出）

    所以 ``result["messages"][0]`` 是**用户消息**（即填充后的提示词本身），
    模型真正的回答在 ``result["messages"][-1]``。

本文件演示如何把模型返回的**完整信息**打印出来：
    - 每条消息的类型（HumanMessage / AIMessage / SystemMessage ...）
    - content：正文内容
    - usage_metadata：token 用量统计（input / output / total tokens）
    - response_metadata：厂商原始返回（模型名、finish_reason、system_fingerprint 等）
    - additional_kwargs：额外字段（如 DeepSeek 的 reasoning_content 思维链、refusal）
    - 以及完整的 result 字典本身
"""

import json
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    messages_to_dict,
)
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek

# 加载 .env 中的 API Key，避免硬编码密钥。
load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

# 初始化 DeepSeek 聊天模型：
#   model：模型名，deepseek-flash 速度快、适合演示；
#   temperature：0.5 控制随机性，0 最确定，1 以上越发散；
#   api_key：从环境变量显式传入。
llm = ChatDeepSeek(model="deepseek-flash", temperature=0.5, api_key=api_key)

# --------------------------------------------------------------------------- #
# 1. 定义提示词模板
# --------------------------------------------------------------------------- #
# {customer_message} 是变量占位符，模板化后可复用。
# 实际项目中通常还会加上角色设定、输出格式约束、few-shot 示例等。
basic_template = """
    Reply to this customer message: {customer_message}
"""
basic_prompt = PromptTemplate(
    input_variables=["customer_message"], template=basic_template
)


# --------------------------------------------------------------------------- #
# 2. 准备输入并渲染系统提示词
# --------------------------------------------------------------------------- #
customer_message = """
I've been waiting for 2 hours for your delivery and it still hasn't arrived!
This is absolutely unacceptable! I paid extra for express delivery and this
is what I get? I want my money back immediately!
"""

# PromptTemplate.format(...) 把占位符替换成实际内容，返回的是**纯字符串**。
# 注意：create_agent 的 system_prompt 只接受 str | SystemMessage，
# 不能直接传 PromptTemplate 对象，否则会报 pydantic 校验错误。
system_prompt = basic_prompt.format(customer_message=customer_message)


# --------------------------------------------------------------------------- #
# 3. 创建 Agent
# --------------------------------------------------------------------------- #
# create_agent 会返回一个 LangGraph 编译好的状态图。
# 这里没有传 tools，所以它就是一个"单轮问答"的 Agent：
# 输入消息 -> 模型 -> 直接返回 AIMessage。
agent = create_agent(model=llm, system_prompt=system_prompt)


# --------------------------------------------------------------------------- #
# 4. 调用
# --------------------------------------------------------------------------- #
result = agent.invoke(
    {"messages": [HumanMessage(content="Please respond to my message.")]}
)
# --------------------------------------------------------------------------- #
# 4. 打印完整信息
# --------------------------------------------------------------------------- #
print("=" * 70)
print("1) 完整 result 字典（原始状态对象）")
print("=" * 70)
# result 里的 messages 是 BaseMessage 对象列表，不能直接被 json 序列化。
# 用 messages_to_dict 把它转成标准字典结构（LangChain 的消息 JSON 规范），
# 这样输出可读、可直接存日志或发到前端。
print(
    json.dumps(
        {
            **{k: v for k, v in result.items() if k != "messages"},
            "messages": messages_to_dict(result["messages"]),
        },
        indent=2,
        ensure_ascii=False,
        default=str,
    )
)

print("\n" + "=" * 70)
print("2) 逐条消息详情")
print("=" * 70)

for i, message in enumerate(result["messages"]):
    # isinstance 判断消息类型，便于区分用户消息与模型消息
    if isinstance(message, AIMessage):
        role = "AI（模型回答）"
    elif isinstance(message, HumanMessage):
        role = "Human（用户输入）"
    else:
        role = type(message).__name__

    print(f"\n--- Message[{i}] | 类型: {type(message).__name__} | 角色: {role} ---")

    # content：正文内容。可能是字符串，也可能是 content blocks 列表。
    print(f"content:\n{message.content}")

    # response_metadata：厂商返回的原始元数据
    # 包含 model_name、finish_reason、token_usage、system_fingerprint、id 等。
    response_metadata = getattr(message, "response_metadata", None)
    print(
        f"\nresponse_metadata:\n{json.dumps(response_metadata, indent=2, ensure_ascii=False, default=str)}"
    )

    # usage_metadata：标准化的 token 用量统计
    # input_tokens（输入消耗）、output_tokens（输出消耗）、total_tokens（合计），
    # 以及细分的 reasoning tokens（推理模型思考过程消耗的 token）。
    usage_metadata = getattr(message, "usage_metadata", None)
    print(
        f"\nusage_metadata:\n{json.dumps(usage_metadata, indent=2, ensure_ascii=False, default=str)}"
    )

    # additional_kwargs：额外字段，不同厂商内容不同。
    # DeepSeek 会在这里返回 reasoning_content（思维链）和 refusal（拒答标记）。
    additional_kwargs = getattr(message, "additional_kwargs", None)
    print(
        f"\nadditional_kwargs:\n{json.dumps(additional_kwargs, indent=2, ensure_ascii=False, default=str)}"
    )

    # 消息 ID 与类型标记，便于做链路追踪
    print(f"\nid: {getattr(message, 'id', None)}")
    print(f"type: {getattr(message, 'type', None)}")

print("\n" + "=" * 70)
print("3) 快速取用：模型最终回答")
print("=" * 70)
# messages[0] 是用户消息（提示词本身），模型真正的回答在最后一条。
print(result["messages"][-1].content)

# --------------------------------------------------------------------------- #
# 附：其他常用取数方式
# --------------------------------------------------------------------------- #
#   result["messages"][0].content   -> 用户消息（填充后的提示词）
#   result["messages"][-1].content  -> 模型最终回答
#   result["messages"][-1].usage_metadata["total_tokens"] -> 本次调用的总 token 数
#   result["messages"][-1].response_metadata["model_name"]  -> 实际使用的模型名
#
# 需要流式输出时改用 agent.stream(..., stream_mode="values") 逐节点获取。
