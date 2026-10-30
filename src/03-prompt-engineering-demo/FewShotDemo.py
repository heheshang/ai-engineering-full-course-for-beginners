"""Prompt 工程演示 4：少样本提示（Few-Shot Prompting）。

核心概念：
    少样本提示（Few-Shot Prompting）是指在提示词里**先给几个"输入 -> 输出"的示例**，
    再给出真实的待处理输入，让模型**模仿示例的风格、结构和用词**完成新任务。

    直觉解释：
        人类遇到没见过的任务时，看几个范例比读十行抽象描述学得快得多。
        大模型同理 —— 示例相当于"隐式的任务说明书"，比纯文字描述更精确。

    少样本提示的三要素：
      1. 任务描述（Instruction）：告诉模型要做什么；
      2. 示例（Examples）：若干组 "输入 -> 理想输出"；
      3. 真实输入（Query）：最终要处理的那一条。

    本例的示例刻意做成"**风格示例**"而不是"知识问答"：
    两个 Angry Customer -> Response 的对照，
    目的是让模型学会那种"先共情、再给方案、最后反问推进"的客服话术，
    而不是学会某个具体问题的答案。

    示例设计的实践要点：
      - 示例要**同分布**：示例的输入要和真实输入相似（都是愤怒的差评/投诉）；
      - 示例要**覆盖多种情况**：两个示例分别体现"产品不好用"和"要退款"，
        避免模型只会一种句式；
      - 示例要有**代表性**：如果示例里全是短句，模型也会写短句；
      - 示例数量一般 2~5 个就够，太多会显著增加 token 成本且收益递减。

    少样本 vs 其他技巧：
      - vs 零样本（Zero-shot）：只给指令，不给示例。少样本通常准确率更高；
      - vs 微调（Fine-tuning）：少样本不改模型权重、只改输入；
        微调要数据、要训练、成本高。
        数据少、任务变化快时，优先用少样本。

关于 LangChain 1.x 的 API 约束（重要）：
    ``create_agent()`` 的系统提示词参数叫 ``system_prompt``，
    类型是 ``str | SystemMessage | None``，
    **不接受** ``PromptTemplate`` / ``ChatPromptTemplate`` 对象。
    所以必须先用模板把变量填好，再把**渲染后的字符串**（或包成 SystemMessage）传进去。

关于返回结构：
    ``agent.invoke(...)`` 返回 LangGraph 的状态字典（dict），其中：
      - ``result["messages"]`` 是消息列表，依次为
        [SystemMessage(系统提示词), HumanMessage(用户输入), AIMessage(模型回答)]
      - 可能还有 ``structured_response`` / ``remaining_steps`` 等键
        （取决于是否启用工具调用或结构化输出）

    因此：
      - ``result["messages"][0]``  -> 系统提示词（SystemMessage）
      - ``result["messages"][1]``  -> 用户输入（HumanMessage）
      - ``result["messages"][-1]`` -> **模型真正的最终回答**（AIMessage）
"""

import json
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    messages_to_dict,
)
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek

# --------------------------------------------------------------------------- #
# 0. 准备工作
# --------------------------------------------------------------------------- #
# 加载 .env 中的环境变量，避免把 API Key 硬编码进源码。
# load_dotenv() 会自动在当前目录及父目录查找 .env 文件。
load_dotenv()

# 从环境变量读取 DeepSeek 的 API Key。
api_key = os.getenv("DEEPSEEK_API_KEY")

# 初始化 DeepSeek 聊天模型：
#   model：模型名。deepseek-flash 速度快、便宜，适合演示与批量实验；
#   temperature：0.5 控制随机性。
#     少样本提示希望模型"贴着示例写"而不是自由发挥，
#     所以温度不宜太高，0.2~0.5 比较合适。
#   api_key：从环境变量显式传入。
llm = ChatDeepSeek(model="deepseek-flash", temperature=0.5, api_key=api_key)

# --------------------------------------------------------------------------- #
# 1. 定义少样本提示词模板
# --------------------------------------------------------------------------- #
# 模板结构（三段式）：
#   [1] Instruction：角色 + 任务
#   [2] Examples   ：若干组 "Angry Customer: ... / Response: ..."
#   [3] Query      ：真实待处理的输入 {customer_message}
#
# 注意这里的示例是"对话风格示例"：
# 模型会模仿示例里的**结构**（先共情 -> 再给方案 -> 最后反问）
# 和**语气**（专业、克制、有同理心），而不是照抄示例里的具体内容。
few_shot_template = """
You are a highly professional customer support specialist at a premium tech company.
Your goal is to address the customer's concerns while maintaining a calm and helpful demeanor.

Here are some examples of good responses to angry customers:

Angry Customer: "Your product is terrible! I've been trying to make it work for hours and nothing helps!"
Response: "I sincerely apologize for the frustration you're experiencing. I understand how valuable your time is, and it's unacceptable that you've spent hours trying to resolve this. Let's work together to fix this immediately. Could you please tell me what specific issues you're encountering? I'll guide you through the solution step by step."

Angry Customer: "I want a refund right now! This is the worst service ever!"
Response: "I completely understand your disappointment, and I want to make this right for you. I'll help you with the refund process right away. While I process this, could you share what specific aspects of our service didn't meet your expectations? This will help us improve and prevent similar issues in the future."

Now, please respond to this customer message: {customer_message}
"""

# PromptTemplate 会解析模板里的 {customer_message}，
# 声明 input_variables 后，运行时可以安全地传入对应关键字参数。
few_shot_prompt = PromptTemplate(
    input_variables=["customer_message"], template=few_shot_template
)

# --------------------------------------------------------------------------- #
# 2. 准备真实输入（待处理的客户消息）
# --------------------------------------------------------------------------- #
# 注意：真实输入必须和示例**同分布** —— 都是愤怒的投诉类消息。
# 如果真实输入是"如何配置 XX 功能"这种咨询类问题，
# 用愤怒投诉的示例就会导致模型答非所问。
customer_message = """
I've been waiting for 2 hours for your delivery and it still hasn't arrived!
This is absolutely unacceptable! I paid extra for express delivery and this
is what I get? I want my money back immediately!
"""

# --------------------------------------------------------------------------- #
# 3. 创建 Agent
# --------------------------------------------------------------------------- #
# 关键点：system_prompt 必须是 str 或 SystemMessage，不能直接传 PromptTemplate。
# 所以这里先用 .format(...) 把占位符渲染成完整字符串，
# 再用 SystemMessage 包一层，明确"这是系统规则 + 示例"而不是"用户说的话"。
#
# 小技巧：如果示例里含有本该属于用户的输入（像本例的 "Angry Customer: ..."），
# 更严谨的做法是把示例做成真正的多轮消息历史
# （[HumanMessage, AIMessage, HumanMessage, AIMessage, ...]），
# 这样模型能清晰区分"示例"和"这次真正的提问"。
# 这里为了保持演示简洁，仍然采用单字符串模板的写法。
few_shot_agent = create_agent(
    model=llm,
    system_prompt=SystemMessage(
        content=few_shot_prompt.format(customer_message=customer_message)
    ),
)

# --------------------------------------------------------------------------- #
# 4. 调用 Agent
# --------------------------------------------------------------------------- #
# 输入必须是 {"messages": [...]} 形式的消息列表。
# 注意区分两类内容：
#   - system_prompt（角色 + 示例 + 真实输入）-> 走 system 位置
#   - HumanMessage                         -> 走 user 位置，触发模型作答
result = few_shot_agent.invoke(
    {"messages": [HumanMessage(content="Please respond to my message.")]}
)


# --------------------------------------------------------------------------- #
# 5. 打印完整信息
# --------------------------------------------------------------------------- #
print("=" * 70)
print("1) 完整 result 字典（原始状态对象）")
print("=" * 70)
# result 里的 messages 是 BaseMessage 对象列表，不能直接被 json 序列化。
# 用 messages_to_dict 把它转成标准字典结构（LangChain 的消息 JSON 规范），
# 这样输出可读、可直接存日志或发到前端。
# ensure_ascii=False 保留中文原样；default=str 兜底处理无法 JSON 序列化的对象。
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
    # isinstance 判断消息类型，便于区分系统提示 / 用户输入 / 模型回答
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
# messages[0] 是系统提示词，messages[1] 是用户输入，
# 模型真正的回答在最后一条。
answer = result["messages"][-1].content
print(answer)

# --------------------------------------------------------------------------- #
# 附：少样本效果的自检（验证模型是否"学会了示例的风格"）
# --------------------------------------------------------------------------- #
# 少样本提示的效果很难用"对不对"衡量，更适合用**风格特征**做启发式检查：
# 模型是否沿用了示例里的关键结构（先共情、再给方案、最后以问句推进）。
#
# 下面这些规则就是从本例两个示例中提炼出来的"风格指纹"，
# 在生产中可以做成回归测试：换模型、换提示词后跑一遍，看风格有没有漂移。
print("\n" + "=" * 70)
print("4) 风格指纹自检（模型是否模仿了示例的结构）")
print("=" * 70)

sentences = [s.strip() for s in answer.replace("\n", " ").split(".") if s.strip()]
has_question = answer.rstrip().endswith("?")

checks = {
    "包含共情表达（understand / sorry / apologize）": any(
        kw in answer.lower() for kw in ("understand", "sorry", "apolog")
    ),
    "给出了可执行方案（action 动词）": any(
        kw in answer.lower()
        for kw in ("refund", "check", "arrange", "process", "send", "track", "escalate")
    ),
    "以问句结尾（与示例一致）": has_question,
    "长度与示例相当（不显著偏离）": 80 <= len(answer) <= 800,
    "句数不少于 3 句": len(sentences) >= 3,
}

for name, ok in checks.items():
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")

# --------------------------------------------------------------------------- #
# 附：其他常用取数方式
# --------------------------------------------------------------------------- #
#   result["messages"][0].content   -> 系统提示词（渲染后的示例 + 真实输入）
#   result["messages"][1].content   -> 用户输入
#   result["messages"][-1].content  -> 模型最终回答
#   result["messages"][-1].usage_metadata["total_tokens"] -> 本次调用的总 token 数
#   result["messages"][-1].response_metadata["model_name"]  -> 实际使用的模型名
#
# 需要流式输出时改用 agent.stream(..., stream_mode="values") 逐节点获取。
