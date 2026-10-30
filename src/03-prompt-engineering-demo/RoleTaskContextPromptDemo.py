"""Prompt 工程演示 5：角色-任务-上下文（Role / Task / Context）结构化提示词。

核心概念：
    RTCF（Role-Task-Context-Format）是提示词工程里最经典的**结构化拆解框架**，
    把一整段"人话"提示词拆成四个明确区块，让模型（或同事）一眼看清每个部分的作用：

      - **Role（角色）**：你是谁？专业身份、领域、资历、说话风格。
        作用：把模型的输出锁定在某个知识域和语气上。
        例："You are a highly professional customer support specialist at a premium tech company."

      - **Task（任务）**：要做什么？动词开头，说明要产出的东西。
        作用：消除歧义。"Summarize" 和 "Reply to" 会导致完全不同的输出。

      - **Context（上下文）**：任务的背景信息 —— 客户是谁、发生了什么、
        有哪些限制条件、历史对话、业务规则。
        作用：给模型判断依据。没有 Context，模型只能靠"猜"。

      - **Format（格式）**：输出长什么样？结构、字段、长度、语气。
        作用：让结果可被下游程序消费（可解析、可校验）。

    为什么值得这么写？
      - 对模型：每个区块职责单一，注意力更容易聚焦，减少指令冲突；
      - 对人：提示词即文档，便于 review、版本管理、多人协作；
      - 对工程：Format 区块写出的规则可以直接变成自动化验收脚本。

    本例还演示了一个关键取舍 —— **上下文该放 system 还是 user**：
      角色 / 任务 / 格式属于"永久规则"，应该放 system；
      客户这次说的话属于"本轮输入"，应该放 user。
      如果把两者都塞进 system（常见的初学者写法），会造成：
        1) 同一段客户消息被重复传输，白白浪费 token；
        2) 模型难以区分"规则"和"用户当下的诉求"，容易把诉求当成示例来模仿；
        3) system 越来越长，真正的规则被淹没。

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

import sys
from pathlib import Path

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.logging_utils import get_logger, log_exception

# 初始化 logger：日志同时输出到控制台和 logs/course.log
logger = get_logger(__name__)


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
#   temperature：0.5 控制随机性。0 最确定、1 以上越发散。
#   api_key：从环境变量显式传入。
llm = ChatDeepSeek(model="deepseek-flash", temperature=0.5, api_key=api_key)
logger.info("LLM 初始化完成：model=deepseek-flash, temperature=0.5")

# --------------------------------------------------------------------------- #
# 1. 定义提示词模板（按 RTCF 四区块组织）
# --------------------------------------------------------------------------- #
# 用 ChatPromptTemplate.from_messages 也能做，但这里用 PromptTemplate +
# SystemMessage 的组合更直观：模板只负责"填变量"，消息类型由调用处显式决定。
#
# 注意模板里**只有** {customer_message} 一个变量，
# 它出现在 Context 区块 —— 但我们会把渲染结果用在 system prompt 里，
# 同时把原始客户消息作为 HumanMessage 传入（见第 4 节的说明）。
role_context_template = """
[ROLE]
You are a highly professional customer support specialist at a premium tech company.
Your goal is to address the customer's concerns while maintaining a calm and helpful demeanor.

[TASK]
Write a reply to the customer message provided below. Respond to the customer directly,
do not describe or evaluate the message itself.

[CONTEXT]
Customer Message: {customer_message}

[FORMAT]
The response must:
1. Acknowledge the customer's feelings
2. Address their concerns professionally
3. Offer concrete solutions
4. End with a positive note
"""

# PromptTemplate 会解析模板里的 {customer_message}，
# 声明 input_variables 后，运行时可以安全地传入对应关键字参数。
role_context_prompt = PromptTemplate(
    input_variables=["customer_message"], template=role_context_template
)

# --------------------------------------------------------------------------- #
# 2. 准备本轮输入
# --------------------------------------------------------------------------- #
customer_message = """
I've been waiting for 2 hours for your delivery and it still hasn't arrived!
This is absolutely unacceptable! I paid extra for express delivery and this
is what I get? I want my money back immediately!
"""

# --------------------------------------------------------------------------- #
# 3. 渲染模板
# --------------------------------------------------------------------------- #
# .format(...) 把占位符替换成实际内容，返回**纯字符串**。
#
# 命名上的讲究：用 system_prompt 承接渲染结果，
# 不要复用 role_context_prompt 这个变量名去存字符串 ——
# 同名变量一会儿是模板、一会儿是字符串，是很常见的 bug 来源。
system_prompt = role_context_prompt.format(customer_message=customer_message)
logger.info("RTCF 模板渲染完成，系统提示词长度=%d 字符", len(system_prompt))

# --------------------------------------------------------------------------- #
# 4. 创建 Agent 并调用
# --------------------------------------------------------------------------- #
# 关键点：system_prompt 必须是 str 或 SystemMessage，不能直接传 PromptTemplate。
# 这里用 SystemMessage 包一层，明确告诉模型"这是系统规则"，约束力更强。
role_context_agent = create_agent(
    model=llm,
    system_prompt=SystemMessage(content=system_prompt),
)
logger.info("Agent 创建完成（RTCF 四区块结构）")

# 说明：客户消息**既**出现在 system prompt 的 Context 区块，
# **又**作为 HumanMessage 传入。这样看起来重复，但各有分工：
#   - Context 区块：告诉模型"这条消息属于哪个场景"，帮助它理解语境；
#   - HumanMessage：告诉模型"现在请回答这条消息"，明确本轮的触发对象。
# 如果追求 token 极致，可从模板中删掉 Context 区块的 {customer_message}，
# 只保留 HumanMessage 里的那一份；两种写法都能跑通，值得对比一下效果。
logger.info("开始调用 Agent（RTCF 结构化提示）")
try:
    result = role_context_agent.invoke(
        {"messages": [HumanMessage(content=customer_message)]}
    )
except Exception as exc:  # noqa: BLE001 - 演示脚本，捕获后打印完整堆栈便于排查
    log_exception(logger, "Agent 调用失败")
    raise

logger.info("Agent 调用完成，返回 %d 条消息", len(result["messages"]))
usage = getattr(result["messages"][-1], "usage_metadata", None)
if usage:
    logger.info(
        "token 用量：输入=%s，输出=%s，合计=%s",
        usage.get("input_tokens"), usage.get("output_tokens"), usage.get("total_tokens"),
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
# 6. 格式规则自检（把 [FORMAT] 区块变成可执行断言）
# --------------------------------------------------------------------------- #
# [FORMAT] 里写的每一条规则都应该是**可机械验证**的，
# 否则它只是"愿望"而不是"约束"。下面这段就是对应的验收脚本，
# 在生产中可以直接作为回归测试复用。
print("\n" + "=" * 70)
print("4) FORMAT 规则自动校验")
print("=" * 70)

# 按空行切分段落，过滤掉空白段
paragraphs = [p for p in answer.split("\n\n") if p.strip()]
# 按句号/问号/感叹号切句，得到句子列表
sentences = [
    s.strip()
    for s in answer.replace("\n", " ").replace("!", ".").replace("?", ".").split(".")
    if s.strip()
]
lower_answer = answer.lower()
# 取最后一段的结尾部分做语气检查（先判空，避免 IndexError）
tail = paragraphs[-1].lower()[-80:] if paragraphs else ""

# 可执行动作动词表：用来判断模型是否真的给出了"能做的事"，
# 而不是只表达情绪。相比数数字/数编号，这种判断更贴近"是否提供了方案"的语义。
ACTION_VERBS = (
    "refund",
    "check",
    "arrange",
    "process",
    "send",
    "track",
    "escalate",
    "cancel",
    "contact",
    "update",
)

checks = {
    "1) 回应了客户的感受（致歉/共情）": any(
        kw in lower_answer
        for kw in ("sorry", "apolog", "understand", "frustrat", "unacceptable")
    ),
    "2) 专业地处理了诉求（给出动作/方案）": any(
        kw in lower_answer
        for kw in ("refund", "check", "arrange", "process", "send", "track", "escalate")
    ),
    "3) 提供了具体解决方案（可执行动作 >= 2 项）": (
        sum(1 for kw in ACTION_VERBS if kw in lower_answer) >= 2
    ),
    "4) 以积极/正面语气收尾（末尾不含负面词）": not any(
        kw in tail for kw in ("unacceptable", "worst", "terrible", "disappointed")
    ),
    "结构完整：不少于 3 个段落": len(paragraphs) >= 3,
    # 粗略的长度合理性检查（上下界放宽，只排除"空回复"和"异常冗长"两种极端情况；
    # 具体长度要求应该写进 [FORMAT] 区块由模型遵守，而不是靠这里猜）
    "长度合理（80~2000 字符）": 80 <= len(answer) <= 2000,
}

# 日志：把 FORMAT 规则自检结果写进日志，便于回归测试直接判定成败
failed = [name for name, ok in checks.items() if not ok]
if failed:
    logger.warning("FORMAT 规则自检未通过，失败项：%s", failed)
else:
    logger.info("FORMAT 规则自检全部通过（共 %d 项）", len(checks))
logger.info("回答统计：段落数=%d，句子数=%d，长度=%d 字符",
            len(paragraphs), len(sentences), len(answer))

for name, ok in checks.items():
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")

# --------------------------------------------------------------------------- #
# 附：其他常用取数方式
# --------------------------------------------------------------------------- #
#   result["messages"][0].content   -> 系统提示词（渲染后的 RTCF 模板）
#   result["messages"][1].content   -> 用户输入
#   result["messages"][-1].content  -> 模型最终回答
#   result["messages"][-1].usage_metadata["total_tokens"] -> 本次调用的总 token 数
#   result["messages"][-1].response_metadata["model_name"]  -> 实际使用的模型名
#
# 需要流式输出时改用 agent.stream(..., stream_mode="values") 逐节点获取。
