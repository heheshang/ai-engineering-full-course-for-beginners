"""Prompt 工程演示 3：约束式输出（Constrained Output Prompting）。

核心概念：
    约束式输出提示词（Constrained Output Prompting）是在提示词里写死**可验证的硬性规则**，
    让模型的输出必须满足这些规则，而不是只描述"希望模型怎么回答"。

    本例给出 8 条可机械校验的约束：
      - 段落数固定为 3 段
      - 每段的句子数固定（第 1 段 2 句、第 2 段 2 句）
      - 第 3 段必须以一个问句结尾
      - 不得使用感叹号
      - 道歉（apology）恰好出现 1 次
      - 短语 "I understand" 恰好出现 1 次
      - 整体保持专业语气

    对比一下三种提示词风格的差异：
      1. 模糊型："Write a nice reply to the customer."      -> 模型自由发挥，无法验收
      2. 描述型："Reply politely and mention the refund."    -> 约束偏软，仍要人工检查
      3. 约束型（本例）：逐条列出可计数、可检查的规则       -> 输出可被脚本自动验收

    实际项目中，"约束式输出"通常还会配合 **结构化输出**
    （``response_format`` / Pydantic / JSON Schema）把结果变成可解析的结构体，
    约束从"文字规则"升级为"schema 强约束"。

关于 LangChain 1.x 的 API 约束（重要）：
    ``create_agent()`` 的系统提示词参数叫 ``system_prompt``，
    它的类型是 ``str | SystemMessage | None``，
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

关于本文件中的 SystemMessage：
    ``SystemMessage(content=...)`` 是 LangChain 的系统消息对象，
    作用是给模型设定"身份/角色/规则"。
    相比直接把整段文本塞进 user 消息，
    放在 system 位置对模型的约束力更强（尤其在推理模型上更明显）。
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
#     约束式输出场景通常希望结果稳定，可以调低到 0.2~0.4。
#   api_key：从环境变量显式传入。
llm = ChatDeepSeek(model="deepseek-flash", temperature=0.5, api_key=api_key)
logger.info("LLM 初始化完成：model=deepseek-flash, temperature=0.5")

# --------------------------------------------------------------------------- #
# 1. 定义约束式提示词模板
# --------------------------------------------------------------------------- #
# 写法要点：
#   1) 先给角色（You are a customer support specialist）；
#   2) 再用编号列表逐条列出**可验证**的约束；
#   3) 把用户输入放在最后，用 {customer_message} 占位；
#   4) 约束写成祈使句/规则句，避免"最好""尽量"这类软措辞。
constrained_template = """
You are a customer support specialist. Generate a response to the customer's message following these strict constraints:

1. Response must be exactly 3 paragraphs long
2. First paragraph must be exactly 2 sentences showing empathy
3. Second paragraph must provide exactly 2 concrete solutions
4. Third paragraph must end with a question to engage the customer
5. Total response must maintain a professional tone
6. Must not use any exclamation marks
7. Must include exactly one apology
8. Must use the phrase "I understand" exactly once

Customer message: {customer_message}
"""

# PromptTemplate 会解析模板里的 {customer_message}，
# 声明 input_variables 后，运行时可以安全地传入对应关键字参数。
constrained_prompt = PromptTemplate(
    input_variables=["customer_message"], template=constrained_template
)

# --------------------------------------------------------------------------- #
# 2. 准备客户消息（用户输入）
# --------------------------------------------------------------------------- #
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
# 再用 SystemMessage 包一层，明确"这是系统规则"而不是"用户说的话"。
constrained_agent = create_agent(
    model=llm,
    system_prompt=SystemMessage(
        content=constrained_prompt.format(customer_message=customer_message)
    ),
)
logger.info("Agent 创建完成，约束模板渲染后长度=%d 字符",
             len(constrained_prompt.format(customer_message=customer_message)))

# --------------------------------------------------------------------------- #
# 4. 调用 Agent
# --------------------------------------------------------------------------- #
# 输入必须是 {"messages": [...]} 形式的消息列表。
# 注意区分两类内容：
#   - system_prompt（上面的约束模板）-> 走 system 位置，约束力强
#   - HumanMessage                -> 走 user 位置，表示"用户此刻说的话"
# 本例中客户消息已经写进 system prompt 了，所以 user 轮只需一句触发语。
# 如果希望完全按"纯 user 输入"的语义来写，可以把 customer_message
# 作为 HumanMessage 的内容传入，并把 system prompt 里那句
# "Customer message: {customer_message}" 删掉。
logger.info("开始调用 Agent（约束式提示）")
try:
    result = constrained_agent.invoke(
        {
            "messages": [
                HumanMessage(content="Please respond to my message."),
            ]
        }
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
print(result["messages"][-1].content)

# --------------------------------------------------------------------------- #
# 附：约束校验（把"提示词约束"变成"代码断言"）
# --------------------------------------------------------------------------- #
# 约束型提示词的最大价值在于：规则是可机械校验的。
# 下面这段就是典型的"自动验收"，可在生产中直接复用。
answer = result["messages"][-1].content

paragraphs = [p for p in answer.split("\n\n") if p.strip()]
exclamation_count = answer.count("!")
apology_count = answer.lower().count("apolog")
understand_count = answer.lower().count("i understand")

checks = {
    "段落数为 3": len(paragraphs) == 3,
    "无感叹号": exclamation_count == 0,
    "道歉恰好 1 次": apology_count == 1,
    '"I understand" 恰好 1 次': understand_count == 1,
    "第 3 段以问句结尾": paragraphs[-1].strip().endswith("?") if paragraphs else False,
}

# 日志：把校验结果同时写进日志文件，方便在 CI 或回归测试里直接判定成败
failed = [name for name, ok in checks.items() if not ok]
if failed:
    logger.warning("约束校验未通过，失败项：%s", failed)
else:
    logger.info("约束校验全部通过（共 %d 项）", len(checks))
logger.info(
    "校验统计：段落数=%d，感叹号=%d，道歉=%d，I understand=%d",
    len(paragraphs), exclamation_count, apology_count, understand_count,
)

print("\n" + "=" * 70)
print("4) 约束自动校验")
print("=" * 70)
for name, ok in checks.items():
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")

# --------------------------------------------------------------------------- #
# 附：其他常用取数方式
# --------------------------------------------------------------------------- #
#   result["messages"][0].content   -> 系统提示词（渲染后的约束模板）
#   result["messages"][1].content   -> 用户输入
#   result["messages"][-1].content  -> 模型最终回答
#   result["messages"][-1].usage_metadata["total_tokens"] -> 本次调用的总 token 数
#   result["messages"][-1].response_metadata["model_name"]  -> 实际使用的模型名
#
# 需要流式输出时改用 agent.stream(..., stream_mode="values") 逐节点获取。
