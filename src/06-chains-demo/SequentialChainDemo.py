"""链（Chain）演示 3：顺序链（SequentialChain / 多段串联）。

核心概念
--------
把 **多段 LLM 调用按固定顺序串起来**，每一段的输出自动成为下一段的输入。
本文件是一个「绩效改进计划」的三段式流水线：

    员工绩效评语原文（employee_review）
       │
       ▼
    ① chain_1：总结这段绩效评语
       │  输出：AIMessage（summary）
       ▼
    ② chain_2：从总结中找出主要弱点
       │  输出：AIMessage（weaknesses）
       ▼
    ③ chain_3：针对这些弱点制定改进计划
       │
       ▼
    最终结果：一份个性化的改进计划

与 SimpleSequentialChainDemo 的区别
-----------------------------------
  - ``SimpleSequentialChainDemo.py``：两段，第二段模板用 ``{abc}`` 这类
    随意命名的占位符；
  - 本文件：三段，且三段的占位符名分别叫 ``{review}`` / ``{review_summary}`` /
    ``{weaknesses}``，语义更清晰，方便理解「每段吃什么、吐什么」。

实现方式
--------
旧版 ``langchain.chains.SequentialChain`` 已随 ``langchain.chains`` 模块一起
从 LangChain 1.x 中移除。现在统一用 **LCEL（LangChain Expression Language）**：

    seq_chain = chain_1 | chain_2 | chain_3

依赖：pip install langchain langchain-deepseek python-dotenv
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

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

# 优先读 DEEPSEEK_API_KEY，其次兼容通用命名 API_KEY；
# 都没有就立刻报错，避免等到请求接口时才失败、难以排查。
api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY")
if not api_key:
    raise RuntimeError(
        "未找到 API Key，请在 .env 中设置 DEEPSEEK_API_KEY（或 API_KEY）"
    )
# 再显式写回环境变量，供 init_chat_model 按 provider 自动读取。
os.environ["DEEPSEEK_API_KEY"] = api_key

# --------------------------------------------------------------------------- #
# 1. 初始化 LLM
# --------------------------------------------------------------------------- #
# init_chat_model 是统一的模型工厂函数，签名形如 init_chat_model("provider:model", ...)
#   - provider 决定用哪家的客户端（deepseek / openai / anthropic ...）
#   - model    是具体模型名
# temperature=0 让输出尽量确定，便于复现同一条流水线的结果。
llm = init_chat_model(model="deepseek-chat", api_key=api_key, temperature=0)
logger.info("LLM 初始化完成：model=deepseek-chat, temperature=0")

# --------------------------------------------------------------------------- #
# 2. 准备输入数据：一段员工绩效评语
# --------------------------------------------------------------------------- #
# 真实项目里这段文本可能来自数据库、HR 系统或上传的文档；
# 这里直接内联一个多行字符串作为演示输入。
#
# 内容分为两块：
#   Strengths  —— 优势（技术能力、协作、主动性、适应力、问题解决）
#   Weaknesses —— 不足（时间管理、书面沟通、不善委派）
# 后面三段链会围绕这份评语做「总结 → 找弱点 → 定计划」。
employee_review = """
Employee Information:
Name: Joe Smith
Position: Software Engineer
Date of Review: Jan 01, 2000

Strengths:
Joe is a highly skilled software engineer with a deep understanding of programming languages, algorithms, and software development best practices. His technical expertise shines through in his ability to efficiently solve complex problems and deliver high-quality code.

One of Joe's greatest strengths is his collaborative nature. He actively engages with cross-functional teams, contributing valuable insights and seeking input from others. His open-mindedness and willingness to learn from colleagues make him a true team player.

Joe consistently demonstrates initiative and self-motivation. He takes the lead in seeking out new projects and challenges, and his proactive attitude has led to significant improvements in existing processes and systems. His dedication to self-improvement and growth is commendable.

Another notable strength is Joe's adaptability. He has shown great flexibility in handling changing project requirements and learning new technologies. This adaptability allows him to seamlessly transition between different projects and tasks, making him a valuable asset to the team.

Joe's problem-solving skills are exceptional. He approaches issues with a logical mindset and consistently finds effective solutions, often thinking outside the box. His ability to break down complex problems into manageable parts is key to his success in resolving issues efficiently.

Weaknesses:
While Joe possesses numerous strengths, there are a few areas where he could benefit from improvement. One such area is time management. Occasionally, Joe struggles with effectively managing his time, resulting in missed deadlines or the need for additional support to complete tasks on time. Developing better prioritization and time management techniques would greatly enhance his efficiency.

Another area for improvement is Joe's written communication skills. While he communicates well verbally, there have been instances where his written documentation lacked clarity, leading to confusion among team members. Focusing on enhancing his written communication abilities will help him effectively convey ideas and instructions.

Additionally, Joe tends to take on too many responsibilities and hesitates to delegate tasks to others. This can result in an excessive workload and potential burnout. Encouraging him to delegate tasks appropriately will not only alleviate his own workload but also foster a more balanced and productive team environment.
"""

# --------------------------------------------------------------------------- #
# 3. 第一段链：总结绩效评语
# --------------------------------------------------------------------------- #
# 提示词模板：{review} 是输入变量名，调用时按名字传值。
template1 = "Give a summary of this employee's performance review:\n{review}"

# 编译成 ChatPromptTemplate：内部会记录 input_variables == ("review",)
prompt1 = ChatPromptTemplate.from_template(template1)

# 组装成链：提示词 | 模型
# 运行时：prompt1.invoke(employee_review) -> PromptValue
#         llm.invoke(PromptValue)           -> AIMessage
chain_1 = prompt1 | llm
logger.info("第一段链构建完成：总结绩效评语，输入变量=%s", prompt1.input_variables)

# 单独跑第一段，方便观察中间结果
# invoke 对只有一个输入变量的模板，允许直接传值（不必写 {"review": ...}）；
# 内部会自动包成 {"review": employee_review}。
logger.info("调用第一段链（总结）")
summary_result = chain_1.invoke(employee_review)
logger.info("第一段完成，摘要长度=%d 字符", len(summary_result.content))
print("\n\033[1m Summary of employee's performance review")
print("-" * 50)
print("\033[0m" + summary_result.content)

# --------------------------------------------------------------------------- #
# 4. 第二段链：识别关键弱点
# --------------------------------------------------------------------------- #
# 输入变量名 {review_summary} 表示「喂进来的是上一段的总结」。
template2 = "Identify key employee weaknesses in this review summary:\n{review_summary}"

prompt2 = ChatPromptTemplate.from_template(template2)

chain_2 = prompt2 | llm
logger.info("第二段链构建完成：识别关键弱点，输入变量=%s", prompt2.input_variables)

# 这里演示「临时串联两段」：
#   chain_1 | chain_2  →  RunnableSequence
# 第一段输出的 AIMessage 会被整体作为下一段的输入，
# 而下一段模板只有一个变量 {review_summary}，因此该变量被赋值为上一段结果。
logger.info("调用两段串联链（总结 -> 识别弱点）")
two_step_result = (chain_1 | chain_2).invoke(employee_review)
logger.info("两段串联完成，弱点列表长度=%d 字符", len(two_step_result.content))
print("\n\033[1m Identify key employee weaknesses in the performance review summary")
print("-" * 50)
print("\033[0m" + two_step_result.content)

# --------------------------------------------------------------------------- #
# 5. 第三段链：制定改进计划
# --------------------------------------------------------------------------- #
# 输入变量名 {weaknesses} 对应第二段的输出。
# 模板用括号包裹、多行书写，方便后续继续加内容而不破坏格式。
template3 = (
    "Create a personalized plan to help address and fix these weaknesses:\n{weaknesses}"
)

prompt3 = ChatPromptTemplate.from_template(template3)

chain_3 = prompt3 | llm
logger.info("第三段链构建完成：制定改进计划，输入变量=%s", prompt3.input_variables)

# --------------------------------------------------------------------------- #
# 6. 串成完整顺序链并调用
# --------------------------------------------------------------------------- #
# seq_chain 的运行数据流（逐段传递）：
#   employee_review (str)
#     → chain_1 → AIMessage(总结)
#     → chain_2 → AIMessage(弱点列表)
#     → chain_3 → AIMessage(改进计划)
#
# 每次 invoke 都会实际调用一次大模型接口，因此这条链是「三次 API 调用」串联。
seq_chain = chain_1 | chain_2 | chain_3
logger.info("完整顺序链组装完成：3 段串联（共 3 次模型调用）")

logger.info("开始执行完整顺序链")
results = seq_chain.invoke(employee_review)
logger.info("顺序链执行完成，最终结果长度=%d 字符", len(results.content))

print("\n\033[1m Create a personalized plan to help address and fix these weaknesses")
print("-" * 50)
print("\033[0m" + results.content)

# --------------------------------------------------------------------------- #
# 7. 补充说明与常见坑
# --------------------------------------------------------------------------- #
# 7.1 隐式包装规则
#     当上游输出是 AIMessage（而非字典）而下游模板只有一个变量时，
#     LangChain 会把整个消息对象赋给那个唯一的变量。
#
# 7.2 模板里出现「看起来像占位符的普通文本」时要转义
#     例如想输出字面量 {review}，应写成 {{review}}，
#     否则会被当成变量名，导致 "missing variables" 报错。
#
# 7.3 若想在段与段之间做显式加工（例如只取 .content、裁剪字数、加格式），
#     可插入 RunnableLambda：
#         from langchain_core.runnables import RunnableLambda
#         to_text = RunnableLambda(lambda msg: {"review_summary": msg.content})
#         seq_chain = chain_1 | to_text | chain_2 | chain_3
#
# 7.4 调试技巧
#     - 打印链的结构：print(chain_1)
#     - 查看某段的输入变量：print(prompt1.input_variables)
#     - 流式观察：for chunk in seq_chain.stream(employee_review): ...
#
# 7.5 旧版 API（已移除，仅供理解历史写法）：
#         from langchain.chains import SequentialChain
#         chain = SequentialChain(
#             chains=[chain_1, chain_2, chain_3],
#             input_variables=["review"],
#             output_variables=["weaknesses"],
#         )
