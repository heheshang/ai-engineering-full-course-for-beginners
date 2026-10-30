"""CrewAI 示例：网页抓取 + TXT 语义检索 + DeepSeek 问答。

说明：
1. DeepSeek 通过本地 OpenAI 兼容代理访问（``src/utils/deepseek_proxy.py``）。
   代理会把 OpenAI 的 ``json_schema`` 结构化输出请求改写成 DeepSeek 支持的
   tool-calling，并把响应还原成 OpenAI 格式。
   启动代理（另一个终端）：``python src/utils/deepseek_proxy.py``
2. ``TXTSearchTool`` 默认使用 OpenAI 嵌入模型，这里改为本地 ONNX MiniLM
   （all-MiniLM-L6-v2），无需额外的 OpenAI Key，离线也能跑。
"""

import os

from crewai import LLM, Agent, Crew, Task
from crewai_tools import ScrapeWebsiteTool, TXTSearchTool
from dotenv import load_dotenv

import sys
from pathlib import Path

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.logging_utils import get_logger

# 初始化 logger：日志同时输出到控制台和 logs/course.log
logger = get_logger(__name__)


load_dotenv()

# ---------------------------------------------------------------------------
# 0. 环境准备
# ---------------------------------------------------------------------------
if not os.getenv("DEEPSEEK_API_KEY"):
    raise RuntimeError("未找到 API Key，请在 .env 中设置 DEEPSEEK_API_KEY")
logger.info("环境检查通过：已检测到 DEEPSEEK_API_KEY")

# ---------------------------------------------------------------------------
# 1. 初始化 DeepSeek 模型（经由本地 OpenAI 兼容代理）
# ---------------------------------------------------------------------------
# 注意：这里的 base_url 指向本地代理（默认 http://127.0.0.1:8765/v1），
# 代理负责把 OpenAI 的 json_schema 结构化输出请求改写成 DeepSeek 支持的 tool-calling。
# 若代理没启动，请求会报「连接被拒绝」，请先运行：python src/utils/deepseek_proxy.py
PROXY_BASE_URL = os.getenv("DEEPSEEK_PROXY_URL", "http://127.0.0.1:8765/v1")
llm = LLM(
    model="openai/deepseek-chat",
    base_url=PROXY_BASE_URL,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.0,
    max_tokens=8000,
)
logger.info("LLM 初始化完成：model=openai/deepseek-chat, base_url=%s, max_tokens=8000", PROXY_BASE_URL)

# ---------------------------------------------------------------------------
# 2. 抓取 Wikipedia 页面
# ---------------------------------------------------------------------------
SCRAPE_URL = "https://en.wikipedia.org/wiki/Artificial_intelligence"
logger.info("开始抓取网页：%s", SCRAPE_URL)
scrape_tool = ScrapeWebsiteTool(website_url=SCRAPE_URL)

text = scrape_tool.run()
logger.info("网页抓取完成，正文长度=%d 字符", len(text))

# 保存抓取结果到本地文本文件
os.makedirs("data", exist_ok=True)
OUTPUT_TXT = "data/ai.txt"
with open(OUTPUT_TXT, "w", encoding="utf-8") as file:
    file.write(text)
logger.info("抓取结果已保存到 %s（%d 字节）", OUTPUT_TXT, os.path.getsize(OUTPUT_TXT))

# ---------------------------------------------------------------------------
# 3. 基于本地 ONNX 嵌入的 TXT 语义检索工具
# ---------------------------------------------------------------------------
# 说明：TXTSearchTool 默认使用 OpenAI 嵌入模型（需要 OPENAI_API_KEY），
# 这里改用 provider="onnx"，走本地 ONNX 版 all-MiniLM-L6-v2，离线可用、无额外费用。
logger.info("正在初始化 TXT 语义检索工具（embedding=onnx, vectordb=chromadb）")
search_tool = TXTSearchTool(
    txt=OUTPUT_TXT,
    config={
        "embedding_model": {"provider": "onnx"},
        "vectordb": {
            "provider": "chromadb",
            "config": {"collection_name": "ai_txt"},
        },
    },
)

RETRIEVAL_QUERY = "What is natural language processing?"
logger.info("开始语义检索：query=%r", RETRIEVAL_QUERY)
context = search_tool.run(RETRIEVAL_QUERY)
logger.info("语义检索完成，返回上下文长度=%d 字符", len(context))

# ---------------------------------------------------------------------------
# 4. 创建 Agent
# ---------------------------------------------------------------------------
# 说明：goal 里直接把检索到的 context 拼了进去，
# 这样 Agent 即使不主动调工具，也已经拿到了作答所需的素材。
# 代价是 context 会被重复传输（一次进 goal，一次进工具返回），token 翻倍。
# 更省 token 的做法是把 context 只放进 Task.description，让 Agent 按需调用工具。
logger.info("创建 Agent：role=Educator, allow_delegation=False")
data_analyst = Agent(
    role="Educator",
    goal=(
        "Based on the context provided, answer the question - "
        f"What is Natural Language Processing? Context - {context}"
    ),
    backstory="You are a data expert,中文回答内容",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm,
)

# ---------------------------------------------------------------------------
# 5. 创建 Task
# ---------------------------------------------------------------------------
logger.info("创建 Task：description=%r", test_task_description := "Understand the topic and give the correct response")
test_task = Task(
    description=test_task_description,
    tools=[search_tool],
    agent=data_analyst,
    expected_output="Give a correct response",
)

# ---------------------------------------------------------------------------
# 6. 创建并运行 Crew
# ---------------------------------------------------------------------------
logger.info("Crew 组装完成：agents=1, tasks=1，开始执行 kickoff")
crew = Crew(agents=[data_analyst], tasks=[test_task])

try:
    output = crew.kickoff()
except Exception as exc:  # noqa: BLE001 - 演示脚本，捕获后打印完整堆栈便于排查
    # 常见失败原因：本地代理未启动（连接被拒绝）、API Key 无效、模型返回超时
    logger.error("Crew 执行失败：%s", exc)
    logger.exception("Crew 执行异常详情")
    raise

logger.info("Crew 执行完成，最终输出长度=%d 字符", len(output.raw))
print(output.raw)
