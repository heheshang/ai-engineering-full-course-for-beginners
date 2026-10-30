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

load_dotenv()

# ---------------------------------------------------------------------------
# 0. 环境准备
# ---------------------------------------------------------------------------
if not os.getenv("DEEPSEEK_API_KEY"):
    raise RuntimeError("未找到 API Key，请在 .env 中设置 DEEPSEEK_API_KEY")

# ---------------------------------------------------------------------------
# 1. 初始化 DeepSeek 模型（经由本地 OpenAI 兼容代理）
# ---------------------------------------------------------------------------
llm = LLM(
    model="openai/deepseek-chat",
    base_url=os.getenv("DEEPSEEK_PROXY_URL", "http://127.0.0.1:8765/v1"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.0,
    max_tokens=8000,
)

# ---------------------------------------------------------------------------
# 2. 抓取 Wikipedia 页面
# ---------------------------------------------------------------------------
scrape_tool = ScrapeWebsiteTool(
    website_url="https://en.wikipedia.org/wiki/Artificial_intelligence"
)

text = scrape_tool.run()

# 保存抓取结果到本地文本文件
os.makedirs("data", exist_ok=True)
with open("data/ai.txt", "w", encoding="utf-8") as file:
    file.write(text)

# ---------------------------------------------------------------------------
# 3. 基于本地 ONNX 嵌入的 TXT 语义检索工具
# ---------------------------------------------------------------------------
search_tool = TXTSearchTool(
    txt="data/ai.txt",
    config={
        "embedding_model": {"provider": "onnx"},
        "vectordb": {
            "provider": "chromadb",
            "config": {"collection_name": "ai_txt"},
        },
    },
)

context = search_tool.run("What is natural language processing?")

# ---------------------------------------------------------------------------
# 4. 创建 Agent
# ---------------------------------------------------------------------------
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
test_task = Task(
    description="Understand the topic and give the correct response",
    tools=[search_tool],
    agent=data_analyst,
    expected_output="Give a correct response",
)

# ---------------------------------------------------------------------------
# 6. 创建并运行 Crew
# ---------------------------------------------------------------------------
crew = Crew(agents=[data_analyst], tasks=[test_task])

output = crew.kickoff()

print(output.raw)
