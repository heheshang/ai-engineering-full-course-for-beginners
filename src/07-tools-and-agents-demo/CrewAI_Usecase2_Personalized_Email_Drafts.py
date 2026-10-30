"""CrewAI 示例：线索画像 + 个性化外联邮件（DeepSeek）。

本脚本针对 CrewAI 1.15.x + DeepSeek 修复了以下几处问题：

1. **结构化输出**：DeepSeek 官方 API 不支持 OpenAI 的 ``json_schema``
   ``response_format``（会报 ``This response_format type is unavailable now``），
   而 CrewAI 的 Memory 分析等链路正是走这条路径。
   这里通过本地 OpenAI 兼容代理（``src/utils/deepseek_proxy.py``）访问 DeepSeek：
   代理把 ``json_schema`` 请求改写成 DeepSeek 支持的 tool-calling，再把响应
   还原成 OpenAI 期望的格式。于是 CrewAI 可以用最标准的 ``openai`` provider，
   无需任何子类 hack。
2. **Memory 嵌入**：CrewAI Memory 默认使用 OpenAI ``text-embedding-3-large``，
   没有 ``OPENAI_API_KEY`` 时 ``search_memory`` / ``remember`` 会直接报错。
   这里改为本地 ONNX MiniLM（all-MiniLM-L6-v2），完全离线可用。
3. **网页搜索**：Serper API Key 无效（403 Unauthorized）。这里默认使用
   DuckDuckGo（``ddgs``，无需 Key）；若配置了 ``SERPER_API_KEY`` 则优先用 Serper。
4. **instructions 目录不存在**：脚本会自动创建，并写入示例内部资料，
   避免 ``list_files_in_directory`` 报 "is not a directory"。

运行前先启动代理（另一个终端）::

    python src/utils/deepseek_proxy.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from crewai import LLM, Agent, Crew, Task
from crewai.memory.unified_memory import Memory
from crewai.rag.embeddings.factory import build_embedder
from crewai.tools import BaseTool
from crewai_tools import DirectoryReadTool, FileReadTool, SerperDevTool
from dotenv import load_dotenv

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

INSTRUCTIONS_DIR = Path("instructions")
INSTRUCTIONS_DIR.mkdir(exist_ok=True)
logger.info("内部资料目录就绪：%s", INSTRUCTIONS_DIR.resolve())

# 内部资料：让 Agent 有可读取的「ABC 解决方案 / ICP」上下文
INSTRUCTIONS_FILE = INSTRUCTIONS_DIR / "abc_solutions.md"
if not INSTRUCTIONS_FILE.exists():
    INSTRUCTIONS_FILE.write_text(
        """# ABC Solutions & Ideal Customer Profile (ICP)

## What we sell
- **AI Mentorship Program**: 1:1 mentorship for engineers moving into AI/ML roles.
- **Production AI Course**: 6-week hands-on cohort course on LLM apps, agents and evals.
- **Team Enablement**: Workshop + reusable templates for product/engineering teams.

## Ideal Customer Profile (ICP)
- Online learning platforms and ed-tech companies (10-200 employees).
- Recently launched an AI-powered product, feature or course.
- Building their team around AI engineers / ML practitioners.
- Pain points: upskilling engineers, shipping AI features faster, keeping pace
  with model changes.

## Recent customer wins
- TechWithMala (AI content & education) - AI mentorship program, 12 engineers.
- AcademiaAI - Production AI course for their content team.
""",
        encoding="utf-8",
    )
    logger.info("已生成示例内部资料：%s", INSTRUCTIONS_FILE)
else:
    # 已存在则复用，避免每次运行都覆盖用户自己补充的内容
    logger.info("内部资料已存在，跳过生成：%s", INSTRUCTIONS_FILE)


# ---------------------------------------------------------------------------
# 1. DeepSeek 模型：经由本地 OpenAI 兼容代理访问
# ---------------------------------------------------------------------------
# DeepSeek 官方 API 不支持 OpenAI 的 ``response_format={"type": "json_schema"}``，
# 而 CrewAI 的结构化输出（Memory 分析等）正是走这条路径，会直接报
# ``400 This response_format type is unavailable now``。
#
# 解决办法：启动本地代理 ``src/utils/deepseek_proxy.py``，它会把 json_schema
# 请求改写成 DeepSeek 支持的 tool-calling，再把结果还原成 OpenAI 期望的格式。
# 这样 CrewAI 可以用最标准的 ``openai`` provider，无需任何子类 hack。
#
# 启动代理（另一个终端）：
#     python src/utils/deepseek_proxy.py
PROXY_BASE_URL = os.getenv("DEEPSEEK_PROXY_URL", "http://127.0.0.1:8765/v1")

llm = LLM(
    model="openai/deepseek-chat",
    base_url=PROXY_BASE_URL,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.0,
    # 多智能体协作会产生很长的 tool-call arguments，max_tokens 给足，
    # 否则会被截断导致 JSON 解析失败（代理侧也有自动重试兜底）。
    max_tokens=8000,
)
logger.info(
    "LLM 初始化完成：model=openai/deepseek-chat, base_url=%s, max_tokens=8000", PROXY_BASE_URL
)


# ---------------------------------------------------------------------------
# 2. 网页搜索工具：优先 Serper（若 Key 有效），否则回退 DuckDuckGo
# ---------------------------------------------------------------------------
class DuckDuckGoSearchTool(BaseTool):
    """无需 API Key 的网页搜索工具（DuckDuckGo）。"""

    name: str = "Search the internet with DuckDuckGo"
    description: str = (
        "A tool that can be used to search the internet for a company, person or topic. "
        "It returns the top search results (title, url and snippet)."
    )
    max_results: int = 6
    # ddgs 的各后端可用性会变（限流/下线），逐个回退直到拿到结果。
    # "auto" 让 ddgs 自己挑一个可用后端，通常最稳；后面是显式兜底列表。
    backends: tuple[str, ...] = (
        "auto",
        "duckduckgo",
        "brave",
        "wikipedia",
        "mojeek",
        "yahoo",
        "startpage",
        "grokipedia",
    )

    def _search(self, search_query: str) -> list[dict[str, Any]]:
        from ddgs import DDGS

        errors: list[str] = []
        for backend in self.backends:
            try:
                logger.info("尝试搜索后端：%s，query=%r", backend, search_query)
                results = list(
                    DDGS().text(
                        search_query,
                        max_results=self.max_results,
                        backend=backend,
                    )
                )
                if results:
                    logger.info("后端 %s 命中 %d 条结果", backend, len(results))
                    return results
                errors.append(f"{backend}: no results")
                logger.warning("后端 %s 未返回结果，尝试下一个后端", backend)
            except Exception as exc:  # pragma: no cover - 网络异常兜底
                errors.append(f"{backend}: {exc}")
                logger.warning("后端 %s 搜索失败：%s，尝试下一个后端", backend, exc)
        logger.error("所有搜索后端均失败：%s", "; ".join(errors))
        raise RuntimeError("; ".join(errors))

    def _run(self, search_query: str) -> str:
        try:
            results = self._search(search_query)
        except Exception as exc:  # pragma: no cover - 网络异常兜底
            # 搜索失败不让整个 Crew 崩掉：把错误信息作为工具返回值交给模型
            logger.error("网页搜索失败：%s", exc)
            return f"Search failed: {exc}"

        if not results:
            logger.warning("网页搜索无结果：query=%r", search_query)
            return f"No results found for '{search_query}'."

        lines = []
        for item in results:
            lines.append(
                f"- {item.get('title', '')}\n  URL: {item.get('href', '')}\n"
                f"  {item.get('body', '')}"
            )
        logger.info("网页搜索完成：query=%r, 共 %d 条结果", search_query, len(results))
        return f"Search results for '{search_query}':\n" + "\n".join(lines)


serper_api_key = os.getenv("SERPER_API_KEY")
if serper_api_key:
    os.environ["SERPER_API_KEY"] = serper_api_key
    search_tool: BaseTool = SerperDevTool()
    logger.info("网页搜索工具已选用：SerperDevTool")
else:
    search_tool = DuckDuckGoSearchTool()
    logger.info("未配置 SERPER_API_KEY，网页搜索工具选用 DuckDuckGoSearchTool")


# ---------------------------------------------------------------------------
# 3. 工具
# ---------------------------------------------------------------------------
directory_read_tool = DirectoryReadTool(directory=str(INSTRUCTIONS_DIR))
file_read_tool = FileReadTool()


class SentimentAnalysisTool(BaseTool):
    """自定义工具：情感分析（示例实现）。"""

    name: str = "Sentiment Analysis Tool"
    description: str = (
        "Analyzes the sentiment of text to ensure positive and engaging communication."
    )

    def _run(self, text: str) -> str:
        # 自定义工具的核心：把「真实业务逻辑」写在这里，返回一个字符串给模型。
        # 这里为了演示只返回固定值 "positive"；
        # 真实项目可以接入情感分析模型 / 第三方 API / 规则引擎。
        # 注意：工具的返回值会原样进入模型的上下文，所以应当简洁、信息量高。
        return "positive"


sentiment_analysis_tool = SentimentAnalysisTool()


# ---------------------------------------------------------------------------
# 4. Agents
# ---------------------------------------------------------------------------
# 两个 Agent 的分工：
#   - sales_rep_agent（销售代表）：做「研究」——挖掘线索、编画像；
#   - lead_sales_rep_agent（线索销售）：做「创作」——基于画像写个性化外联邮件。
#
# allow_delegation=False：不允许互相委派。
# 两者是上下游关系而非平级协作，禁掉委派可以避免模型「自己给自己派活」，
# 让流程更确定、token 消耗也更可控。
#
# 说明：role / goal / backstory 保持英文，因为它们会原样拼进系统提示词，
# 英文人设对模型的指令遵循度更好。
sales_rep_agent = Agent(
    role="Sales Representative",
    goal="Identify high-value leads that match our ideal customer profile",
    backstory=(
        "As a part of the dynamic sales team at ABC, "
        "your mission is to scour "
        "the digital landscape for potential leads. "
        "Armed with cutting-edge tools "
        "and a strategic mindset, you analyze data, "
        "trends, and interactions to "
        "unearth opportunities that others might overlook. "
        "Your work is crucial in paving the way "
        "for meaningful engagements and driving the company's growth."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

# 线索销售 Agent：把前面产出的画像转成「可直接发出去」的邮件文案
lead_sales_rep_agent = Agent(
    role="Lead Sales Representative",
    goal="Nurture leads with personalized, compelling communications",
    backstory=(
        "Within the vibrant ecosystem of ABC's sales department, "
        "you stand out as the bridge between potential clients "
        "and the solutions they need."
        "By creating engaging, personalized messages, "
        "you not only inform leads about our offerings "
        "but also make them feel seen and heard."
        "Your role is pivotal in converting interest "
        "into action, guiding leads through the journey "
        "from curiosity to commitment."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm,
)


# ---------------------------------------------------------------------------
# 5. Tasks
# ---------------------------------------------------------------------------
# Task 1：线索画像
# 说明：description 里的 {lead_name} / {industry} 等占位符
# 会在 crew.kickoff(inputs=...) 时被真实值替换。
lead_profiling_task = Task(
    description=(
        "Conduct an in-depth analysis of {lead_name}, "
        "a company in the {industry} sector "
        "that recently showed interest in our solutions. "
        "Utilize all available data sources "
        "to compile a detailed profile, "
        "focusing on key decision-makers, recent business "
        "developments, and potential needs "
        "that align with our offerings. "
        "This task is crucial for tailoring "
        "our engagement strategy effectively.\n"
        "Don't make assumptions and "
        "only use information you are absolutely sure about."
    ),
    expected_output=(
        "A comprehensive report on {lead_name}, "
        "including company background, "
        "key personnel, recent milestones, and identified needs. "
        "Highlight potential areas where "
        "our solutions can provide value, "
        "and suggest personalized engagement strategies."
    ),
    tools=[directory_read_tool, file_read_tool, search_tool],
    agent=sales_rep_agent,
)

personalized_outreach_task = Task(
    description=(
        "Using the insights gathered from "
        "the lead profiling report on {lead_name}, "
        "craft a personalized outreach campaign "
        "aimed at {key_decision_maker}, "
        "the {position} of {lead_name}. "
        "The campaign should address their recent {milestone} "
        "and how our solutions can support their goals. "
        "Your communication must resonate "
        "with {lead_name}'s company culture and values, "
        "demonstrating a deep understanding of "
        "their business and needs.\n"
        "Don't make assumptions and only "
        "use information you are absolutely sure about."
    ),
    expected_output=(
        "A series of personalized email drafts "
        "tailored to {lead_name}, "
        "specifically targeting {key_decision_maker}."
        "Each draft should include "
        "a compelling narrative that connects our solutions "
        "with their recent achievements and future goals. "
        "Ensure the tone is engaging, professional, "
        "and aligned with {lead_name}'s corporate identity."
    ),
    tools=[sentiment_analysis_tool, search_tool],
    agent=lead_sales_rep_agent,
)


# ---------------------------------------------------------------------------
# 6. Crew
# ---------------------------------------------------------------------------
# 显式构造 Memory，好处是可以指定：
#   - embedder: 本地 ONNX MiniLM，替代默认的 OpenAI text-embedding-3-large，
#     这样 memory=True 时不再需要 OPENAI_API_KEY。
memory = Memory(
    llm=llm,
    embedder=build_embedder({"provider": "onnx"}),
)

crew = Crew(
    agents=[sales_rep_agent, lead_sales_rep_agent],
    tasks=[lead_profiling_task, personalized_outreach_task],
    verbose=True,
    memory=memory,
    embedder={"provider": "onnx"},
)


# ---------------------------------------------------------------------------
# 7. 运行
# ---------------------------------------------------------------------------
inputs = {
    "lead_name": "aiwithMala",
    "industry": "Online Learning Platform",
    "key_decision_maker": "Mala H",
    "position": "CEO",
    "milestone": "product launch",
}
logger.info("Crew 输入参数：%s", inputs)
logger.info("开始执行 Crew：agents=2, tasks=2, memory=ONNX MiniLM")

try:
    result = crew.kickoff(inputs=inputs)
except Exception as exc:
    # 常见失败原因：本地代理未启动（连接被拒绝）、Memory 嵌入模型下载失败
    logger.error("Crew 执行失败：%s", exc)
    logger.exception("Crew 执行异常详情")
    raise

logger.info("Crew 执行完成，最终输出长度=%d 字符", len(result.raw))
print(result.raw)
