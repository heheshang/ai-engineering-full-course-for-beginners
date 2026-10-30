"""CrewAI 示例：交易平台多智能体协作（DeepSeek）。

与前两个示例保持一致的技术方案：

1. **DeepSeek 经由本地 OpenAI 兼容代理访问**（``src/utils/deepseek_proxy.py``）。
   DeepSeek 官方 API 不支持 OpenAI 的 ``response_format={"type": "json_schema"}``，
   而 CrewAI 的结构化输出链路会用到它。代理把 ``json_schema`` 请求改写成
   DeepSeek 支持的 tool-calling，再把响应还原成 OpenAI 格式。
   启动代理（另一个终端）：``python src/utils/deepseek_proxy.py``
2. **网页搜索**：Serper API Key 无效（403 Unauthorized）时回退到
   DuckDuckGo（``ddgs``，无需 Key）；配置了 ``SERPER_API_KEY`` 则优先用 Serper。
3. **网页抓取**保留 ``ScrapeWebsiteTool``。
"""

import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import BaseTool
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from dotenv import load_dotenv
from typing import Any

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
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    raise RuntimeError("未找到 API Key，请在 .env 中设置 DEEPSEEK_API_KEY")
logger.info("环境检查通过：已检测到 DEEPSEEK_API_KEY")

serper_api_key = os.getenv("SERPER_API_KEY")
if serper_api_key:
    os.environ["SERPER_API_KEY"] = serper_api_key
    logger.info("已配置 SERPER_API_KEY，网页搜索将使用 Serper")
else:
    # 没有配置 Serper Key 时回退到 DuckDuckGo，无需任何 Key 即可运行
    logger.info("未配置 SERPER_API_KEY，网页搜索将回退到 DuckDuckGo")

# ---------------------------------------------------------------------------
# 1. 初始化 DeepSeek 模型（经由本地 OpenAI 兼容代理）
# ---------------------------------------------------------------------------
PROXY_BASE_URL = os.getenv("DEEPSEEK_PROXY_URL", "http://127.0.0.1:8765/v1")
llm = LLM(
    model="openai/deepseek-chat",
    base_url=PROXY_BASE_URL,
    api_key=deepseek_api_key,
    temperature=0.0,
    # 这里必须给足：多智能体协作会产生很长的 tool-call arguments
    # （delegate_work_to_coworker 常常超过 2000 tokens），
    # 截断会导致 JSON 解析失败。代理侧也有自动重试兜底。
    max_tokens=8000,
)
logger.info("LLM 初始化完成：model=openai/deepseek-chat, base_url=%s, max_tokens=8000", PROXY_BASE_URL)


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
            # 搜索失败不让整个 Crew 崩掉：把错误信息作为工具返回值交给模型，
            # 模型可以据此调整策略或如实告知无法获取信息
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


if serper_api_key:
    search_tool: BaseTool = SerperDevTool()
    logger.info("网页搜索工具已选用：SerperDevTool")
else:
    search_tool = DuckDuckGoSearchTool()
    logger.info("网页搜索工具已选用：DuckDuckGoSearchTool")
scrape_tool = ScrapeWebsiteTool()
logger.info("网页抓取工具已就绪：ScrapeWebsiteTool")

# ~~~~~~~~~~~Creating Agents ~~~~~~~~~~~~~~~
# 定义「数据分析师」Agent
#
# 说明：这里的 role / goal / backstory 都是**英文**，
# 因为 CrewAI 会把它们直接拼进系统提示词，模型的推理质量更高。
# 如果希望模型用中文回答，可以在 goal / expected_output 里显式写「请用中文回答」，
# 或者把 role / backstory 也改成中文（效果略降，但更可控）。
#
# allow_delegation=True：允许该 Agent 把工作委派给其他 Agent。
# 在 hierarchical 流程中，Manager 会根据 Agent 的能力描述自动分配任务。
data_analyst_agent = Agent(
    role="Data Analyst",
    goal="Monitor and analyze market data in real-time "
    "to identify trends and predict market movements.",
    backstory="Specializing in financial markets, this agent "
    "uses statistical modeling and machine learning "
    "to provide crucial insights. With a knack for data, "
    "the Data Analyst Agent is the cornerstone for "
    "informing trading decisions.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool],
    llm=llm,
)

# 定义「交易策略开发」Agent
#
# 职责边界：接收数据分析师的洞察 -> 设计并测试多种交易策略。
# 四个 Agent 各自负责一个环节，形成「分析 -> 策略 -> 执行 -> 风控」的流水线。
trading_strategy_agent = Agent(
    role="Trading Strategy Developer",
    goal="Develop and test various trading strategies based "
    "on insights from the Data Analyst Agent.",
    backstory="Equipped with a deep understanding of financial "
    "markets and quantitative analysis, this agent "
    "devises and refines trading strategies. It evaluates "
    "the performance of different approaches to determine "
    "the most profitable and risk-averse options.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool],
    llm=llm,
)

# 定义「交易执行顾问」Agent
#
# 职责边界：基于已批准的策略，分析时机、价格、执行细节等落地问题。
execution_agent = Agent(
    role="Trade Advisor",
    goal="Suggest optimal trade execution strategies "
    "based on approved trading strategies.",
    backstory="This agent specializes in analyzing the timing, price, "
    "and logistical details of potential trades. By evaluating "
    "these factors, it provides well-founded suggestions for "
    "when and how trades should be executed to maximize "
    "efficiency and adherence to strategy.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool],
    llm=llm,
)

# 定义「风险顾问」Agent
#
# 职责边界：审查前面所有环节产出的方案，指出风险敞口并给出对冲/规避建议。
# 放在最后一位，因为风控结论需要基于前序所有输出。
risk_management_agent = Agent(
    role="Risk Advisor",
    goal="Evaluate and provide insights on the risks "
    "associated with potential trading activities.",
    backstory="Armed with a deep understanding of risk assessment models "
    "and market dynamics, this agent scrutinizes the potential "
    "risks of proposed trades. It offers a detailed analysis of "
    "risk exposure and suggests safeguards to ensure that "
    "trading activities align with the firm’s risk tolerance.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool],
    llm=llm,
)

# ~~~~~~~~~~~~~~~ Creating Tasks ~~~~~~~~~~~~~~~~~~~~~~
# Task for Data Analyst Agent: Analyze Market Data
#
# Task 的两个核心字段：
#   - description：做什么（会被填入实际输入变量后发给 Agent）；
#   - expected_output：要产出什么（决定 Manager 如何验收这个 Task 的结果）。
#
# 注意 description 里的 {stock_selection} 等占位符，
# 会在 kickoff(inputs=...) 时被真实值替换。
data_analysis_task = Task(
    description=(
        "Continuously monitor and analyze market data for "
        "the selected stock ({stock_selection}). "
        "Use statistical modeling and machine learning to "
        "identify trends and predict market movements."
    ),
    expected_output=(
        "Insights and alerts about significant market "
        "opportunities or threats for {stock_selection}."
    ),
    agent=data_analyst_agent,
)

# Task for Trading Strategy Agent: Develop Trading Strategies
#
# 这个 Task 同时消费了前面 Task 的产出（"insights from the Data Analyst"）、
# 以及用户直接给出的约束（risk_tolerance、trading_strategy_preference），
# 体现了 hierarchical 流程中「Agent 之间可以传递上下文」的特点。
strategy_development_task = Task(
    description=(
        "Develop and refine trading strategies based on "
        "the insights from the Data Analyst and "
        "user-defined risk tolerance ({risk_tolerance}). "
        "Consider trading preferences ({trading_strategy_preference})."
    ),
    expected_output=(
        "A set of potential trading strategies for {stock_selection} "
        "that aligns with the user's risk tolerance."
    ),
    agent=trading_strategy_agent,
)

# Task for Trade Advisor Agent: Plan Trade Execution
execution_planning_task = Task(
    description=(
        "Analyze approved trading strategies to determine the "
        "best execution methods for {stock_selection}, "
        "considering current market conditions and optimal pricing."
    ),
    expected_output=(
        "Detailed execution plans suggesting how and when to "
        "execute trades for {stock_selection}."
    ),
    agent=execution_agent,
)

# Task for Risk Advisor Agent: Assess Trading Risks
risk_assessment_task = Task(
    description=(
        "Evaluate the risks associated with the proposed trading "
        "strategies and execution plans for {stock_selection}. "
        "Provide a detailed analysis of potential risks "
        "and suggest mitigation strategies."
    ),
    expected_output=(
        "A comprehensive risk analysis report detailing potential "
        "risks and mitigation recommendations for {stock_selection}."
    ),
    agent=risk_management_agent,
)

# ~~~~~~~~~~~~~~ Creating the Crew~~~~~~~~~~~~~~~~~~
# Define the crew with agents and tasks
#
# process=Process.hierarchical：层级式流程
#   - 由 manager_llm（这里复用同一个 DeepSeek）担任 Manager，负责：
#     1) 把整体目标拆解成子任务；
#     2) 判断每个子任务该交给哪个 Agent；
#     3) 汇总各 Agent 的产出，形成最终报告。
#   - 对比 Process.sequential：严格按 tasks 列表顺序串行执行，不做动态拆解。
#
# 层级式流程的代价是 token 消耗明显更高（Manager 要反复思考委派），
# 所以 max_tokens 必须给足，否则 delegate_work_to_coworker 的参数会被截断。
financial_trading_crew = Crew(
    agents=[
        data_analyst_agent,
        trading_strategy_agent,
        execution_agent,
        risk_management_agent,
    ],
    tasks=[
        data_analysis_task,
        strategy_development_task,
        execution_planning_task,
        risk_assessment_task,
    ],
    manager_llm=llm,
    process=Process.hierarchical,
    verbose=True,
)

# Example data for kicking off the process
financial_trading_inputs = {
    "stock_selection": "AAPL",
    "initial_capital": "100000",
    "risk_tolerance": "Medium",
    "trading_strategy_preference": "Day Trading",
    "news_impact_consideration": True,
}
logger.info("Crew 输入参数：%s", financial_trading_inputs)

# ~~~~~~~~~~~~~~ Running the Crew ~~~~~~~~~~~~~~~~~~
# hierarchical 流程：Manager Agent 负责拆解与委派，四个专业 Agent 各执行一个 Task。
logger.info("开始执行 hierarchical 流程（Crew 会自动委派任务给合适的 Agent）")
try:
    result = financial_trading_crew.kickoff(inputs=financial_trading_inputs)
except Exception as exc:  # noqa: BLE001 - 演示脚本，捕获后打印完整堆栈便于排查
    # 常见失败原因：本地代理未启动、模型超时、委派过程超过 max_tokens 限制
    logger.error("Crew 执行失败：%s", exc)
    logger.exception("Crew 执行异常详情")
    raise

logger.info("Crew 执行完成，最终输出长度=%d 字符", len(result.raw))
print("Final result ===> ", result)

file_name = r"output.md"  # Name of the Markdown file
with open(file_name, "w", encoding="utf-8") as file:
    file.write(result.raw)
logger.info("Markdown 结果已写入 %s（%d 字节）", file_name, os.path.getsize(file_name))

print(f"Markdown content has been saved to {file_name}")
