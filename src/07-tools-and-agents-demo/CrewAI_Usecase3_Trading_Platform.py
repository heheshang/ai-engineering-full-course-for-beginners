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

load_dotenv()

# ---------------------------------------------------------------------------
# 0. 环境准备
# ---------------------------------------------------------------------------
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    raise RuntimeError("未找到 API Key，请在 .env 中设置 DEEPSEEK_API_KEY")

serper_api_key = os.getenv("SERPER_API_KEY")
if serper_api_key:
    os.environ["SERPER_API_KEY"] = serper_api_key

# ---------------------------------------------------------------------------
# 1. 初始化 DeepSeek 模型（经由本地 OpenAI 兼容代理）
# ---------------------------------------------------------------------------
llm = LLM(
    model="openai/deepseek-chat",
    base_url=os.getenv("DEEPSEEK_PROXY_URL", "http://127.0.0.1:8765/v1"),
    api_key=deepseek_api_key,
    temperature=0.0,
    # 这里必须给足：多智能体协作会产生很长的 tool-call arguments
    # （delegate_work_to_coworker 常常超过 2000 tokens），
    # 截断会导致 JSON 解析失败。代理侧也有自动重试兜底。
    max_tokens=8000,
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
                results = list(
                    DDGS().text(
                        search_query,
                        max_results=self.max_results,
                        backend=backend,
                    )
                )
                if results:
                    return results
                errors.append(f"{backend}: no results")
            except Exception as exc:  # pragma: no cover - 网络异常兜底
                errors.append(f"{backend}: {exc}")
        raise RuntimeError("; ".join(errors))

    def _run(self, search_query: str) -> str:
        try:
            results = self._search(search_query)
        except Exception as exc:  # pragma: no cover - 网络异常兜底
            return f"Search failed: {exc}"

        if not results:
            return f"No results found for '{search_query}'."

        lines = []
        for item in results:
            lines.append(
                f"- {item.get('title', '')}\n  URL: {item.get('href', '')}\n"
                f"  {item.get('body', '')}"
            )
        return f"Search results for '{search_query}':\n" + "\n".join(lines)


search_tool: BaseTool = SerperDevTool() if serper_api_key else DuckDuckGoSearchTool()
scrape_tool = ScrapeWebsiteTool()

# ~~~~~~~~~~~Creating Agents ~~~~~~~~~~~~~~~
# Define Data Analyst Agent
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

# Define Trading Strategy Developer Agent
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

# Define Trade Advisor Agent
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

# Define Risk Advisor Agent
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

# ~~~~~~~~~~~~~~ Running the Crew ~~~~~~~~~~~~~~~~~~
result = financial_trading_crew.kickoff(inputs=financial_trading_inputs)

print("Final result ===> ", result)

file_name = r"output.md"  # Name of the Markdown file
with open(file_name, "w", encoding="utf-8") as file:
    file.write(result.raw)

print(f"Markdown content has been saved to {file_name}")
