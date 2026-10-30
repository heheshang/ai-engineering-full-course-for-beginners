import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import (
    Column,
    Date,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
)
from sqlalchemy.dialects.sqlite import insert  # 关键：必须用这个 insert

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.logging_utils import get_logger

# 初始化 logger：日志同时输出到控制台和 logs/course.log
logger = get_logger(__name__)


load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY")
if not api_key:
    raise RuntimeError("未找到 API Key，请在 .env 中设置 DEEPSEEK_API_KEY（或 API_KEY）")
os.environ["DEEPSEEK_API_KEY"] = api_key

# 1. 初始化模型
model = init_chat_model("deepseek:deepseek-chat", temperature=0.8, api_key=api_key)
logger.info("LLM 初始化完成：deepseek:deepseek-chat, temperature=0.8")

# 2. 数据库与数据初始化（与原代码一致）
metadata_obj = MetaData()
stocks = Table(
    "stocks",
    metadata_obj,
    Column("obs_id", Integer, primary_key=True),
    Column("stock_ticker", String(4), nullable=False),
    Column("price", Float, nullable=False),
    Column("date", Date, nullable=False),
)
engine = create_engine("sqlite:///stocks.db")
metadata_obj.create_all(engine)
logger.info("数据库表结构就绪：sqlite:///stocks.db，表=stocks")

observations = [
    [1, "ABC", 200, datetime(2050, 1, 1)],
    [2, "ABC", 208, datetime(2050, 1, 2)],
    [3, "ABC", 232, datetime(2050, 1, 3)],
    [4, "ABC", 225, datetime(2050, 1, 4)],
    [5, "ABC", 226, datetime(2050, 1, 5)],
    [6, "XYZ", 810, datetime(2050, 1, 1)],
    [7, "XYZ", 803, datetime(2050, 1, 2)],
    [8, "XYZ", 798, datetime(2050, 1, 3)],
    [9, "XYZ", 795, datetime(2050, 1, 4)],
    [10, "XYZ", 791, datetime(2050, 1, 5)],
]

# with engine.begin() as conn:
#     conn.execute(
#         insert(stocks),
#         [
#             {"obs_id": o[0], "stock_ticker": o[1], "price": o[2], "date": o[3]}
#             for o in observations
#         ],
#     )
with engine.begin() as conn:
    stmt = insert(stocks).values(
        [{"obs_id": o[0], "stock_ticker": o[1], "price": o[2], "date": o[3]} for o in observations]
    )
    stmt = stmt.on_conflict_do_nothing(index_elements=["obs_id"])
    conn.execute(stmt)
# on_conflict_do_nothing：主键冲突时跳过，保证脚本可重复运行而不会报唯一约束错误
logger.info("样例数据写入完成：共 %d 条观测记录（冲突已自动忽略）", len(observations))

db = SQLDatabase(engine)
logger.info("SQLDatabase 包装完成：方言=%s", db.dialect)

# 3. 构建 SQL 工具与 Agent（新写法）
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()
logger.info("SQL 工具集构建完成，共 %d 个工具：%s", len(tools), [t.name for t in tools])

system_prompt = """You are an agent designed to interact with a SQL database.
Given a question, create a syntactically correct SQLite query, execute it, and return the answer.
Always look at the tables first. Only use SELECT queries. Never modify data."""

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
)

# 4. 执行查询
question = (
    "What is the multiplication of the ratio between stock prices for 'ABC' and 'XYZ' "
    "on January 3rd and the ratio between the same stock prices on January the 4th?"
)
# recursion_limit=3：限制 LangGraph 的递归步数。
# Agent 需要「看表结构 -> 生成 SQL -> 执行 -> 总结」共 3~4 步，
# 设得太小会中途报错终止，太大则可能在出错时空转、浪费 token。
config = {"recursion_limit": 3}
logger.info("开始执行 SQL Agent 查询：%s", question)
try:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
except Exception as exc:
    # 常见失败原因：recursion_limit 太小导致步骤不够、模型生成了非法 SQL
    logger.error("SQL Agent 查询失败：%s", exc)
    logger.exception("SQL Agent 异常详情")
    raise

final_answer = result["messages"][-1].content
logger.info("查询完成，答案长度=%d 字符", len(final_answer))
print("Answer:", final_answer)

# 5. Token 计数（正确方式）
# 若只需统计输入问题的 token 数：
token_count = model.get_num_tokens(question)
logger.info("输入问题的 token 数：%d", token_count)
print("Input token count:", token_count)

# 若需获取完整调用消耗的 token（含工具调用、推理等），可从最终消息元数据读取：
usage = result["messages"][-1].usage_metadata
logger.info("完整调用 token 用量：%s", usage)
print("Usage:", usage)
