import os
from datetime import datetime

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

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY")
if not api_key:
    raise RuntimeError(
        "未找到 API Key，请在 .env 中设置 DEEPSEEK_API_KEY（或 API_KEY）"
    )
os.environ["DEEPSEEK_API_KEY"] = api_key

# 1. 初始化模型
model = init_chat_model("deepseek:deepseek-chat", temperature=0.8, api_key=api_key)

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
        [
            {"obs_id": o[0], "stock_ticker": o[1], "price": o[2], "date": o[3]}
            for o in observations
        ]
    )
    stmt = stmt.on_conflict_do_nothing(index_elements=["obs_id"])
    conn.execute(stmt)

db = SQLDatabase(engine)

# 3. 构建 SQL 工具与 Agent（新写法）
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

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
config = {"recursion_limit": 3}
result = agent.invoke({"messages": [{"role": "user", "content": question}]})
final_answer = result["messages"][-1].content
print("Answer:", final_answer)

# 5. Token 计数（正确方式）
# 若只需统计输入问题的 token 数：
token_count = model.get_num_tokens(question)
print("Input token count:", token_count)

# 若需获取完整调用消耗的 token（含工具调用、推理等），可从最终消息元数据读取：
usage = result["messages"][-1].usage_metadata
print("Usage:", usage)
