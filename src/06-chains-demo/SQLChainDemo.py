"""链（Chain）演示 8：SQL 链（Text-to-SQL）——LangChain 1.x 现代写法。

背景
----
课程里的旧写法是：

    from langchain.chains import SQLDatabaseChain      # ❌ LangChain 1.x 已彻底移除
    chain = llm | SQLDatabaseChain.from_llm(llm=llm, db=db)
    response = chain.invoke({"question": "How many employees are there?"})

``SQLDatabaseChain`` 在 1.x 中**连 ``langchain_classic`` 都没有了**
（实测 ``from langchain_classic.chains import SQLDatabaseChain`` 会抛 ImportError），
旧的 ``SQLDatabaseChain`` 内部流程是「生成 SQL → 执行 → 再让模型总结」三步黑盒。

本文件用 **LCEL（表达式语言）** 显式重写这条链，并把每一步都摊开给你看。
同时给出第二条**更现代、更可控**的实现路线：**``create_agent`` + SQL 工具**。

两条路线对比：

    ┌───────────────────── 路线 A：LCEL 显式链 ─────────────────────┐
    │  问题 → [生成 SQL] → [执行 SQL] → [总结成自然语言] → 答案      │
    │  特点：流程固定、可预测、易调试；每步都能单独替换/观测          │
    └──────────────────────────────────────────────────────────────┘

    ┌───────────────────── 路线 B：Agent + 工具 ───────────────────┐
    │  问题 → 模型自主决定调哪个工具 → 工具结果回灌 → 模型继续 → 答案 │
    │  特点：模型可先看表结构再决定查什么，能自我纠错；             │
    │        代价是行为不如固定链可控，token 消耗更高                │
    └──────────────────────────────────────────────────────────────┘

依赖：pip install langchain langchain-deepseek langchain-community sqlalchemy
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

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

# 读取 Key：优先 DEEPSEEK_API_KEY，其次兼容通用命名 API_KEY；
# 都没有就立刻报错，避免等到请求接口时才失败、难以排查。
api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY")
if not api_key:
    raise RuntimeError("未找到 API Key，请在 .env 中设置 DEEPSEEK_API_KEY（或 API_KEY）")
# 再显式写回环境变量，供 init_chat_model 按 provider 自动读取。
os.environ["DEEPSEEK_API_KEY"] = api_key

# --------------------------------------------------------------------------- #
# 1. 初始化 LLM
# --------------------------------------------------------------------------- #
# init_chat_model 是统一的模型工厂函数：
#   - provider 决定用哪家的客户端（deepseek / openai / anthropic ...）
#   - model    是具体模型名
#
# temperature=0：Text-to-SQL 要求输出尽量确定，避免同一个问题生成不同 SQL。
llm = init_chat_model(
    "deepseek:deepseek-chat",
    api_key=api_key,
    temperature=0,
)

# --------------------------------------------------------------------------- #
# 2. 连接数据库
# --------------------------------------------------------------------------- #
# SQLDatabase 是 SQLAlchemy 的 LangChain 封装，把「数据库」包装成
# 「模型能理解的结构化信息」，主要提供三样东西：
#
#   db.get_usable_table_names()  →  表名列表
#   db.get_table_info()          →  建表语句 + 样例数据（喂给模型写 SQL 用）
#   db.run(sql)                  →  执行 SQL 并返回结果
#
# from_uri 接受标准 SQLAlchemy 连接串：
#   sqlite:///chinook.db      → SQLite（相对路径）
#   sqlite:////abs/path.db    → SQLite（绝对路径，注意四个斜杠）
#   postgresql://user:pw@host/db → PostgreSQL
#
# ⚠️ 安全提示：生产环境务必给数据库账号设置**只读权限**，
#    并用 include_tables / ignore_tables 限定模型能看到的表。
db = SQLDatabase.from_uri("sqlite:///chinook.db")


# --------------------------------------------------------------------------- #
# 3. 工具函数：清洗模型生成的 SQL
# --------------------------------------------------------------------------- #
def clean_sql(raw_sql: str) -> str:
    """把模型输出的 SQL 清洗成可以直接执行的形式。

    模型很爱给 SQL 裹一层 markdown（```sql ... ```），或者在结尾多个分号，
    直接丢给数据库会报语法错误，所以这里统一剥掉。

    同时用正则只保留第一条语句，防止模型一次生成多条 SQL
    （既是防注入，也是避免误执行 DDL/DML）。
    """
    text = raw_sql.strip()
    # 去掉 markdown 代码围栏（可能是 ```sql 或 ```）
    if "```" in text:
        text = text.split("```")[1] if text.count("```") >= 2 else text.replace("```", "")
        # 去掉 ``` 后面可能带的语言标识（如 sql）
        if "\n" in text:
            first, rest = text.split("\n", 1)
            text = rest if first.strip().lower() in {"sql", "sqlite", "postgresql"} else text
    text = text.strip()
    # 只保留第一条语句
    text = text.split(";")[0]
    return text.strip()


# --------------------------------------------------------------------------- #
# 4. 路线 A：用 LCEL 显式实现「问题 → SQL → 结果 → 答案」
# --------------------------------------------------------------------------- #
# 4.1 写「生成 SQL」的提示词
# --------------------------------------------------------------------------- #
# 模板里的三个变量都由后面的 RunnablePassthrough.assign() 注入：
#   {question}   用户问题
#   {table_info} 表结构 + 样例数据（db.get_table_info() 的结果）
#   {dialect}    数据库方言（sqlite / postgresql ...），语法有差异必须告诉模型
#
# 提示词要点：
#   - 限定只能用给出的表，防止模型臆造表名
#   - 强调只生成 SELECT，避免模型写出 DELETE/UPDATE
#   - 提醒留意样例数据里的枚举值（如 'Paid' / 'Unpaid'）
sql_prompt = ChatPromptTemplate.from_template(
    """You are a SQL expert. Given a question, write a single SQLite query that answers it.

Rules:
- Only use the tables listed below.
- Write ONE query only, and only a SELECT statement.
- Do not explain anything, output the raw SQL query only.

Table info:
{table_info}

Dialect: {dialect}

Question: {question}
SQLQuery:"""
)

# 4.2 写「把 SQL 结果总结成自然语言」的提示词
# --------------------------------------------------------------------------- #
# 这一步很关键：模型拿着原始表格（如 [(3,)]）自己翻译成「共有 3 名员工」，
# 比直接把原始结果丢给用户友好得多。
answer_prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the SQL query result below.
If the result is empty, say that there is no matching record.
If the result is an error message, explain what went wrong instead of guessing.

Question: {question}
SQLQuery: {sql}
SQLResult: {context}

Answer:"""
)

# 4.3 组装「生成 SQL」子链
# --------------------------------------------------------------------------- #
# RunnablePassthrough.assign(...) 会把「额外的键」加到输入字典上：
#   输入 {"question": ...}  →  输出 {"question": ..., "table_info": ..., "dialect": ...}
#
# lambda 是无状态的即时计算函数：
#   lambda x: db.get_table_info()  每次调用时实时取表结构
#   lambda x: db.dialect           取数据库方言
#
# 管道用 | 串起来：
#   assign(补充变量) → sql_prompt(渲染提示词) → llm(生成) → StrOutputParser(取纯文本)
#   → clean_sql(清洗掉 markdown 代码块)
sql_chain = (
    RunnablePassthrough.assign(
        table_info=lambda x: db.get_table_info(),
        dialect=lambda x: db.dialect,
    )
    | sql_prompt
    | llm
    | StrOutputParser()
    | clean_sql
)

# 4.4 组装完整 Text-to-SQL 链
# --------------------------------------------------------------------------- #
# 注意这里是**顺序依赖**的三段，不能像 RAG 那样用字典分支并行：
#   第 2 段要用到第 1 段生成的 SQL，第 3 段要用到第 2 段执行的结果。
# 所以必须用 RunnablePassthrough.assign() 一层层往下加字段。
#
#   {"question": ...}
#      │ assign(sql=sql_chain)      → 加上生成的 SQL
#      ▼
#   {"question": ..., "sql": "SELECT ..."}
#      │ assign(context=...)       → 执行 SQL，拿到结果
#      ▼
#   {"question": ..., "sql": ..., "context": "[(3,)]"}
#      ▼
#   answer_prompt → llm → StrOutputParser()
#      ▼
#   "There are 3 employees."
#
# 这里用 db.run_no_throw() 而不是 db.run()，两者行为不同：
#   - db.run()          执行失败会**直接抛异常**，整个程序崩掉
#   - db.run_no_throw() 执行失败把错误信息**当作结果返回**（一段错误文本）
# 用 no_throw 的好处：错误会变成模型看得见的文本，
# 模型下一轮就能自己改写 SQL 重试；代价是必须检查输出，
# 别把 "near \"SELCT\": syntax error" 当成答案返回给用户。
chain = (
    RunnablePassthrough.assign(sql=sql_chain)
    | RunnablePassthrough.assign(context=lambda x: db.run_no_throw(x["sql"]))
    | answer_prompt
    | llm
    | StrOutputParser()
)

# --------------------------------------------------------------------------- #
# 5. 调用路线 A
# --------------------------------------------------------------------------- #
question = "How many employees are there?"
logger.info("开始执行路线 A（LCEL 固定链），问题：%s", question)

# 单独跑一下 sql_chain，看看模型到底生成了什么 SQL（调试时非常有用）
generated_sql = sql_chain.invoke({"question": question})
logger.info("模型生成的 SQL：%s", generated_sql)
print("=" * 70)
print("1) 模型生成的 SQL")
print("=" * 70)
print(generated_sql)

# 再看执行结果
sql_result = db.run_no_throw(generated_sql)
logger.info("SQL 执行结果：%s", sql_result)
print("\n" + "=" * 70)
print("2) SQL 执行结果")
print("=" * 70)
print(sql_result)

# 最后跑完整链，拿最终答案
answer = chain.invoke({"question": question})
logger.info("路线 A 完成，最终答案：%s", answer)
print("\n" + "=" * 70)
print("3) 最终答案（路线 A：LCEL 固定链）")
print("=" * 70)
print(answer)

# --------------------------------------------------------------------------- #
# 6. 路线 B：用 create_agent + SQL 工具（更现代、可自我纠错）
# --------------------------------------------------------------------------- #
# 官方 1.x 推荐思路：不再用「一条固定的黑盒链」，
# 而是给模型一组「SQL 工具」，让模型通过 function calling 自己决定怎么查。
#
# 提供的 4 个工具（来自 langchain_community.tools.sql_database.tool）：
#
#   sql_db_list_tables   → 列出所有表名
#   sql_db_schema        → 查看指定表的结构（等价于 get_table_info）
#   sql_db_query         → 执行任意 SQL，返回结果或错误信息
#   sql_db_query_checker → 用 LLM 检查 SQL 语法/逻辑是否正确
#
# 模型的工作流（可能多轮）：
#   1. 先调 sql_db_list_tables 看看有哪些表
#   2. 再调 sql_db_schema 看 Employee 表长什么样
#   3. 生成 SQL，调 sql_db_query 执行
#   4. 若报错，调 sql_db_query_checker 看看哪里错了，改写后重试
#   5. 最后用自然语言总结结果
from langchain.agents import create_agent
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDatabaseTool,
)

sql_tools = [
    ListSQLDatabaseTool(db=db),
    InfoSQLDatabaseTool(db=db),
    QuerySQLDatabaseTool(db=db),
    # QuerySQLCheckerTool 需要一个 llm 来做「SQL 检查」，必须显式传入
    QuerySQLCheckerTool(db=db, llm=llm),
]

sql_agent = create_agent(
    model=llm,
    tools=sql_tools,
    system_prompt=(
        "You are a SQLite expert. Use the provided SQL tools to answer the question. "
        "First list the tables, then inspect the relevant schema, then run a SELECT "
        "query with `sql_db_query`. Never modify data. "
        "Finally, answer in natural language with the concrete numbers."
    ),
)
logger.info("SQL Agent 创建完成，已注册 %d 个工具：%s", len(sql_tools), [t.name for t in sql_tools])

# 调用 Agent：返回的是状态字典 {"messages": [...]}
logger.info("开始执行路线 B（Agent + SQL 工具）")
agent_result = sql_agent.invoke({"messages": [("user", question)]})
logger.info("路线 B 完成，共产生 %d 条消息", len(agent_result["messages"]))

print("\n" + "=" * 70)
print("4) Agent 路线：完整对话过程")
print("=" * 70)
for message in agent_result["messages"]:
    # tool_calls 里有内容说明这是一次「工具调用」消息
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        for call in tool_calls:
            logger.info("Agent 调用工具：%s(%s)", call["name"], call["args"])
            print(f"  → 调用工具 {call['name']}({call['args']})")
    else:
        logger.debug("Agent 消息：type=%s", message.type)
        print(f"  · {message.type}: {message.content}")

print("\n" + "=" * 70)
print("5) 最终答案（路线 B：Agent + 工具）")
print("=" * 70)
# messages[-1] 就是模型的最终回答
agent_answer = agent_result["messages"][-1].content
logger.info("路线 B 最终答案：%s", agent_answer)
print(agent_answer)

# --------------------------------------------------------------------------- #
# 7. 常见坑 / 进阶方向
# --------------------------------------------------------------------------- #
# 7.1 ``SQLDatabaseChain`` 已彻底移除，不要再 import 它。
#     ``create_sql_query_chain``（在 langchain_classic.chains 里）只负责「生成 SQL」，
#     不负责执行和总结，适合拆开复用，但不是 1.x 的推荐主路线。
# 7.2 模型生成的 SQL 常带 markdown 包裹（```sql ... ```）或结尾分号，
#     直接执行会报错。本文件用 clean_sql() 统一清洗。
# 7.3 db.run() 抛异常、db.run_no_throw() 返回错误文本，两者别用混：
#     固定链用 no_throw 便于把错误喂回模型；Agent 路线里
#     QuerySQLDatabaseTool 内部已经处理了错误并返回给模型。
# 7.4 一定要给模型完整的表结构（db.get_table_info() 带样例数据），
#     否则模型不知道枚举字段的取值，生成的 WHERE 条件经常是空的。
# 7.5 安全是第一要务：
#     - 数据库账号用只读权限
#     - include_tables=[...] 限定可见表
#     - 绝不要把用户原始输入直接拼进 SQL（提示词注入风险）
#     - clean_sql() 里只保留第一条语句，防多语句注入
# 7.6 想换成 PostgreSQL 只需改 from_uri 的连接串，链本身不用动。
