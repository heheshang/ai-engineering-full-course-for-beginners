"""文档加载器演示：CSV（Document Loaders - CSV）。

本例演示 LangChain 的 **Document Loader（文档加载器）** 是什么、怎么用，
以及它在整个 RAG 链路中的位置。

核心概念：
    Document Loader 负责把**非结构化/半结构化数据**（CSV、PDF、Markdown、HTML、
    数据库、Notion、邮件……）统一转换成 LangChain 的 ``Document`` 对象。

    ``Document`` 只有两个核心字段：
      - ``page_content``：正文内容（字符串），最终会被切块、向量化、送进 LLM；
      - ``metadata``：元数据（字典），记录来源、行号、页码、创建时间等，
        用于**溯源、过滤、权限控制**。metadata 里的 ``source`` 字段尤其重要，
        RAG 回答里给出的引用链接就是从它来的。

    在 RAG 流水线中的位置：
        加载（Load） -> 切块（Split/Chunk） -> 向量化（Embed） -> 存储（Store）
        -> 检索（Retrieve） -> 生成（Generate）
        本例负责第一步"加载"。

关于导入路径（LangChain 1.x 的重要变化）：
    ``langchain_core.document_loaders`` 在 1.x 里**只有** BaseLoader / Blob 等
    基础抽象，**不包含** CSVLoader 这类具体格式的加载器。

    具体格式的加载器有两个来源：
      1) ``langchain_community.document_loaders`` —— 社区维护的通用加载器合集；
      2) ``langchain_classic.document_loaders`` —— 1.x 拆分出来的旧版经典 API，
         里面的 CSVLoader 实际仍然指向 langchain_community 的实现。

    注意：``langchain-community`` 官方已宣布进入 sunset（停止积极维护），
    未来可能迁移到独立集成包。目前仍可用，但新项目应优先考虑
    ``langchain-classic`` 或各厂商的独立集成包。

    另外，Python 内置的 ``csv`` 模块就能读 CSV，不一定非要 LangChain；
    LangChain 的价值在于**统一接口**：所有 loader 都返回 ``List[Document]``，
    可以和 ``DirectoryLoader`` / ``RecursiveCharacterTextSplitter`` /
    向量库无缝串起来。
"""

import csv
import json
import tempfile
from pathlib import Path

import sys
from pathlib import Path

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.logging_utils import get_logger

# 初始化 logger：日志同时输出到控制台和 logs/course.log
logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# 0. 准备工作
# --------------------------------------------------------------------------- #
# 用 Path 而不是拼字符串路径，好处：
#   - 跨平台（Windows 用 / 也能工作）；
#   - BASE_DIR 通过 __file__ 推导，无论从哪个工作目录运行脚本都能找到数据文件；
#   - / 运算符拼接路径更直观。
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "products.csv"

# --------------------------------------------------------------------------- #
# 1. 方式一：CSVLoader（LangChain 统一接口）
# --------------------------------------------------------------------------- #
from langchain_community.document_loaders import CSVLoader  # noqa: E402

print("=" * 70)
print("1) 方式一：CSVLoader —— 默认把每一行渲染成 '字段: 值' 的文本")
print("=" * 70)

# CSVLoader 的默认行为：
#   - **一行 CSV = 一个 Document**；
#   - 该行的所有字段被渲染成 "key: value" 的多行文本放进 page_content；
#   - metadata 里自动带上 {"source": 文件路径, "row": 行号}（从 0 开始）。
loader = CSVLoader(file_path=str(CSV_PATH))

# load() 一次性把整个文件读进内存，返回 List[Document]。
# 大文件场景应该用 lazy_load()，返回迭代器，逐行处理不占内存。
documents = loader.load()
logger.info("CSVLoader 加载完成，共 %d 个 Document（每行一个）", len(documents))

print(f"共加载 {len(documents)} 个 Document\n")

# 只展示第一条，省得刷屏
first_doc = documents[0]
print("--- documents[0].page_content ---")
print(first_doc.page_content)
print("\n--- documents[0].metadata ---")
print(json.dumps(first_doc.metadata, indent=2, ensure_ascii=False))
logger.info("首条 Document 的 metadata：%s", first_doc.metadata)

# 打印一条最后一条，方便确认行号递增
print("\n--- documents[-1].metadata ---")
print(json.dumps(documents[-1].metadata, indent=2, ensure_ascii=False))

# --------------------------------------------------------------------------- #
# 2. 方式二：自定义 csv_args —— 控制分隔符、引号、表头
# --------------------------------------------------------------------------- #
# csv_args 会原样透传给 Python 内置的 csv 模块，常用选项：
#   delimiter    分隔符，默认 ','；读 TSV 用 '\t'
#   quotechar    引号字符，默认 '"'
#   fieldnames   手动指定表头；指定后**第一行会被当作普通数据行**，
#                所以一般用于"无表头文件"
#   encoding     文件编码（注意：这个要传给 CSVLoader 的 encoding 参数，
#                不是 csv_args）
print("\n" + "=" * 70)
print("2) 方式二：指定 csv_args（以 TSV / 制表符分隔为例）")
print("=" * 70)

tsv_path = Path(tempfile.gettempdir()) / "products_demo.tsv"
# 用内置 csv 模块生成一个 TSV 文件，顺便演示分隔符的差异
# （TSV 写到临时目录，避免在项目里留下演示产生的垃圾文件）

with open(CSV_PATH, encoding="utf-8") as f:
    rows = list(csv.reader(f))

with open(tsv_path, "w", encoding="utf-8", newline="") as f:
    csv.writer(f, delimiter="\t").writerows(rows)

tsv_loader = CSVLoader(file_path=str(tsv_path), csv_args={"delimiter": "\t"})
tsv_documents = tsv_loader.load()
logger.info("TSV 加载完成，共 %d 个 Document（分隔符改为 \\t）", len(tsv_documents))

print(f"TSV 加载 {len(tsv_documents)} 个 Document")
print("--- documents[0].page_content ---")
print(tsv_documents[0].page_content)

# --------------------------------------------------------------------------- #
# 3. 方式三：source_column —— 让 metadata.source 指向业务字段而不是文件路径
# --------------------------------------------------------------------------- #
# 默认情况下所有 Document 的 metadata["source"] 都是同一个文件路径，
# 无法区分"这条记录来自哪一行/哪条数据"。
# 指定 source_column 后，source 取自该列的值，溯源和按来源过滤就都能用了。
print("\n" + "=" * 70)
print("3) 方式三：source_column —— 把 source 指向业务字段")
print("=" * 70)

source_loader = CSVLoader(file_path=str(CSV_PATH), source_column="name")
source_documents = source_loader.load()
logger.info("source_column='name' 加载完成，共 %d 个 Document（metadata.source 指向业务字段）",
             len(source_documents))

print("--- 前 3 条的 metadata ---")
for doc in source_documents[:3]:
    print(json.dumps(doc.metadata, ensure_ascii=False))

# --------------------------------------------------------------------------- #
# 4. 方式四：metadata_columns —— 抽取列到 metadata（不进正文）
# --------------------------------------------------------------------------- #
# metadata_columns 里列出的字段会从 page_content 中**移除**，
# 只保留在 metadata 中。适合那些"检索时用来过滤，但不该占 token"的字段，
# 比如 category（类别）、stock（库存）。
#
# 对比一下 page_content 的差别：下面这行的正文里不再有 category 和 stock。
print("\n" + "=" * 70)
print("4) 方式四：metadata_columns —— 把字段抽到 metadata（不进正文）")
print("=" * 70)

meta_loader = CSVLoader(
    file_path=str(CSV_PATH),
    metadata_columns=["category", "stock", "price"],
)
meta_documents = meta_loader.load()
logger.info("metadata_columns 模式加载完成，共 %d 个 Document（category/stock/price 已抽到 metadata）",
             len(meta_documents))

print("--- documents[0].page_content ---")
print(meta_documents[0].page_content)
print("\n--- documents[0].metadata ---")
print(json.dumps(meta_documents[0].metadata, indent=2, ensure_ascii=False))

# --------------------------------------------------------------------------- #
# 5. 方式五：content_columns —— 反过来，只让指定列进正文
# --------------------------------------------------------------------------- #
# content_columns 指定"只有这些列的内容进 page_content"，
# 与 metadata_columns 是互补的两个方向。
print("\n" + "=" * 70)
print("5) 方式五：content_columns —— 只让指定列进正文")
print("=" * 70)

content_loader = CSVLoader(
    file_path=str(CSV_PATH),
    content_columns=["name", "description"],
)
content_documents = content_loader.load()
logger.info("content_columns 模式加载完成，共 %d 个 Document（正文只含 name/description）",
             len(content_documents))

print("--- documents[0].page_content ---")
print(content_documents[0].page_content)
print("\n--- documents[0].metadata ---")
print(json.dumps(content_documents[0].metadata, indent=2, ensure_ascii=False))

# --------------------------------------------------------------------------- #
# 6. 方式六：lazy_load —— 流式加载大文件
# --------------------------------------------------------------------------- #
# load() 会把整个文件读进内存；文件很大时用 lazy_load()，
# 返回一个生成器，边遍历边产出 Document，内存占用恒定。
print("\n" + "=" * 70)
print("6) 方式六：lazy_load —— 流式加载（适合大文件）")
print("=" * 70)

lazy_loader = CSVLoader(file_path=str(CSV_PATH))
count = 0
for doc in lazy_loader.lazy_load():
    count += 1
    if count <= 2:
        print(f"[lazy] 第 {count} 条 -> {doc.metadata}")
logger.info("lazy_load 流式迭代完成，共产出 %d 个 Document（内存占用恒定）", count)
print(f"lazy_load 共迭代出 {count} 个 Document（内存占用恒定）")

# --------------------------------------------------------------------------- #
# 7. 方式七：不用 LangChain，直接用内置 csv 模块
# --------------------------------------------------------------------------- #
# 演示一下"为什么 LangChain 有价值"：
#   - 内置 csv 模块更快、更轻，但返回的是 dict 列表，需要自己转 Document；
#   - CSVLoader 统一了接口，能直接对接切块器/向量库。
print("\n" + "=" * 70)
print("7) 方式七：内置 csv 模块（对照）")
print("=" * 70)

with open(CSV_PATH, encoding="utf-8") as f:
    raw_rows = list(csv.DictReader(f))
logger.info("内置 csv.DictReader 读到 %d 条 dict", len(raw_rows))

print(f"内置 csv.DictReader 读到 {len(raw_rows)} 条 dict")
print(json.dumps(raw_rows[0], indent=2, ensure_ascii=False))

# --------------------------------------------------------------------------- #
# 8. 汇总
# --------------------------------------------------------------------------- #
print("\n" + "=" * 70)
print("8) 各加载方式对比")
print("=" * 70)

summary = [
    ("CSVLoader 默认", len(documents), "整行渲染成 key: value 文本"),
    ("csv_args(delimiter)", len(tsv_documents), "自定义分隔符/引号/表头"),
    ("source_column", len(source_documents), "source 指向业务字段，便于溯源"),
    ("metadata_columns", len(meta_documents), "字段抽到 metadata，不进正文"),
    ("content_columns", len(content_documents), "只让指定列进正文"),
    ("lazy_load", count, "流式迭代，内存恒定"),
    ("内置 csv 模块", len(raw_rows), "返回 dict，需自己转 Document"),
]

for name, n, desc in summary:
    print(f"  {name:<22} {n:>3} 个  | {desc}")

# --------------------------------------------------------------------------- #
# 附：把 Loader 接进 RAG 流水线
# --------------------------------------------------------------------------- #
# 下一步就是切块 + 向量化 + 入库，示例：
#
#   from langchain_text_splitters import RecursiveCharacterTextSplitter
#   from langchain_chroma import Chroma
#   from langchain_huggingface import HuggingFaceEmbeddings
#
#   splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
#   chunks = splitter.split_documents(documents)     # List[Document]
#   vectorstore = Chroma.from_documents(chunks, HuggingFaceEmbeddings(...))
#
# 关键点：loader 输出的 Document 里的 **metadata 会一路传递到最终 chunk**，
# 所以在 loader 阶段就把 source / row 整理好，后面做引用溯源就不用返工了。
