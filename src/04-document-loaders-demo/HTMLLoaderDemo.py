"""文档加载器演示：HTML（Document Loaders - HTML）。

本例演示如何把 HTML 网页文件转换成 LangChain 的 ``Document`` 对象，
并对比几种 HTML 加载方式的差异。

核心概念：
    HTML 加载器要解决的核心问题是：**从嵌套的标签结构里提取有意义的文本**。
    直接读文件会把 ``<div>``、``<script>``、``<style>`` 这些标签原样带进正文，
    既浪费 token 又污染向量。好的做法是：
      - 剥掉样式/脚本等非内容标签；
      - 按语义标签（``<p>``/``<h1>``/``<li>``/``<td>``）切分文本块；
      - 尽量保留标题层级（``h1``/``h2``），便于后续按标题切块。

    关键认知：LangChain 官方的 HTML 加载器**几乎都是"整页一个 Document"**，
    它们只负责"把 HTML 变成纯文本"，块级切分交给后面的 TextSplitter。
    如果你需要更细粒度（按标题/按表格行切），就得自己用 BeautifulSoup 实现。

关于可用加载器（实测 LangChain 1.x + langchain-community 0.4.2）：
    - ``BSHTMLLoader``   —— 基于 BeautifulSoup 的 ``get_text()``，**不需要额外依赖**
    - ``AsyncHtmlLoader`` —— 抓取**网络 URL**（需要网络，本例不用）
    - ``MHTMLLoader``    —— 读 Outlook 的 .mhtml 邮件存档
    - ``UnstructuredHTMLLoader`` —— 基于 unstructured，**需要额外装 unstructured**，
                                   且对表格/版式解析更细，但依赖很重

    注意：老教程里的 ``langchain.document_loaders.HTMLLoader``（lxml text_content 版）
    在当前版本**已不存在**，写示例时不要再引用它。

关于导入路径（LangChain 1.x）：
    具体格式的加载器不在 ``langchain_core`` 里，
    而在 ``langchain_community.document_loaders``（该包已进入 sunset 阶段，
    未来会拆分为独立集成包，当前仍可用）。

关于路径：
    ``BSHTMLLoader(file_path="data/titanic.html")`` 这种相对路径**依赖当前工作目录**，
    从仓库根目录运行就会报 FileNotFoundError。
    本例用 ``Path(__file__).resolve().parent`` 推导绝对路径，无论从哪运行都能找到文件。
"""

import json
import sys
from pathlib import Path

from langchain_community.document_loaders import BSHTMLLoader

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.logging_utils import get_logger

# 初始化 logger：日志同时输出到控制台和 logs/course.log
logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# 0. 准备工作
# --------------------------------------------------------------------------- #
# 用 Path 推导绝对路径，避免"相对路径依赖当前工作目录"的坑。
BASE_DIR = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR / "data" / "titanic.html"

# --------------------------------------------------------------------------- #
# 1. 方式一：BSHTMLLoader —— 官方标准用法
# --------------------------------------------------------------------------- #
# BSHTMLLoader 的行为：
#   - 底层是 BeautifulSoup 的 soup.get_text(separator)，**整页产出 1 个 Document**；
#   - metadata 自动带上 {"source": 文件路径, "title": <title> 标签内容}；
#   - get_text_separator 控制各文本节点之间的连接符：
#       默认 ""  -> 文本直接粘连，可读性差但 token 少
#       "\n"   -> 每个块之间换行，**推荐**，切块时语义更清晰
print("=" * 70)
print("1) BSHTMLLoader —— 官方标准用法（整页 1 个 Document）")
print("=" * 70)

loader = BSHTMLLoader(file_path=str(HTML_PATH), get_text_separator="\n")
docs = loader.load()
logger.info("BSHTMLLoader 加载完成，共 %d 个 Document（整页 1 个）", len(docs))

print(f"共加载 {len(docs)} 个 Document")
print("metadata：")
print(json.dumps(docs[0].metadata, indent=2, ensure_ascii=False))
logger.info("metadata=%s", docs[0].metadata)

print("\n--- page_content 前 500 字符 ---")
print(docs[0].page_content[:500])

# 对比：不传 separator 会怎样（文本粘连）
plain_docs = BSHTMLLoader(file_path=str(HTML_PATH)).load()
logger.info(
    "对比：默认 separator='' 的正文长度=%d 字符（文本直接粘连，可读性差）",
    len(plain_docs[0].page_content),
)
print("\n--- 对比：默认 get_text_separator='' 的效果（前 200 字符）---")
print(plain_docs[0].page_content[:200])

# --------------------------------------------------------------------------- #
# 2. 方式二：自定义 —— 按语义标签切分成多个 Document
# --------------------------------------------------------------------------- #
# 官方加载器只给"一坨纯文本"，丢失了标题层级。
# 真实项目里更常用的是：自己用 BeautifulSoup 按块级标签切分，
# 并把标题层级写进 metadata（后续切块器可以做 MarkdownHeaderTextSplitter 那样的处理）。
#
# 这是理解"Loader 到底该做到什么程度"的关键：
#   官方 Loader  = 格式解析（HTML -> 文本）
#   自定义 Loader = 格式解析 + 语义分段（文本 -> 结构化 chunks）
print("\n" + "=" * 70)
print("2) 方式二：自定义 —— 按语义标签切成多个 Document")
print("=" * 70)

from bs4 import BeautifulSoup
from langchain_core.documents import Document

html_text = HTML_PATH.read_text(encoding="utf-8")
soup = BeautifulSoup(html_text, "html.parser")
logger.info("HTML 解析完成，原始长度=%d 字符", len(html_text))

# 剥掉对 RAG 没价值的噪音标签
for tag in soup(["style", "script", "head", "nav", "footer"]):
    tag.decompose()
logger.info("已剥离 style/script/head/nav/footer 等噪音标签")

# 逐个块级标签提取文本
BLOCK_TAGS = ["h1", "h2", "h3", "p", "li", "td", "th", "blockquote"]
blocks: list[Document] = []
for tag in soup.find_all(BLOCK_TAGS):
    text = tag.get_text(strip=True)
    if not text:
        continue
    blocks.append(
        Document(
            page_content=text,
            metadata={
                "source": str(HTML_PATH),
                "tag": tag.name,
                "title": soup.title.string if soup.title else "",
            },
        )
    )

print(f"按语义标签切出 {len(blocks)} 个 Document")
logger.info("按语义标签切分完成，共 %d 个 Document", len(blocks))
print("\n--- 前 12 个块的 tag / 内容预览 ---")
for i, doc in enumerate(blocks[:12]):
    print(f"[{i}] <{doc.metadata['tag']}> {doc.page_content[:80]}")
    logger.debug("  块[%d] 标签=%s 长度=%d", i, doc.metadata["tag"], len(doc.page_content))

# --------------------------------------------------------------------------- #
# 3. 方式三：HTML 表格 -> DataFrame（而不是当纯文本切）
# --------------------------------------------------------------------------- #
# 很多"HTML 表格"其实是数据表，直接当纯文本切块很浪费。
# 更合理的做法是解析成 DataFrame，再转成 Markdown/CSV 交给 CSVLoader 或切块器。
print("\n" + "=" * 70)
print("3) 方式三：HTML 表格 -> DataFrame")
print("=" * 70)

import pandas as pd

tables = pd.read_html(HTML_PATH)
logger.info("从 HTML 中解析出 %d 个表格", len(tables))
print(f"从 HTML 中解析出 {len(tables)} 个表格")
df = tables[0]
print(f"表格形状：{df.shape}")
print(df.head(3).to_string(index=False))

# 转成 Markdown 文本 —— 这是 RAG 里非常常用的小技巧：
# 保留了列名和行结构，比纯文本拼接更容易被 LLM 理解。
markdown_table = df.to_markdown(index=False)
print("\n--- 表格转 Markdown（前 400 字符）---")
print(markdown_table[:400])

# --------------------------------------------------------------------------- #
# 4. 三种方式对比
# --------------------------------------------------------------------------- #
print("\n" + "=" * 70)
print("4) 三种方式对比")
print("=" * 70)

official_chars = len(docs[0].page_content)
custom_chars = sum(len(d.page_content) for d in blocks)
logger.info(
    "三种方式对比：官方=%d 文档/%d 字符，自定义=%d 文档/%d 字符，Markdown 表格=%d 字符",
    len(docs),
    official_chars,
    len(blocks),
    custom_chars,
    len(markdown_table),
)

print(f"{'方式':<28}{'Document 数':<14}{'总字符数':<12}")
print("-" * 54)
print(f"{'BSHTMLLoader（官方）':<26}{len(docs):<14}{official_chars:<12}")
print(f"{'自定义按标签切分':<25}{len(blocks):<14}{custom_chars:<12}")
print(f"{'pandas.read_html -> Markdown':<26}{1:<14}{len(markdown_table):<12}")

print("\n选型建议：")
print("  - 网页/新闻等长文本、关心语义连贯 -> BSHTMLLoader（整页文本交给切块器）")
print("  - 需要保留标题层级做分段        -> 自定义 BeautifulSoup 切分")
print("  - 表格型页面、需要按行列检索     -> pandas.read_html 或 CSVLoader")
