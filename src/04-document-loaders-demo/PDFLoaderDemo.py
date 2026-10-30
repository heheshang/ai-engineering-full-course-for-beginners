"""文档加载器演示：PDF（Document Loaders - PDF）。

本例演示如何把 PDF 文档转换成 LangChain 的 ``Document`` 对象，
并解释 PDF 加载与其它格式加载的本质差异。

核心概念：
    PDF 是 RAG 里最常见也最麻烦的格式，麻烦在于三点：

    1) **PDF 记录的是"绘制指令"，不是"文本流"**。
       同一个字可能由多个绘制片段拼出来，段落顺序依赖阅读顺序推断。
       所以提取结果里经常出现断行、乱序、页眉页脚混入正文等问题。

    2) **没有天然的语义边界**。
       PDF 里没有"段落"这个概念，只有坐标和字号。
       想要按章节切分，必须结合字体大小、坐标等信息做启发式判断。

    3) **扫描件需要 OCR**。
       纯图片 PDF 提取出来是空字符串，必须先做 OCR（文字识别），
       例如用 ``UnstructuredPDFLoader`` 配合 OCR 策略。

    提取方式对照：
      - ``pypdf``（PyPDFLoader 默认后端）：纯 Python、无额外依赖、快，
        但对复杂版式和表格支持一般；
      - ``pdfplumber``（更精细，能拿到表格和坐标）；
      - ``unstructured``：版面分析能力强，能识别标题/表格/图片区域，
        但依赖很重（需要系统二进制支持）；
      - ``OCR``：处理扫描件的唯一出路。

    本例默认使用 ``PyPDFLoader``（依赖已装），它基于 ``pypdf``。

关于 ``load()`` 与 ``load_and_split()`` 的区别（很关键）：
    - ``load()``：**一页 = 一个 Document**，不做任何切分。
      metadata 里带 ``page``（从 0 开始的页码）、``total_pages``、
      ``page_label``（PDF 里显示的页码，如 "1"）、``title`` 等信息。
    - ``load_and_split()``：先按页加载，再**用默认的 RecursiveCharacterTextSplitter
      切分**（chunk_size=4000, chunk_overlap=200），每个 chunk 一个 Document，
      并在 metadata 里加上 ``page`` 指向它来自哪一页。

    所以 ``load_and_split()`` 的结果数 >= 页数，**不是**"一页一个"。
    很多人误以为 ``load_and_split`` 只是按页切，这其实是它默认会做字符级切分。

关于导入路径（LangChain 1.x）：
    具体格式的加载器不在 ``langchain_core`` 里，
    而在 ``langchain_community.document_loaders``（该包已进入 sunset 阶段，
    未来会拆分为独立集成包，当前仍可用）。

关于路径：
    ``PyPDFLoader("data/SamplePDFFile.pdf")`` 这种相对路径**依赖当前工作目录**，
    从仓库根目录运行就会报 FileNotFoundError。
    本例用 ``Path(__file__).resolve().parent`` 推导绝对路径。
"""

import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

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
# 用 Path 推导绝对路径，避免"相对路径依赖当前工作目录"的坑。
BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "SamplePDFFile.pdf"

# --------------------------------------------------------------------------- #
# 1. 方式一：load() —— 一页一个 Document，不做切分
# --------------------------------------------------------------------------- #
print("=" * 70)
print("1) load() —— 一页 = 一个 Document（不做任何切分）")
print("=" * 70)

loader = PyPDFLoader(str(PDF_PATH))
pages = loader.load()
logger.info("PDF 加载完成：文件=%s，共 %d 页 -> %d 个 Document", PDF_PATH.name, len(pages), len(pages))

print(f"共加载 {len(pages)} 个 Document（对应 PDF 的 {len(pages)} 页）")
print("类型：", type(pages))

print("\n--- pages[0] 的 metadata ---")
# PDF 的 metadata 非常丰富：既包含文件级信息（title/author/producer），
# 也包含页级信息（page/total_pages/page_label）。
# 做引用溯源时，page_label 是给人看的（"3"），page 是给程序用的（0 起始）。
print(json.dumps(pages[0].metadata, indent=2, ensure_ascii=False))

print("\n--- pages[0].page_content ---")
print(pages[0].page_content)

print("\n--- pages[1].page_content（第 2 页，含表格）---")
print(pages[1].page_content)

print("\n--- pages[2].page_content（第 3 页，含列表）---")
print(pages[2].page_content)

# --------------------------------------------------------------------------- #
# 2. 方式二：load_and_split() —— 按字符切分成 chunk
# --------------------------------------------------------------------------- #
# load_and_split() 用的是默认的 RecursiveCharacterTextSplitter：
#   chunk_size=4000, chunk_overlap=200
# 我们的示例文档每页才几百字，所以不会被切开 —— 但真实 PDF 通常会。
print("\n" + "=" * 70)
print("2) load_and_split() —— 再叠加字符级切分（默认 chunk_size=4000）")
print("=" * 70)

chunks = loader.load_and_split()
logger.info("默认切分完成：%d 页 -> %d 个 chunk（默认 chunk_size=4000）", len(pages), len(chunks))

print(f"切分后共 {len(chunks)} 个 chunk（>= 页数 {len(pages)}）")
for i, chunk in enumerate(chunks):
    md = chunk.metadata
    print(
        f"[{i}] page={md.get('page')} "
        f"page_label={md.get('page_label')} "
        f"total_pages={md.get('total_pages')} "
        f"字符数={len(chunk.page_content)}"
    )

# --------------------------------------------------------------------------- #
# 3. 方式三：自定义切分参数 —— 控制 chunk 大小与重叠
# --------------------------------------------------------------------------- #
# 真实场景下要自己指定 chunk_size / chunk_overlap：
#   chunk_size 太大 -> 一个 chunk 里混进多个主题，召回精度下降；
#   chunk_size 太小 -> 上下文断裂，模型拿到半句话；
#   chunk_overlap  -> 防止句子正好被切断在边界上。
print("\n" + "=" * 70)
print("3) 方式三：自定义切分参数（chunk_size=150, chunk_overlap=30）")
print("=" * 70)

from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
# 注意参数名是 text_splitter（不是 splitter）——
# load_and_split(text_splitter=None) 的默认值是 RecursiveCharacterTextSplitter()，
# 即默认 chunk_size=4000, chunk_overlap=200。
small_chunks = loader.load_and_split(text_splitter=splitter)
logger.info("自定义切分完成：%d 页 -> %d 个 chunk（chunk_size=150, chunk_overlap=30）",
            len(pages), len(small_chunks))

print(f"切分后共 {len(small_chunks)} 个 chunk")
for i, chunk in enumerate(small_chunks):
    md = chunk.metadata
    preview = chunk.page_content.replace("\n", " ")[:70]
    print(
        f"[{i}] page={md.get('page')} 字符数={len(chunk.page_content):>4} | {preview}..."
    )

# --------------------------------------------------------------------------- #
# 4. 方式四：自定义 loader —— 只保留需要的 metadata
# --------------------------------------------------------------------------- #
# PDF 自带的 metadata 非常杂（producer、creator、trapped 等），
# 如果要入库到向量库，建议精简一下，只留真正有用的字段，
# 否则每个 chunk 都带着一堆无用字符串，白占存储。
print("\n" + "=" * 70)
print("4) 方式四：精简 metadata（去掉 producer/trapped 等无用字段）")
print("=" * 70)

KEEP_KEYS = {"source", "page", "page_label", "total_pages", "title"}
slim_pages = [
    {
        "page_content": doc.page_content,
        "metadata": {k: v for k, v in doc.metadata.items() if k in KEEP_KEYS},
    }
    for doc in pages
]

print("精简前第 0 页的 metadata 字段数：", len(pages[0].metadata))
logger.info("metadata 精简：字段数 %d -> %d", len(pages[0].metadata), len(slim_pages[0]["metadata"]))
print("精简后第 0 页的 metadata：")
print(json.dumps(slim_pages[0]["metadata"], indent=2, ensure_ascii=False))

# --------------------------------------------------------------------------- #
# 5. 常见坑位
# --------------------------------------------------------------------------- #
print("\n" + "=" * 70)
print("5) 常见坑位提醒")
print("=" * 70)
print(
    """
1) 中文字体：PDF 必须嵌入 CJK 字体，否则提取出来全是 ■■■■。
   本例的 SamplePDFFile.pdf 由 make_sample_pdf.py 生成，
   已嵌入 Arial Unicode 字体，中文可正常提取。
2) 扫描件：纯图片 PDF 提取结果是空字符串，必须做 OCR。
3) 相对路径：PyPDFLoader("data/x.pdf") 依赖当前工作目录，
   请用 Path(__file__).resolve().parent 推导绝对路径。
4) load_and_split 不是"按页切"：它默认还会做字符级切分，
   结果数量通常远大于页数。
5) 页眉页脚噪音：每页重复的页码/标题会污染向量，
   切分前建议用正则清洗。
"""
)

# --------------------------------------------------------------------------- #
# 附：接进 RAG 流水线
# --------------------------------------------------------------------------- #
#   from langchain_chroma import Chroma
#   from langchain_huggingface import HuggingFaceEmbeddings
#
#   splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#   chunks = PyPDFLoader("paper.pdf").load_and_split(splitter=splitter)
#   vectorstore = Chroma.from_documents(chunks, HuggingFaceEmbeddings(...))
#
# 由于 metadata 里带着 page/page_label，检索结果可以直接生成
# "见第 3 页" 这样的引用，比只给文件名有用得多。
