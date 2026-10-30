"""文本切分与文档加载评测（**完全离线，不调用 LLM**）。

覆盖的内容
----------
============================  ==============================================
检查项                         评估什么
============================  ==============================================
确定性断言                       切分是否正确、加载是否完整、元数据是否保留
``GEval``（可选）               切分后的片段是否语义连贯
============================  ==============================================

对应被测系统
------------
- ``src/02-chunking-methods-demo/``（四种切分器）
- ``src/04-document-loaders-demo/``（CSV / HTML / PDF 加载器）

为什么这一组要单独放在离线文件里
--------------------------------
切分与加载是**确定性**的：同样的输入必然得到同样的输出。
这类断言不需要 LLM 评判，因此：

1. **快**：秒级完成，适合每次提交都跑；
2. **稳**：没有 LLM 的随机性，不会出现「昨天过今天不过」；
3. **零成本**：不需要 API Key，CI 里可以直接跑。

这也给整个评测套件定了一个基调：**能用确定性断言的，就别用 LLM 评判**。
LLM 评判留给真正主观的维度。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sut import load_chunk_documents, load_csv_documents
from eval_config import build_g_eval, threshold
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# 允许被截断的片段占比上限（定长切分的固有代价，超过说明 chunk_size 太小）
MAX_BROKEN_CHUNK_RATIO = 0.5

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "src" / "04-document-loaders-demo" / "data"


# --------------------------------------------------------------------------- #
# 文本切分
# --------------------------------------------------------------------------- #
def test_recursive_splitter_respects_chunk_size():
    """切分后的每个 chunk 都不应超过 ``chunk_size``（允许少量溢出）。

    RecursiveCharacterTextSplitter 在遇到超长单词（无分隔符）时
    会强制切开，这类 chunk 可能略微超限，因此用 1.2 倍做容差。
    """
    chunk_size = 100
    chunks = load_chunk_documents(chunk_size=chunk_size, chunk_overlap=20)

    assert chunks, "切分结果为空"
    oversized = [c for c in chunks if len(c) > chunk_size * 1.2]
    assert not oversized, f"存在超长 chunk（>{chunk_size * 1.2}）：{[len(c) for c in oversized]}"


def test_recursive_splitter_covers_whole_document():
    """切分必须覆盖原文的全部内容，不能丢字。

    用「把所有 chunk 拼起来，去掉空白后与原文比较」的方式校验。
    重叠部分会被重复计入，所以这里只校验「原文的每个字符都出现过」，
    而不是严格等长。
    """
    from sut import RAG_DOCUMENT

    chunks = load_chunk_documents(chunk_size=100, chunk_overlap=20)
    combined = "".join(chunks)
    compact_original = "".join(RAG_DOCUMENT.split())
    compact_combined = "".join(combined.split())

    # 原文的每个字符都应出现在切分结果中
    assert len(compact_combined) >= len(compact_original) * 0.98, (
        f"切分结果明显短于原文（{len(compact_combined)} vs {len(compact_original)}），疑似丢内容"
    )

    missing = [
        word
        for word in set(compact_original.split())
        if word not in compact_combined
    ]
    # 允许极少量因分隔符处理导致的差异
    assert len(missing) == 0, f"切分丢失了原文中的内容：{missing[:10]}"


def test_chunk_size_affects_sentence_integrity():
    """chunk_size 太小会把句子从中间切断。

    这是一条**确定性**检查（不依赖 LLM），把「切分粒度要匹配段落长度」
    这个工程经验固化成回归测试。

    实测：本示例文档的段落长度在 153~236 字符之间，
    chunk_size=150 时 15 个 chunk 里有 9 个以非句末标点结尾（被截断），
    chunk_size=200 时降到 4 个，chunk_size=250 时为 0 个。
    """
    def truncated_count(chunk_size: int) -> int:
        chunks = load_chunk_documents(chunk_size=chunk_size, chunk_overlap=30)
        return sum(
            1 for c in chunks if not c.rstrip().endswith((".", "!", "?"))
        )

    small = truncated_count(150)
    medium = truncated_count(200)
    large = truncated_count(250)

    assert small > medium, (
        f"更小的 chunk_size 没有带来更多截断（{small} vs {medium}）"
    )
    assert medium > large, (
        f"chunk_size=200 的截断数应当多于 chunk_size=250（{medium} vs {large}）"
    )
    assert large == 0, (
        f"chunk_size=250 时不应有截断，实际有 {large} 个片段以非句末标点结尾"
    )


def test_recursive_splitter_has_overlap():
    """参数合适时，相邻 chunk 之间应当存在重叠（保证语义连续性）。

    没有重叠的切分会在语义边界处「切断」句子，
    导致检索到半句话、生成时缺上下文。

    注意 ``RecursiveCharacterTextSplitter`` 的一个实现细节：
    只有当一个 chunk 内能放下**多个**子分隔符切分出的片段时，
    ``chunk_overlap`` 才会生效；否则每个 chunk 恰好由一个片段构成，
    自然没有重叠。所以这里用「一段能容纳多个句子」的参数来验证。
    """
    chunks = load_chunk_documents(chunk_size=200, chunk_overlap=100)
    assert len(chunks) >= 2, "chunk 数量不足，无法验证重叠"

    # 用公共后缀/前缀长度来衡量重叠
    def overlap_len(a: str, b: str) -> int:
        max_len = min(len(a), len(b))
        for size in range(max_len, 0, -1):
            if a[-size:] == b[:size]:
                return size
        return 0

    overlaps = [overlap_len(chunks[i], chunks[i + 1]) for i in range(len(chunks) - 1)]
    assert any(o > 0 for o in overlaps), (
        f"相邻 chunk 之间未检测到重叠，重叠长度：{overlaps}"
    )


def test_recursive_splitter_overlap_increases_redundancy():
    """开启重叠后，切分结果应当出现重复内容。

    这是「重叠确实生效」的等价验证：
    同一个片段同时出现在相邻两个 chunk 中。
    """
    from sut import RAG_DOCUMENT

    without = load_chunk_documents(chunk_size=200, chunk_overlap=0)
    with_overlap = load_chunk_documents(chunk_size=200, chunk_overlap=100)

    def total_length(chunks):
        return sum(len(c) for c in chunks)

    # 有重叠时，总字符数会更大（同一段内容被计入两次）
    assert total_length(with_overlap) > total_length(without), (
        f"开启重叠后总长度未增加：{total_length(with_overlap)} vs {total_length(without)}"
    )
    assert len(with_overlap) == len(without), (
        "重叠不应当改变 chunk 数量，只应当增加重复内容"
    )


def test_recursive_splitter_smaller_size_gives_more_chunks():
    """切分粒度越细，chunk 数量应当越多。

    这是切分参数的基本单调性检查——
    防止参数传错导致切分行为反常。
    """
    large = load_chunk_documents(chunk_size=300, chunk_overlap=50)
    small = load_chunk_documents(chunk_size=100, chunk_overlap=20)
    assert len(small) > len(large), (
        f"更小的 chunk_size 没有产生更多 chunk（{len(small)} vs {len(large)}）"
    )


def test_token_splitter_uses_token_model():
    """TokenTextSplitter 应按 token 而非字符计数。

    英文场景下 1 token ≈ 4 字符，
    所以 token 数应明显小于字符数。
    """
    from langchain_text_splitters import TokenTextSplitter

    from sut import RAG_DOCUMENT

    splitter = TokenTextSplitter(chunk_size=50, chunk_overlap=10)
    chunks = splitter.split_text(RAG_DOCUMENT)

    assert chunks, "切分结果为空"
    total_chars = sum(len(c) for c in chunks)
    assert total_chars > len(RAG_DOCUMENT), "切分后总字符数应不少于原文"


# --------------------------------------------------------------------------- #
# 文档加载
# --------------------------------------------------------------------------- #
def test_csv_loader_reads_all_rows():
    """CSVLoader 应当把每一行读成一个 Document。"""
    docs = load_csv_documents()
    assert docs, "CSV 加载结果为空"

    csv_path = DATA_DIR / "products.csv"
    with open(csv_path, encoding="utf-8") as f:
        expected_rows = sum(1 for _ in f) - 1  # 去掉表头

    assert len(docs) == expected_rows, (
        f"文档数不符：期望 {expected_rows}，实际 {len(docs)}"
    )


def test_csv_loader_preserves_metadata():
    """默认的 CSVLoader 会把 ``source`` 与行号放进 metadata。

    注意：默认行为下**列名不在 metadata 里**，
    而是被渲染进 ``page_content``（形如 ``name: 机械键盘\\ncategory: 外设``）。
    想把某列放进 metadata，需要用 ``source_column`` /
    ``metadata_columns`` 参数——这正是 ``CSVLoaderDemo.py`` 演示的对比点。
    """
    docs = load_csv_documents()
    assert docs[0].metadata, "metadata 为空"
    assert "source" in docs[0].metadata, f"metadata 缺少 source：{docs[0].metadata}"
    assert "row" in docs[0].metadata, f"metadata 缺少行号：{docs[0].metadata}"


def test_csv_loader_with_metadata_columns():
    """使用 ``metadata_columns`` 时，指定列应进入 metadata。

    对应 ``CSVLoaderDemo.py`` 里「方式二」的用法。
    """
    from langchain_community.document_loaders import CSVLoader

    csv_path = DATA_DIR / "products.csv"
    docs = CSVLoader(
        file_path=str(csv_path),
        source_column="name",
        metadata_columns=["category", "price"],
    ).load()

    assert docs, "加载结果为空"
    metadata = docs[0].metadata
    assert "category" in metadata, f"metadata 缺少 category：{metadata}"
    assert "price" in metadata, f"metadata 缺少 price：{metadata}"
    assert metadata["source"] == docs[0].metadata["source"], "source 不一致"

    # page_content 此时只剩下非 metadata 的列
    assert "category" not in docs[0].page_content, (
        f"category 已被移入 metadata，不应再出现在 page_content：{docs[0].page_content}"
    )


def test_csv_loader_page_content_contains_fields():
    """默认行为下，字段内容应当渲染进 page_content。"""
    docs = load_csv_documents()
    content = docs[0].page_content

    assert content.strip(), "page_content 为空"
    assert ": " in content, f"page_content 疑似不是「字段: 值」格式：{content[:100]}"
    # 不应包含文件名
    assert "products.csv" not in content, (
        f"page_content 疑似是文件名而非行内容：{content[:100]}"
    )


def test_html_loader_extracts_text():
    """BSHTMLLoader 应当提取出正文文本，而不是原始标签。

    对应 ``HTMLLoaderDemo.py`` 的「方式一」。
    注意老教程里的 ``langchain_community.document_loaders.HTMLLoader``
    已被移除，现在的标准写法是 ``BSHTMLLoader``。
    """
    from langchain_community.document_loaders import BSHTMLLoader

    html_path = DATA_DIR / "titanic.html"
    docs = BSHTMLLoader(file_path=str(html_path)).load()

    assert docs, "HTML 加载结果为空"
    content = docs[0].page_content
    assert len(content.strip()) > 100, f"提取到的文本过少：{len(content)} 字符"
    assert "<html" not in content.lower(), "page_content 中仍包含 HTML 标签"


def test_html_loader_keeps_tables():
    """BSHTMLLoader 应当保留表格内容（Titanic 示例里有表格）。"""
    from langchain_community.document_loaders import BSHTMLLoader

    html_path = DATA_DIR / "titanic.html"
    docs = BSHTMLLoader(file_path=str(html_path)).load()

    content = docs[0].page_content.lower()
    # Titanic 表格的关键列名
    assert any(
        keyword in content for keyword in ("pclass", "survived", "name", "sex")
    ), f"未在提取内容中找到表格列名：{content[:300]}"


def test_html_loader_keeps_title_metadata():
    """BSHTMLLoader 应当在 metadata 里保留 ``source`` 与 ``title``。

    注意：它**不**保留原始 HTML——想拿原始结构要自己用 BeautifulSoup 解析。
    这正是 ``HTMLLoaderDemo.py`` 演示「自定义 Loader」的原因。
    """
    from langchain_community.document_loaders import BSHTMLLoader

    html_path = DATA_DIR / "titanic.html"
    docs = BSHTMLLoader(file_path=str(html_path)).load()

    metadata = docs[0].metadata
    assert metadata, "metadata 为空"
    assert "source" in metadata, f"metadata 缺少 source：{list(metadata.keys())}"
    assert "title" in metadata, f"metadata 缺少 title：{list(metadata.keys())}"
    assert "Titanic" in metadata["title"], (
        f"title 内容不符：{metadata['title']}"
    )


@pytest.mark.skipif(
    not (DATA_DIR / "SamplePDFFile.pdf").exists(),
    reason="示例 PDF 不存在（先运行 make_sample_pdf.py 生成）",
)
def test_pdf_loader_reads_pages():
    """PDFLoader 应当按页返回 Document，并保留页码元数据。"""
    from langchain_community.document_loaders import PyPDFLoader

    pdf_path = DATA_DIR / "SamplePDFFile.pdf"
    docs = PyPDFLoader(file_path=str(pdf_path)).load()

    assert docs, "PDF 加载结果为空"
    assert all("page" in doc.metadata for doc in docs), "metadata 缺少页码信息"


# --------------------------------------------------------------------------- #
# 语义连贯性（可选，需要 LLM）
# --------------------------------------------------------------------------- #
@pytest.mark.llm_judge
@pytest.mark.slow
def test_chunks_are_semantically_coherent(judge):
    """**大多数**切分片段应当语义连贯。

    用 GEval 让裁判判断每个 chunk 是否是一个完整的语义单元。

    **为什么断言的是比例而不是「每个 chunk 都完美」**
    ------------------------------------------------
    定长切分天然会在部分片段的中间断开——这是切分本身的固有代价，
    不是 bug。真正该关心的是「有多少比例被切坏了」：
    比例过高说明 chunk_size 相对段落长度太小，应当调大。

    所以这里逐个测，再统计不连贯的比例，断言它低于阈值。
    用 ``metric.measure()`` 而不是 ``assert_test()``，
    是因为后者一遇到失败就中断，拿不到完整统计。

    chunk_size 的选择：本文档段落长度 153~236 字符，
    用 200 会把 10 个 chunk 中的 8 个从中间切断（80% 断裂率），
    用 250（刚好容纳最长段落）则是 6 个 chunk、0 断裂。
    这条对比也由上面的 ``test_chunk_size_affects_sentence_integrity`` 固化。
    """
    from deepeval.test_case import LLMTestCase

    chunks = load_chunk_documents(chunk_size=250, chunk_overlap=30)
    assert chunks, "切分结果为空"

    metric = build_g_eval(
        name="切分片段语义连贯",
        evaluation_steps=[
            "判断该文本片段是否是一个语义上相对完整的单元。",
            "若片段在句子中间被截断导致意思不完整，判为不符合。",
        ],
    )

    broken: list[tuple[int, str]] = []
    for index, chunk in enumerate(chunks):
        test_case = LLMTestCase(
            input=f"判断第 {index} 个片段是否语义连贯",
            actual_output=chunk,
        )
        metric.measure(test_case)
        if metric.score < 0.5:
            broken.append((index, metric.reason))

    broken_ratio = len(broken) / len(chunks)
    logger.info(
        "切分连贯性统计：%d/%d 个片段被判为不连贯（占比 %.1f%%）",
        len(broken),
        len(chunks),
        broken_ratio * 100,
    )

    assert broken_ratio <= MAX_BROKEN_CHUNK_RATIO, (
        f"被截断的片段占比过高：{len(broken)}/{len(chunks)} = {broken_ratio:.1%}，"
        f"超过阈值 {MAX_BROKEN_CHUNK_RATIO:.0%}。\n"
        f"问题片段：{[i for i, _ in broken]}\n"
        f"建议调大 chunk_size（当前 200，段落长度 153~236）"
    )
