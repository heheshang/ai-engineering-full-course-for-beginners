"""生成用于演示的 SamplePDFFile.pdf（多页、可提取文本）。

运行方式：
    python src/04-document-loaders-demo/data/make_sample_pdf.py

为什么要自己生成 PDF？
    1) 仓库里放一个二进制 PDF 不好 review / diff；
    2) 用脚本生成可以精确控制页数、字体和文字内容，
       便于验证 PDF 加载器是否真的按页切分、metadata 里 page 是否正确；
    3) 后续要加页只要改脚本重新生成即可。

关于中文字体（踩坑记录）：
    PDF 的标准 14 号字体（Helvetica / Times 等）**只支持 Latin-1**，
    用它们写中文，PDF 里会渲染成方框，pypdf 提取出来是 ``■■■■``。
    reportlab 内置的 ``UnicodeCIDFont('STSong-Light')`` 也一样不行——
    CID 字体不嵌入字形，提取时同样拿不到真实字符。

    可行方案是**嵌入 TrueType 字体**：
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        pdfmetrics.registerFont(TTFont('CJK', '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
    注意：字体名必须同时写进 ParagraphStyle.fontName，
    否则样式仍会用默认字体，中文照样变方框。

    这里从 macOS 系统字体目录找一份可用的 CJK 字体并注册。
    跨平台时把候选列表换成对应系统的字体路径即可。

设计要点（针对 PDF 加载器的特性）：
    - 3 页，才能体现"一页 = 一个 Document"以及 load_and_split 的切分效果；
    - 每页内容带明确的页码标记（如 "第 1 页 / 共 3 页"），
      方便肉眼核对提取结果是否正确；
    - 三页结构故意不同：纯段落 / 表格 / 列表+小标题，
      用来验证不同版式的文本都能被提取出来。
"""

import sys
from pathlib import Path

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import sys
from pathlib import Path

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具。
# 注意：本文件位于 src/04-document-loaders-demo/data/ 下，比其它演示脚本深一层，
# 所以这里要用 parents[2] 才能定位到 src 目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.logging_utils import get_logger

# 初始化 logger：日志同时输出到控制台和 logs/course.log
logger = get_logger(__name__)


OUTPUT_PATH = Path(__file__).resolve().parent / "SamplePDFFile.pdf"

# 候选 CJK 字体路径：按优先级依次尝试，找到第一个能用的即可。
# （Windows 上一般用 C:\\Windows\\Fonts\\msyh.ttc，Linux 上用 Noto Sans CJK）
CJK_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",
]
CJK_FONT_NAME = "CJKEmbedded"


def register_cjk_font() -> str:
    """注册并返回可用的 CJK 字体名；找不到时回退到内置 Helvetica。

    中文字体是必须的：PDF 若不嵌入 CJK 字体，
    PyPDFLoader 提取出来的中文会全是 ■■■■。

    Returns:
        成功注册的字体名；全部候选都不可用时返回 "Helvetica"（不支持中文）。
    """
    for candidate in CJK_FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                pdfmetrics.registerFont(TTFont(CJK_FONT_NAME, candidate))
                logger.info("已注册 CJK 字体：%s（来自 %s）", CJK_FONT_NAME, candidate)
                return CJK_FONT_NAME
            except Exception as exc:  # noqa: BLE001 - 字体注册失败时继续尝试下一个
                logger.warning("字体注册失败：%s（%s），尝试下一个候选", candidate, exc)
                continue
    logger.error("未找到可用的 CJK 字体，中文将无法被正确提取")
    print("[warn] 未找到可用的 CJK 字体，中文将无法被正确提取", file=sys.stderr)
    return "Helvetica"


# 三页正文内容，结构故意做得不一样：
# 第 1 页：纯段落（最适合验证基础文本提取）
# 第 2 页：表格（验证表格文本是否也能被提取）
# 第 3 页：列表 + 小标题（验证多块级元素的提取顺序）
PAGES = [
    {
        "title": "RAG 系统设计实践指南",
        "subtitle": "第一章 · 检索增强生成的基本原理",
        "paragraphs": [
            "检索增强生成（Retrieval-Augmented Generation，简称 RAG）是一种把"
            "外部知识库与大语言模型结合起来的架构范式。它的核心思想是：在回答用户"
            "问题之前，先从知识库中检索相关文档片段，把这些片段作为上下文一起交给模型，"
            "从而让模型基于可溯源的事实作答，而不是仅凭训练时记住的参数化知识。",
            "一个典型的 RAG 流水线包含五个阶段：加载（Load）、切块（Chunk）、"
            "向量化（Embed）、检索（Retrieve）与生成（Generate）。其中切块策略"
            "对最终效果影响极大：块太大则噪声多、召回精度低；块太小则上下文断裂、"
            "语义不完整。生产环境中常见的做法是根据文档结构做递归切分，"
            "并保留一定重叠以避免边界信息丢失。",
            "评估 RAG 效果同样需要拆解。检索阶段关注召回率与准确率，"
            "生成阶段关注忠实度（faithfulness）与答案相关性。多数线上问题"
            "并非出在生成环节，而是检索阶段没有把正确的片段召回。"
            "因此在优化模型之前，先建立一套带标注的评测集，往往能取得更高的投入产出比。",
        ],
    },
    {
        "title": "文档切块方法对比",
        "subtitle": "第二章 · 常用分块器一览",
        "paragraphs": [
            "下表列出了几种常见分块器的核心参数与适用场景。"
            "实际选型时，应当先用真实业务文档做小规模实验，"
            "而不是照搬默认值。",
        ],
        "table": [
            ["分块器", "切分依据", "适用场景"],
            ["CharacterTextSplitter", "固定字符数", "结构单一的纯文本"],
            ["RecursiveCharacterTextSplitter", "递归尝试多个分隔符", "通用默认选择"],
            ["TokenTextSplitter", "Token 数量", "需要精确控制模型上下文"],
            ["MarkdownHeaderTextSplitter", "Markdown 标题层级", "结构化技术文档"],
        ],
    },
    {
        "title": "向量数据库选型要点",
        "subtitle": "第三章 · 存储层的工程考量",
        "paragraphs": [
            "向量数据库的选型通常围绕四个维度展开：索引结构、过滤能力、"
            "写入性能与运维成本。",
        ],
        "bullets": [
            "索引结构：HNSW 在召回率和查询延迟之间取得较好平衡，是目前的主流选择；"
            "IVF 适合超大规模数据，但需要重建索引。",
            "过滤能力：支持在相似度检索的同时按 metadata 过滤（如按时间、租户隔离），"
            "是很多企业场景的硬性需求。",
            "写入性能：需要频繁更新文档时，应关注索引的增量写入能力，"
            "避免每次全量重建。",
            "运维成本：单机嵌入式数据库（如 Chroma、FAISS）部署简单；"
            "分布式方案（Milvus、Qdrant）提供高可用与水平扩展，但运维复杂度更高。",
            "混合检索：在向量检索之外叠加关键词检索（BM25），"
            "往往能显著改善对专有名词、型号、编号类问题的召回效果。",
        ],
    },
]


def build_pdf(output_path: Path) -> None:
    """生成多页 PDF 到 output_path。"""
    font_name = register_cjk_font()

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=20,
        leading=26,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=11,
        leading=16,
        alignment=TA_CENTER,
        textColor="#555555",
        spaceAfter=18,
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=10.5,
        leading=17,
        spaceAfter=10,
    )
    marker_style = ParagraphStyle(
        "DocMarker",
        parent=body_style,
        fontSize=8,
        textColor="#888888",
    )
    bullet_style = ParagraphStyle(
        "DocBullet",
        parent=body_style,
        leftIndent=14,
        bulletIndent=4,
        spaceAfter=6,
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        title="RAG 系统设计实践指南",
        author="AI Engineering Course",
        subject="用于演示 PDF 文档加载器的示例文档",
    )

    story = []
    total_pages = len(PAGES)

    for index, page in enumerate(PAGES):
        # 每页开头写入明确的页码标记，便于核对提取结果
        story.append(
            Paragraph(f"第 {index + 1} 页 / 共 {total_pages} 页", marker_style)
        )
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph(page["title"], title_style))
        story.append(Paragraph(page["subtitle"], subtitle_style))

        for para in page.get("paragraphs", []):
            story.append(Paragraph(para, body_style))

        if "table" in page:
            table_data = page["table"]
            table = Table(table_data, colWidths=[45 * mm, 45 * mm, 75 * mm])
            table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.5, "#999999"),
                        ("BACKGROUND", (0, 0), (-1, 0), "#eeeeee"),
                        ("FONTNAME", (0, 0), (-1, 0), font_name),
                        ("FONTNAME", (0, 1), (-1, -1), font_name),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.append(table)

        if "bullets" in page:
            for bullet in page["bullets"]:
                story.append(Paragraph(bullet, bullet_style, bulletText="•"))

        if index < total_pages - 1:
            story.append(PageBreak())

    logger.info("开始构建 PDF，共 %d 页 -> %s", total_pages, output_path)
    doc.build(story)
    logger.info("PDF 构建完成：%s（%d 字节）", output_path, output_path.stat().st_size)


if __name__ == "__main__":
    build_pdf(OUTPUT_PATH)
    print(f"已生成：{OUTPUT_PATH}")
