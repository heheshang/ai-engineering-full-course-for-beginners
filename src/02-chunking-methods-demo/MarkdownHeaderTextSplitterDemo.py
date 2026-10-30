"""演示 3：MarkdownHeaderTextSplitter —— 按 Markdown 标题层级切分。

核心思想：
    不按长度切，而是按"标题"切：遇到配置过的 Markdown 标题（# / ## / ###）时，
    把标题单独拎出来存进 metadata，正文（直到下一个同级或更高级标题）作为
    page_content。也就是"一节 = 一个 chunk"。

特点与适用场景：
    - 优点：结构感知，块与块之间的层级关系被显式保留在 metadata 里
      （Header 1 / Header 2 / Header 3），检索时可以按层级过滤，
      也方便把标题回填到 prompt 里提供上下文。
    - 缺点：只适用于有明确标题层级的文档；对纯文本无效，
      且块大小由文档自身的章节长度决定，无法精确控制 token 数。
    - 适用：技术文档、Wiki、README、课程讲义等结构化文档。

与其他切分器的定位差异：
    - 长度类切分器（Character / Recursive / Token）：控制"块有多大"。
    - 标题类切分器（MarkdownHeader）：控制"块怎么划分才符合文档结构"。
    实际项目中两者常组合使用：先按标题切，再对超长的章节用
    RecursiveCharacterTextSplitter 二次细分。
"""

import sys
from pathlib import Path

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_text_splitters import MarkdownHeaderTextSplitter

from utils.logging_utils import get_logger

# 初始化 logger：日志同时输出到控制台和 logs/course.log
logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# 输入文档：结构清晰的 Markdown 文档，含 1 级到 3 级标题
# --------------------------------------------------------------------------- #
sample_text = """
# Introduction to Machine Learning

Machine Learning (ML) is a subset of artificial intelligence that enables systems to learn and improve from experience. It focuses on developing computer programs that can access data and use it to learn for themselves.

## Types of Machine Learning

### Supervised Learning
Supervised learning is where the model is trained on a labeled dataset. The model learns to predict the output from the input data. Examples include:
- Classification
- Regression
- Neural Networks

### Unsupervised Learning
Unsupervised learning is where the model works on its own to discover patterns and information. Common examples include:
- Clustering
- Dimensionality Reduction
- Association

## Applications

Machine Learning has numerous real-world applications:
1. Image Recognition
2. Natural Language Processing
3. Recommendation Systems
4. Fraud Detection
"""

# --------------------------------------------------------------------------- #
# 切分
# --------------------------------------------------------------------------- #
print("\n3. Markdown Header Text Splitter Example: Ideal for markdown documentation")

# headers_to_split_on：[(标题标记, metadata 中的字段名), ...]
# 键是 Markdown 里实际出现的标记，值是希望存进 metadata 的键名。
# 列表的顺序不重要，切分器会按标记长度从长到短匹配，避免 "##" 被当成 "#" 处理。
headers_to_split_on = [
    ("#", "Header 1"),  # 一级标题 -> metadata["Header 1"]
    ("##", "Header 2"),  # 二级标题 -> metadata["Header 2"]
    ("###", "Header 3"),  # 三级标题 -> metadata["Header 3"]
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# split_text 返回 List[Document]（注意：这里直接就是 Document，不是 str）。
# 每个 Document 的结构：
#   page_content -> 该节的正文（不含标题文字）
#   metadata     -> {"Header 1": "...", "Header 2": "...", ...}，逐级继承上级标题
docs = markdown_splitter.split_text(sample_text)

# 日志：记录切分数量与每个块的标题层级路径。
# 注意：切分数量由文档的标题数量决定，与文本长度无关——
# 这是标题类切分器与长度类切分器最本质的差别。
logger.info(
    "按标题切分完成：原文 %d 字符 -> %d 个 Document（数量由标题数决定）",
    len(sample_text),
    len(docs),
)
for i, doc in enumerate(docs, start=1):
    # 只记录有值的标题键，构造出「层级路径」方便观察继承关系
    header_path = " > ".join(f"{k}={v}" for k, v in doc.metadata.items() if k.startswith("Header"))
    logger.info(
        "  Document[%d] 长度=%d 标题路径=%s", i, len(doc.page_content), header_path or "(无标题)"
    )

# 先整体打印，方便观察 metadata 的层级继承关系。
print(docs)

for i, doc in enumerate(docs, start=1):
    print(f"\n Document {i}:")
    # header_type 在本例中并不存在（我们用的是 "Header 1/2/3"），
    # 这里只是演示用 .get 带默认值的方式读取，避免 KeyError。
    print(f"Header: {doc.metadata.get('header_type', 'No header')}")
    print(f"Content:\n{doc.page_content}")
    print("-" * 50)

# --------------------------------------------------------------------------- #
# 观察小结
# --------------------------------------------------------------------------- #
# 1. 标题本身不会出现在 page_content 里，而是被"提"到 metadata 中。
#    好处是可以按章节做过滤检索；代价是如果下游只用 page_content，
#    会丢失"这段话属于哪一节"的信息。生产中常把标题拼回正文再入库。
# 2. metadata 会沿层级向下继承（实测共 4 个 Document）：
#      {'Header 1': 'Introduction to Machine Learning'}
#      {'Header 1': 'Introduction...', 'Header 2': 'Types of Machine Learning',
#       'Header 3': 'Supervised Learning'}
#      {'Header 1': 'Introduction...', 'Header 2': 'Types of Machine Learning',
#       'Header 3': 'Unsupervised Learning'}
#      {'Header 1': 'Introduction...', 'Header 2': 'Applications'}
#    即"上下文路径（context path）"，与 LlamaIndex 的 HeaderNode / MarkdownNode 机制一致。
# 3. 切分结果数量由文档标题数量决定（本例 4 个），与文本长度无关。
