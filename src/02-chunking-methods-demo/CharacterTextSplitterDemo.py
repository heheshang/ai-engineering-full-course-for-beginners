"""演示 1：CharacterTextSplitter —— 最基础的按字符分隔符切分。

核心思想：
    指定一个固定的分隔符（separator），先按它把文本"硬"切开，
    再把切出来的碎片按 chunk_size 拼成不超过上限的块，最后做 overlap 重叠。

特点与适用场景：
    - 优点：实现最简单，速度快，行为完全可预测（分隔符位置即切点）。
    - 缺点：只能在一个粒度上切。对本例的 Markdown 文档来说，separator="\\n\\n"
      会把"# 标题"和它下面的正文分到不同块，导致语义割裂
      （标题和内容本应放在一起才有意义）。
    - 适用：结构扁平、以自然段为单位且不关心标题层级的纯文本。

对比其他三种切分器（见同目录其他文件）：
    RecursiveCharacterTextSplitter -> 多级分隔符递归切，保留更多结构
    MarkdownHeaderTextSplitter     -> 依据 Markdown 标题层级切，标题进 metadata
    TokenTextSplitter              -> 依据 token 数量切，贴合大模型上下文窗口
"""

import sys
from pathlib import Path

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_text_splitters import CharacterTextSplitter

from utils.logging_utils import get_logger

# 初始化 logger：日志同时输出到控制台和 logs/course.log
logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# 输入文档：一份 Markdown 格式的机器学习入门文档
# --------------------------------------------------------------------------- #
# 文档刻意混合了多种结构元素，用来观察不同切分器的差异：
#   - "#" 一级标题、列表项、编号列表、空行分段
#   - 段落长度不均（有的 1 行，有的 3 行）
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
print("\n1. Character Text Splitter Example: Best for simple text splitting")

splitter = CharacterTextSplitter(
    # separator：唯一的硬分隔符。文本会先在所有 "\n\n"（空行）处断开。
    separator="\n\n",
    # chunk_size：每个块的目标长度上限（单位由 length_function 决定，这里是字符数）。
    chunk_size=100,
    # chunk_overlap：相邻块之间保留的重叠长度，避免答案正好落在切点上被劈成两半。
    # 注意：overlap 会让总长度膨胀，这里 50/100 = 50% 的重叠比例偏高，实际项目常用 10%~20%。
    chunk_overlap=50,
)

# split_text 接收字符串，返回字符串列表（List[str]）。
chunks = splitter.split_text(sample_text)

# 日志：记录切分结果的规模与长度分布，便于对比不同切分器的差异
lengths = [len(c) for c in chunks]
logger.info(
    "切分完成：原文 %d 字符 -> %d 个片段，最短 %d 字符，最长 %d 字符，平均 %.1f 字符",
    len(sample_text),
    len(chunks),
    min(lengths) if lengths else 0,
    max(lengths) if lengths else 0,
    (sum(lengths) / len(lengths)) if lengths else 0.0,
)
# 超过 chunk_size 的片段说明无法在分隔符处断开，属于「超长块」，需要重点关注
oversized = [i for i, n in enumerate(lengths, start=1) if n > 100]
if oversized:
    logger.warning("以下片段长度超过 chunk_size=100（超长块）：%s", oversized)

# 打印原始列表与元素类型，方便直观确认返回结构。
print(chunks)
print(type(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i}:")
    print(chunk)
    print("-" * 30)

# --------------------------------------------------------------------------- #
# 观察小结
# --------------------------------------------------------------------------- #
# 1. 返回类型是 List[str]，若需要 LangChain 统一的 Document 结构，
#    可改用 splitter.create_documents([sample_text])，会额外携带 metadata。
# 2. 切分点严格落在 "\n\n" 上，因此 "# Introduction to Machine Learning"
#    这个标题会与它下面的正文分属不同块 —— 这正是基础字符切分的短板。
# 3. 超长段落无法被进一步细分（CharacterTextSplitter 不会二次递归切分），
#    会原样保留为超长块，并在运行时打印告警：
#        "Created a chunk of size 219, which is longer than the specified 100"
#    本例中就有 3 个这样的告警块（约 200~220 字符）。
#    改用 RecursiveCharacterTextSplitter 可彻底解决（见演示 2）。
