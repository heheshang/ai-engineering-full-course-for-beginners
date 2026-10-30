"""演示 4：TokenTextSplitter —— 按 token 数量切分（最贴合大模型上下文窗口）。

核心思想：
    不按字符数，而是按"token 数"切分。token 才是大模型真正计费和计算上下文占用的单位：
    同样的文字，在不同模型的分词器（tokenizer）下对应的 token 数不同，
    用字符数估算很容易把上下文撑爆或浪费。

特点与适用场景：
    - 优点：切分结果与模型上下文窗口一一对应，可以精确控制"每块最多占多少 token"，
      避免请求被截断（truncation）或报错超限。
    - 缺点：依赖具体的分词器（默认 tiktoken 的 cl100k_base，针对 GPT 系列训练），
      用于其他模型（如 DeepSeek、Llama）时会有偏差；分词本身也比字符切分慢。
    - 适用：需要严格控制上下文长度的场景；长文档压缩/摘要；
      或者需要按 token 计费做成本预估的系统。

    补充：LangChain 还提供 TokenTextSplitter.from_tiktoken_encoder(...)
    来指定编码器，以及 HuggingFaceTokenizer 等变体，以匹配目标模型的分词方式。

与其他切分器的定位差异：
    - Character / Recursive：以"字符"为尺度，通用但不精确。
    - Token：以"模型 token"为尺度，精确对齐上下文窗口。
    - MarkdownHeader：以"文档结构"为尺度，不关心块有多大。
"""

import sys
from pathlib import Path

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_text_splitters import TokenTextSplitter

from utils.logging_utils import get_logger

# 初始化 logger：日志同时输出到控制台和 logs/course.log
logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# 输入文档：与前三份演示相同的 Markdown 文档，保证横向可比
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
print("\n4. Token Text Splitter Example: Perfect for LLM token limit management")

splitter = TokenTextSplitter(
    # chunk_size=100：每块最多 100 个 token。
    # 参考值：1 个 token 约等于 4 个英文字符 / 约 1.5 个中文字符，
    # 所以 100 token 大约相当于 400 个英文字符左右。
    chunk_size=100,
    # chunk_overlap=20：相邻块重叠 20 个 token。
    # 重叠的作用：同一个信息在相邻块都出现一部分，
    # 即使检索只命中了其中一个块，也不会丢失跨界句子的完整语义。
    chunk_overlap=20,
)

# split_text 返回 List[str]。要拿到真实 token 数用于核对，可以用：
#   encoding = tiktoken.get_encoding("cl100k_base")
#   print(len(encoding.encode(chunk)))
chunks = splitter.split_text(sample_text)

# 日志：记录切分规模。Token 切分器的 chunk_size 是「token 数」而非字符数，
# 所以这里额外用 tiktoken 编码器核对真实 token 数（默认的 cl100k_base）。
try:
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    token_counts = [len(encoding.encode(chunk)) for chunk in chunks]
    logger.info(
        "切分完成：原文 %d 字符 -> %d 个片段（token 数：最少 %d，最多 %d，平均 %.1f）",
        len(sample_text),
        len(chunks),
        min(token_counts) if token_counts else 0,
        max(token_counts) if token_counts else 0,
        (sum(token_counts) / len(token_counts)) if token_counts else 0.0,
    )
    oversized = [i for i, n in enumerate(token_counts, start=1) if n > 100]
    if oversized:
        logger.warning("以下片段 token 数超过 chunk_size=100：%s", oversized)
    else:
        logger.info("所有片段 token 数均未超过 chunk_size=100")
except ImportError:
    logger.warning("未安装 tiktoken，跳过 token 数核对（pip install tiktoken）")

for i, chunk in enumerate(chunks, 1):
    print(f"\nChunk {i}:")
    print(chunk)
    print("-" * 50)

# --------------------------------------------------------------------------- #
# 观察小结
# --------------------------------------------------------------------------- #
# 1. 对比演示 2（RecursiveCharacterTextSplitter，chunk_size=150 字符）：
#    同样的文档，这里只切出 3 块，且每块的 token 数都 <= 100（实测最大 94），
#    但字符数可达 500 —— 因为 100 token ≈ 400 字符。
#    两种切分器的 chunk_size 数值不能直接比较。
# 2. 切分点可能落在单词/句子中间（token 边界不等价于字符边界或句子边界），
#    所以纯 token 切分的语义完整性不如 Recursive 切分。
#    实践中常把两者结合：用 Recursive 保证语义边界，再用 Token 做长度兜底。
# 3. 若要严格对齐某个非 OpenAI 模型的窗口，应传入该模型对应的分词器，
#    例如：
#      from langchain_text_splitters import TokenTextSplitter
#      splitter = TokenTextSplitter.from_huggingface_tokenizer(
#          tokenizer, chunk_size=100, chunk_overlap=20
#      )
