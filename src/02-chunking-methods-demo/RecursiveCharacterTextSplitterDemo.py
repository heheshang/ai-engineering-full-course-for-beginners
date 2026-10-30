"""演示 2：RecursiveCharacterTextSplitter —— 多级分隔符递归切分（RAG 最常用）。

核心思想：
    给定一组按"优先级从高到低"排列的分隔符，递归地尝试：
      先用分隔符把文本切开 -> 如果某一块仍然超过 chunk_size，
      就换用下一个（更细粒度的）分隔符继续切这块 -> 直到都切不开或已足够小。

    这样做的效果是：优先在语义边界（段落 -> 换行 -> 句子 -> 单词）处断开，
    只有当所有粒度都不行时才做粗暴切分，因此既保结构又不会产生超长块。

特点与适用场景：
    - 优点：通用性最好，RAG 场景的默认选择；能同时满足 chunk_size 上限和语义完整。
    - 缺点：需要多试几轮分隔符，比 CharacterTextSplitter 略慢；
      separator 顺序需要根据文档结构调优。
    - 适用：绝大多数通用文本（Markdown、纯文本、HTML 抽取文本等）。

与 CharacterTextSplitter 的关键差异：
    CharacterTextSplitter 只用 1 个分隔符、只切 1 轮；
    RecursiveCharacterTextSplitter 用 N 个分隔符、切最多 N 轮，
    并且"段落超长时"会继续下探到更细的分隔符。

调参建议：
    - chunk_size：太小则上下文不足，太大则噪声多、成本高。中文文档常见 300~800 字。
    - chunk_overlap：通常取 chunk_size 的 10%~20%，用来兜住跨块的句子。
    - separators：按语义粒度从粗到细排列，例如 ["\\n\\n", "\\n", "。", "！", "？", "，", " "]。
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------------------------------- #
# 输入文档：与演示 1 相同的 Markdown 文档，便于横向对比两种切分器的结果差异
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
print("\n2. Recursive Character Text Splitter Example: Better for preserving context")

splitter = RecursiveCharacterTextSplitter(
    # separators：分隔符优先级列表，从粗粒度到细粒度。
    #   "\n\n" -> 空行分段（段落边界，最优先）
    #   "\n"   -> 单换行（列表项、标题的边界）
    #   ". "  -> 英文句号+空格（句子边界）
    #   " "   -> 空格（单词边界，最后的兜底）
    # 注意：分隔符本身不会保留在输出片段中。
    separators=["\n\n", "\n", ". ", " "],
    # chunk_size=150：块的目标长度上限（字符数）。
    chunk_size=150,
    # chunk_overlap=30：相邻块重叠 30 字符，防止句子被切在边界导致语义割裂。
    chunk_overlap=30,
    # length_function=len：用"字符数"衡量长度。
    # 若要按 token 计，应传入 tiktoken 编码函数（配合 from_tiktoken_encoder 使用），
    # 因为大模型的上下文限制是以 token 为单位的。
    length_function=len,
)

# split_text 返回 List[str]。若需要 Document，可改用 create_documents([sample_text])。
chunks = splitter.split_text(sample_text)

for i, chunk in enumerate(chunks, 1):
    print(f"\nChunk {i}:")
    print(chunk)
    print("-" * 50)

# --------------------------------------------------------------------------- #
# 观察小结
# --------------------------------------------------------------------------- #
# 1. 对比演示 1：这里每个块都 <= 150 字符（实测最大 149），即使原段落更长也会被
#    继续用 "\n" / ". " / " " 拆开，不存在超长块，也没有 "longer than the
#    specified chunk_size" 的告警。
# 2. 标题（# / ## / ###）与它下方的正文倾向于留在同一个块中，语义更完整。
# 3. 代价是块数变多（本例 12 块，演示 1 只有 7 块），向量库更大、
#    检索时的 token 消耗也更高 —— 这是精度与成本的权衡。
