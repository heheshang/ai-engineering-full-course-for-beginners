"""HuggingFace 演示 3：句子向量（Sentence Embeddings）。

核心概念
--------
**句向量（Sentence Embedding）** 把一句话映射成一个固定长度的稠密向量，
例如 384 维。语义相近的两句话，其向量在高维空间中的距离也更近。

这正是 RAG「语义检索」的基础：
  1. 把知识库切成若干段，每段算一个向量存进向量库；
  2. 用户提问时，把问题也转成向量；
  3. 用余弦相似度找出最接近的若干段落，交给大模型作答。

对比 TF-IDF / BM25（关键词检索）：
  - 关键词检索靠字面匹配，「快递延误」匹配不上「配送延迟」；
  - 向量检索理解语义，同义表达也能召回。

本例使用的 ``all-MiniLM-L6-v2``
-----------------------------
  - 参数量仅 22M，输出 **384 维**向量，CPU 上推理很快；
  - 训练语料含 10 亿+ 对句子，句向量检索（Semantic Search）表现优秀；
  - 许可 Apache-2.0，可商用；
  - 是 LangChain 默认的本地嵌入模型，本项目其它 demo 也复用它。

运行前置
--------
  pip install sentence-transformers torch
"""

import sys
from pathlib import Path

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentence_transformers import SentenceTransformer, util
from transformers.utils import logging as hf_logging

from utils.logging_utils import get_logger

# 初始化 logger
logger = get_logger(__name__)

# 把 HuggingFace 自身的日志级别调到 ERROR，只保留真正的错误，
# 避免 "All model checkpoint shards were used" 这类无用信息刷屏。
hf_logging.set_verbosity_error()

# --------------------------------------------------------------------------- #
# 1. 加载嵌入模型
# --------------------------------------------------------------------------- #
MODEL_NAME = "all-MiniLM-L6-v2"
logger.info("正在加载嵌入模型：%s", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)
# 模型维度：决定后续向量库中每条记录的长度（all-MiniLM-L6-v2 = 384）
dimension = model.get_sentence_embedding_dimension()
logger.info("模型加载完成，输出向量维度=%d", dimension or 0)

# --------------------------------------------------------------------------- #
# 2. 编码两组句子
# --------------------------------------------------------------------------- #
# 编码（encode）：把 List[str] 变成一个 N x 384 的矩阵（torch.Tensor）。
#
# 参数说明：
#   - convert_to_tensor=True：直接返回 torch 张量，方便后续用 util.cos_sim 计算；
#     若返回 numpy 数组则更适合做持久化 / 送入 sklearn 等传统机器学习库。
sentences1 = [
    "The cat sits outside",
    "A man is playing guitar",
    "The movies are awesome",
]

logger.info("正在编码第一组句子（%d 条）：%s", len(sentences1), sentences1)
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
logger.info("第一组编码完成，shape=%s", tuple(embeddings1.shape))

sentences2 = [
    "The dog plays in the garden",
    "A woman watches TV",
    "The new movie is so great",
]

logger.info("正在编码第二组句子（%d 条）：%s", len(sentences2), sentences2)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)
logger.info("第二组编码完成，shape=%s", tuple(embeddings2.shape))

# 打印第一组向量，看看 384 维浮点数的真实样子
# 实际项目中通常不会打印完整向量（太占屏幕），这里只为教学展示
print(embeddings1)

# --------------------------------------------------------------------------- #
# 3. 计算余弦相似度矩阵
# --------------------------------------------------------------------------- #
# util.cos_sim(A, B)：逐行计算 A 与 B 的余弦相似度，输出 (len(A), len(B)) 的矩阵。
#
# 余弦相似度公式：cos(a, b) = (a · b) / (|a| * |b|)
#   - 值域 [-1, 1]，越接近 1 越相似；
#   - 归一化后不受向量长度影响，只看方向（即语义方向）。
cosine_scores = util.cos_sim(embeddings1, embeddings2)
logger.info("余弦相似度矩阵计算完成，shape=%s", tuple(cosine_scores.shape))
print(cosine_scores)

# --------------------------------------------------------------------------- #
# 4. 逐对查看相似度并解读
# --------------------------------------------------------------------------- #
# 相似度的经验判断标准：
#   > 0.9  几乎同义
#   > 0.7  强相关
#   > 0.5  主题相关
#   < 0.3  基本无关
logger.info("开始逐对查看相似度：")
for i in range(len(sentences1)):
    score = float(cosine_scores[i][i])
    logger.info(
        "  %-30s <-> %-32s 余弦相似度=%.4f",
        sentences1[i],
        sentences2[i],
        score,
    )
    print(f"{sentences1[i]} \t\t {sentences2[i]} \t\t Score: {score:.4f}")

# 解读要点：
#   - 第 1 对（猫在户外 / 狗在花园里）主题都是"动物 + 户外活动"，相似度较高；
#   - 第 3 对（电影很棒 / 新电影太棒了）几乎是同义表达，相似度最高；
#   - 这正说明向量模型捕捉到了**语义**而非字面重合，
#     "movies" 与 "movie" 的差异不影响第 3 对的得分。
logger.info("句子向量演示结束")
