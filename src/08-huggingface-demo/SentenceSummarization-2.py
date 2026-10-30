"""HuggingFace 演示 2：句子摘要（bfloat16 精度 + 短文本场景）。

本文件与 ``SentenceSummarization.py`` 是同一套模型、同一套 pipeline 的两个变体，
差别只在于**精度设置**和**输入长度**。

一、为什么要用 bfloat16
----------------------
PyTorch 支持多种数值精度：

  ==========  =========  ==========  ====================================
  精度         字节/参数  相对速度    说明
  ==========  =========  ==========  ====================================
  float32      4          1.0x       默认精度，最精确也最占显存
  float16      2          1.5~2x     精度够用，但数值范围小，容易溢出/NaN
  bfloat16     2          1.5~2x     与 float32 相同的**指数位**，数值范围一样大，
                                        但尾数位被截断 -> 精度略低、极稳定
  ==========  =========  ==========  ====================================

  bfloat16 的优势：
    - 显存占用只有 float32 的一半，能跑更大的 batch / 更长的文本；
    - 动态范围与 float32 相同，训练/推理时不会出现 float16 那种溢出为 inf 的问题；
    - 现代 GPU（RTX 30 系及以上、A100/H100、Apple Silicon）对 bfloat16 有原生加速。

  注意：``torch_dtype=bfloat16`` 只影响**模型权重**的精度，
  输入张量仍由 pipeline 按默认精度构造，属于半精度推理的标准用法。

二、本例的输入特点
------------------
输入是一段结构紧凑、长度适中的百科式短文（约 5 行），
适合直接一次性喂进模型。这种「短文本整段摘要」是常见需求，
比如产品描述、新闻导语、会议纪要的一句话概括。

运行前置
--------
  pip install transformers torch
"""

import sys
from pathlib import Path

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import pipeline

from utils.logging_utils import get_logger

# 初始化 logger
logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# 1. 加载模型（bfloat16 权重）
# --------------------------------------------------------------------------- #
MODEL_NAME = "facebook/bart-large-cnn"
logger.info("正在加载摘要模型（pipeline）：%s，权重精度=bfloat16", MODEL_NAME)

# 检查当前硬件是否支持 bfloat16，给出明确的日志说明而不是让用户困惑
if torch.cuda.is_available():
    logger.info("检测到 CUDA 设备：%s", torch.cuda.get_device_name(0))
else:
    logger.info("未检测到 CUDA 设备，将在 CPU 上以 bfloat16 运行（速度较慢但结果一致）")

# torch_dtype=torch.bfloat16：把模型权重以 bfloat16 加载，显存占用减半
summarizer = pipeline(task="text-generation", model=MODEL_NAME, torch_dtype=torch.bfloat16)
logger.info("摘要模型加载完成")

# --------------------------------------------------------------------------- #
# 2. 准备待摘要的短文本
# --------------------------------------------------------------------------- #
text = """Paris is the capital and most populous city of France, with
          an estimated population of 2,175,601 residents as of 2018,
          in an area of more than 105 square kilometres (41 square
          miles). The City of Paris is the centre and seat of
          government of the region and province of Île-de-France, or
          Paris Region, which has an estimated population of
          12,174,880, or about 18 percent of the population of France
          as of 2017."""

logger.info("待摘要文本长度：%d 字符", len(text))

# --------------------------------------------------------------------------- #
# 3. 生成摘要
# --------------------------------------------------------------------------- #
# 与上一个示例的参数差异：
#   - 这里没有显式传 do_sample，默认走采样解码（do_sample=True），
#     输出带有随机性，多次运行结果可能不同；
#   - min_length=10 / max_length=100：针对短文本放宽了长度区间，
#     避免模型把 7 行输入压成过短的片段。
#
# 如果需要结果可复现（生产环境、回归测试），请显式传 do_sample=False。
logger.info("开始生成摘要（min_length=10, max_length=100）")
summary = summarizer(text, min_length=10, max_length=100)

generated_text = summary[0]["generated_text"]
# 与上一个文件同理：text-generation pipeline 的输出包含原文 + 续写部分
summary_part = generated_text.split("\n\n", 1)[1] if "\n\n" in generated_text else generated_text
summary_part = summary_part.strip()

print(summary)

print("\n" + "=" * 70)
print("仅摘要部分：")
print("=" * 70)
print(summary_part)

logger.info(
    "压缩率：原文 %d 字符 -> 摘要 %d 字符（保留 %.1f%%）",
    len(text),
    len(summary_part),
    len(summary_part) / max(len(text), 1) * 100,
)
logger.info("bfloat16 精度摘要演示结束")
