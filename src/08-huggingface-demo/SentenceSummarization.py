"""HuggingFace 演示 1：句子摘要（Sentence Summarization）。

核心概念
--------
**摘要（Summarization）** 把一段长文本压缩成一句更短的概括性文本，保留主干信息。

本例用的是 **BART（Bidirectional and Auto-Regressive Transformer）**：
  - **编码器**（双向注意力）负责「读懂」整段输入；
  - **解码器**（自回归）负责「逐词写出」摘要。
  训练目标是「噪声 -> 原文」的还原任务，因此天生擅长生成**忠实于原文**的摘要。

两种主流摘要范式
----------------
  1. **抽取式（Extractive）**：直接从原文中挑句子拼成摘要。
     优点：绝不会产生原文没有的信息（零幻觉）；缺点：句子可能读起来别扭。
  2. **生成式（Abstractive）**：重新组织语言生成新句子（本例属于这种）。
     优点：更通顺、更紧凑；缺点：可能"编造"原文没有的事实（需人工/规则校验）。

本例也演示了另一个 HuggingFace 常用范式：**text-generation pipeline**
（``facebook/bart-large-cnn``），它可以在给定前缀的条件下续写文本，
用来做「引导式摘要」（给一个引导句，让模型朝指定方向总结）。

运行前置
--------
  pip install transformers torch
"""

import sys
from pathlib import Path

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import pipeline  # noqa: E402

from utils.logging_utils import get_logger  # noqa: E402

# 初始化 logger
logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# 1. 加载摘要模型
# --------------------------------------------------------------------------- #
# facebook/bart-large-cnn 是 CNN/DailyMail 数据集上微调过的 BART-large，
# 专门针对「新闻摘要」优化，产出的摘要通常一句到三句。
#
# pipeline 做了三件事的封装：
#   1. 分词（tokenize）
#   2. 模型推理（generate）
#   3. 解码与后处理（decode + 截断到完整句子）
#   我们直接拿到的就是可读文本。
MODEL_NAME = "facebook/bart-large-cnn"
logger.info("正在加载摘要模型（pipeline）：%s", MODEL_NAME)
summarizer = pipeline("text-generation", model=MODEL_NAME)
logger.info("摘要模型加载完成")

# --------------------------------------------------------------------------- #
# 2. 准备待摘要的长文本
# --------------------------------------------------------------------------- #
# 真实的新闻正文通常比这个长得多（几千字），
# 而 BART 的输入上限约 1024 token，超出部分需要自己截断或分段处理。
ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney's Office by Immigration and Customs Enforcement and the Department of Homeland Security's
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""

logger.info("待摘要文本长度：%d 字符", len(ARTICLE))

# --------------------------------------------------------------------------- #
# 3. 生成摘要
# --------------------------------------------------------------------------- #
# 参数说明：
#   - max_length=130：生成文本的最大 token 数（不是字符数）。
#     这是硬上限，超出会被截断。
#   - min_length=30：生成的最小 token 数，防止模型过早停在一个残缺的短句上。
#     实践建议 min ≈ max 的 1/4 ~ 1/3。
#   - do_sample=False：**贪心解码**（每步都取概率最大的 token），
#     结果确定、可复现；设为 True 则为采样解码，输出更多样但每次不同。
#
# 另外，text-generation pipeline 会在输入后面接上模型的特殊 token，
# 因此返回结果里通常同时包含「原文开头 + 摘要」，
# 所以下面用 split("\n\n") 之类的方式把摘要部分切出来时要注意分隔符。
logger.info("开始生成摘要（max_length=130, min_length=30, do_sample=False）")
result = summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False)

# pipeline 返回 List[dict]，每个 dict 形如 {"generated_text": "..."}
generated_text = result[0]["generated_text"]
logger.info("摘要生成完成，全文长度=%d 字符", len(generated_text))

# 把摘要部分单独取出来（原始输入以 "\n\n" 之后为模型续写部分）
# 这里做一次健壮的切分：找不到分隔符时退化为输出全文，避免抛异常
summary_part = (
    generated_text.split("\n\n", 1)[1] if "\n\n" in generated_text else generated_text
)
summary_part = summary_part.strip()

print(result)

print("\n" + "=" * 70)
print("仅摘要部分：")
print("=" * 70)
print(summary_part)

logger.info(
    "压缩率：原文 %d 字符 -> 摘要 %d 字符（保留 %.1f%%）",
    len(ARTICLE),
    len(summary_part),
    len(summary_part) / max(len(ARTICLE), 1) * 100,
)
logger.info("句子摘要演示结束")
