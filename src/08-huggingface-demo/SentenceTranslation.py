"""HuggingFace 演示 4：句子翻译（Sentence Translation）。

核心概念
--------
本例用 **NLLB-200**（No Language Left Behind）做「多语言翻译」。
它的特点是：

  - 一个模型支持 **200 种语言**互译，不需要为每对语言单独训练模型；
  - 关键机制是 **目标语言 token（``forced_bos_token_id``）**：
    翻译方向不是靠「你把源语言写在前面」来暗示的，而是靠
    在生成时**强制**模型先输出一个目标语言标记（如 ``spa_Latn`` = 西班牙语拉丁文写），
    之后的所有 token 都属于该语言。

  NLLB 的语言标记格式为 ``<lang>_<script>``，例如：
      ``eng_Latn``   英语（拉丁字母）
      ``spa_Latn``  西班牙语（拉丁字母）
      ``zho_Hans``  简体中文（简体）
      ``fra_Cyrl``  法语（西里尔字母）

  这样一来，同一个源文本只要换一个 ``forced_bos_token_id``，
  就能翻译成完全不同的语言——这正是本例的核心玩法。

适用场景
--------
  - 多语言客服系统：一份知识库，多种语言回答；
  - 内容本地化：产品文档 / 营销文案批量翻译；
  - 数据增强：把语料翻译成多语种，提升 embedding 模型的跨语言检索能力。

运行前置
--------
  pip install transformers torch sentencepiece
  （NLLB 的分词器依赖 SentencePiece，必须安装）
"""

import sys
from pathlib import Path

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402

from utils.logging_utils import get_logger  # noqa: E402

# 初始化 logger
logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# 1. 加载分词器与模型
# --------------------------------------------------------------------------- #
# NLLB-200-distilled-600M 是官方发布的「蒸馏版」：
#   - 600M 参数（完整版 NLLB-200 是 2.4B），显存需求大幅降低；
#   - 质量略低于完整版，但普通业务场景足够用；
#   - 适合本地部署与教学演示。
#
# 注意：AutoTokenizer / AutoModelForSeq2SeqLM 会自动从模型名推断类型，
# 换成 facebook/nllb-200-distilled-1.3B 即可换成更大更强的版本。
MODEL_NAME = "facebook/nllb-200-distilled-600M"
logger.info("正在加载翻译模型：%s", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()  # 推理模式：关闭 dropout 等训练期行为

logger.info(
    "模型加载完成，模型参数量约 %.1fM", sum(p.numel() for p in model.parameters()) / 1e6
)

# --------------------------------------------------------------------------- #
# 2. 设置源语言
# --------------------------------------------------------------------------- #
# src_lang 告诉分词器「输入是哪种语言」，
# 它会影响把原文切分并映射到 NLLB 统一的语言中立词表。
SRC_LANG = "eng_Latn"
tokenizer.src_lang = SRC_LANG
logger.info("源语言已设置为：%s", SRC_LANG)

# --------------------------------------------------------------------------- #
# 3. 准备源文本并分词
# --------------------------------------------------------------------------- #
SOURCE_TEXT = (
    "My puppy is adorable, Your kitten is cute. "
    "Her panda is friendly. His llama is thoughtful. We all have nice pets!"
)
logger.info("源文本（%s）：%s", SRC_LANG, SOURCE_TEXT)

# tokenizer(...) 完成：
#   - 按 src_lang 切分并映射为 token id；
#   - 加上语言标记 token；
#   - return_tensors="pt" 输出 PyTorch 张量（"pt" = PyTorch）。
inputs = tokenizer(SOURCE_TEXT, return_tensors="pt")
logger.info("分词完成，input_ids shape=%s", tuple(inputs["input_ids"].shape))

# --------------------------------------------------------------------------- #
# 4. 生成译文
# --------------------------------------------------------------------------- #
# forced_bos_token_id：强制生成序列的**第一个** token 为指定语言标记。
#   convert_tokens_to_ids("spa_Latn") 把语言标记字符串转成模型词表里的 id。
#   这是 NLLB 决定输出语言的关键参数，不设置的话模型会随机选一种语言。
#
# max_length=512：生成的最大 token 数，防止异常情况下无限生成。
TARGET_LANG = "spa_Latn"
target_token_id = tokenizer.convert_tokens_to_ids(TARGET_LANG)
logger.info("目标语言：%s（token_id=%d）", TARGET_LANG, target_token_id)

# num_beams=4：束搜索（beam search）宽度为 4。
#   逐词解码时不再只保留最优一条路径，而是保留 4 条候选，
#   最终选整体概率最高的一条 -> 译文质量通常优于贪心解码，代价是慢约 4 倍。
translated_tokens = model.generate(
    **inputs,
    forced_bos_token_id=target_token_id,
    max_length=512,
    num_beams=4,
)

# batch_decode：把生成的 token id 转回字符串；
# skip_special_tokens=True 去掉语言标记、<pad> 等特殊 token
result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
logger.info("翻译完成，结果：%s", result)

print(result)

# --------------------------------------------------------------------------- #
# 5. 多语言对照（切换 forced_bos_token_id 即可）
# --------------------------------------------------------------------------- #
# 演示把同一段文本翻译成多种语言，验证「一个模型 + 换一个 token = 换一种语言」。
logger.info("开始多语言批量翻译演示：")
for lang in ("spa_Latn", "fra_Latn", "zho_Hans", "deu_Latn"):
    out = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(lang),
        max_length=512,
        num_beams=4,
    )
    translated = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    logger.info("  %-10s -> %s", lang, translated)
    print(f"[{lang}] {translated}")

logger.info("句子翻译演示结束")
