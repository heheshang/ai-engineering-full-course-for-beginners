"""HuggingFace 演示 5：图像理解（Image Understanding）。

核心概念
--------
本例用 **BLIP（Bootstrapping Language-Image Pre-training）** 做「视觉问答（VQA）」：
给定一张图片和一个自然语言问题，模型直接输出答案。

BLIP 的两段式结构（理解本例的关键）：
  1. **视觉编码器（Vision Encoder）**：把图片转成一组视觉特征向量；
  2. **文本解码器（Text Decoder）**：以「图片特征 + 问题」为条件，自回归地生成答案。

因为第 2 步是自回归生成，所以 BLIP 天然支持「同一张图问不同问题」——
这正是本例连续问两个问题的原因。

适用场景
--------
  - 数据集标注辅助（给未标注图片自动打标签）；
  - 图片内容质检、电商商品图自动描述；
  - 给 RAG 系统提供图片描述（图片 -> 文本 -> 进向量库）。

注意
----
``BlipForQuestionAnswering`` 是**问答**专用头，输出直接是答案字符串；
若要生成长描述，需要换成 ``BlipForConditionalGeneration``（本项目其它示例已演示）。

运行前置
--------
  pip install transformers torch pillow requests
"""

import sys
from pathlib import Path

# 把 src 目录加入模块搜索路径，这样无论从哪个工作目录运行脚本，
# 都能 import 到 src/utils 下的公共工具（如日志模块）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import BlipForQuestionAnswering, BlipProcessor  # noqa: E402

from utils.logging_utils import get_logger  # noqa: E402

# 初始化 logger：日志同时输出到控制台和 logs/course.log
logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# 1. 加载模型与处理器
# --------------------------------------------------------------------------- #
# BlipProcessor：负责「预处理」。
#   - 图像：缩放 / 归一化 / 转成模型要求的张量；
#   - 文本：分词、截断、补齐。
#   预处理必须与预训练时完全一致，否则效果会明显下降，所以要和模型配套加载。
#
# 注意：首次运行会从 HuggingFace 下载约 1GB 权重到 ~/.cache/huggingface，需要联网。
MODEL_NAME = "Salesforce/blip-vqa-base"
logger.info("正在加载 BLIP 模型与处理器：%s", MODEL_NAME)

processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME)
model.eval()  # 切到推理模式：关闭 dropout 等训练期行为，结果更稳定

logger.info(
    "模型加载完成，模型参数量约 %.1fM", sum(p.numel() for p in model.parameters()) / 1e6
)

# --------------------------------------------------------------------------- #
# 2. 准备图片
# --------------------------------------------------------------------------- #
# 图片 URL 是 BLIP 官方 demo 用的样例图（一只狗和一个人）。
IMG_URL = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"

logger.info("开始下载示例图片：%s", IMG_URL)
response = requests.get(IMG_URL, stream=True, timeout=30)
# raise_for_status：非 2xx 直接抛异常，避免把 404 的错误页当成图片读进来
response.raise_for_status()

# .raw 是 urllib3 的响应体流，配合 PIL 可以边下边解码，避免整图进内存两次
raw_image = Image.open(response.raw).convert("RGB")
logger.info("图片下载并解码完成，尺寸=%s，模式=%s", raw_image.size, raw_image.mode)


def ask_about_image(image: Image.Image, question: str) -> str:
    """针对一张图片提问，返回模型的自然语言答案。

    Args:
        image: 已解码为 RGB 的 PIL 图片对象。
        question: 自然语言问题，例如 "What is the color of the dog?"。

    Returns:
        模型生成的答案文本（已去掉特殊 token）。
    """
    logger.info("正在提问：%s", question)

    # processor 同时处理图片和问题，产出可直接喂给模型的张量字典
    inputs = processor(image, question, return_tensors="pt")

    # generate 时不需要梯度，用 torch.no_grad() 省显存、提速度
    out = model.generate(**inputs)

    # decode：把生成的 token id 转回字符串；skip_special_tokens 去掉 <pad>/<bos> 等
    answer = processor.decode(out[0], skip_special_tokens=True)
    logger.info("模型回答：%s", answer)
    return answer


# --------------------------------------------------------------------------- #
# 3. 连续问两个问题（同一张图，不同问题）
# --------------------------------------------------------------------------- #
# BLIP 的解码器以上下文为条件生成，因此可以在同一张图上反复提问，
# 每个问题都会得到独立且不同的答案。
questions = [
    "What is the color of the dog?",
    "What is the color of the girl's shirt?",
]

for q in questions:
    ask_about_image(raw_image, q)

logger.info("图像理解演示结束，共完成 %d 次提问", len(questions))
