"""被测系统（SUT）适配层：把课程演示脚本包装成评测可调用的形式。

为什么需要这一层
----------------
评测不能直接跑 ``src/`` 下的演示脚本，因为它们：

1. 是**脚本**而不是库：执行即打印、不可复用；
2. 多数在 import 时就做环境校验（没有 Key 直接 raise）；
3. 每次运行都重新建向量库、重新调模型，评测成本高。

本模块把这些演示脚本的核心能力抽成可复用的函数，
让评测用例既测「真实实现」，又能重复调用、稳定断言。

对应关系
--------
=========================  ==========================================
被测能力                   来源
=========================  ==========================================
``build_rag_pipeline``     ``src/01-rag-demo/RAGDemo.py``
``answer_with_rag``        同上（Agent 问答）
``load_chunk_documents``   ``src/02-chunking-methods-demo/``
``load_csv_documents``     ``src/04-document-loaders-demo/CSVLoaderDemo.py``
``answer_customer_reply``  ``src/03-prompt-engineering-demo/``
=========================  ==========================================
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "src" / "01-rag-demo"


def _require_api_key() -> str:
    """读取并校验 API Key。

    Returns:
        API Key 字符串。

    Raises:
        RuntimeError: 未配置任何可用的 Key。
    """
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("未找到 API Key，请在 .env 中设置 DEEPSEEK_API_KEY")
    # init_chat_model 内部按 provider 自动读取对应变量名
    os.environ["DEEPSEEK_API_KEY"] = api_key
    return api_key


# --------------------------------------------------------------------------- #
# RAG（对应 src/01-rag-demo/RAGDemo.py）
# --------------------------------------------------------------------------- #
# 演示脚本里的示例文档。评测用同一份文档，
# 保证「脚本能跑通」与「评测有标准答案」指向同一份知识。
RAG_DOCUMENT = """Artificial Intelligence (AI) is a branch of computer science focused on
creating systems that can perform tasks that normally require human intelligence.

Machine Learning is a subset of artificial intelligence that enables systems to learn
and improve from experience. It focuses on developing computer programs that can
access data and use it to learn for themselves.

Deep Learning is a specialized subset of machine learning that uses neural networks
with multiple layers to progressively extract higher-level features from raw input.

Reinforcement Learning is a type of machine learning where an agent learns by
receiving rewards or penalties for its actions in an environment. The agent
repeatedly tries different actions and learns which ones lead to better outcomes.

Natural Language Processing (NLP) is a field of AI that enables computers to
understand, interpret, and generate human language. Real-world applications
include chatbots, sentiment analysis, language translation, and text summarization.

Computer Vision is a field of AI that enables computers to interpret and understand
visual information such as images and video. Applications include image recognition,
object detection, facial recognition, and medical image analysis.
"""


def build_rag_pipeline(persist_directory: str | None = None):
    """构建 RAG 检索器（向量库 + 检索器）。

    与 ``RAGDemo.py`` 的实现保持一致：同样的切分参数（chunk_size=200 /
    chunk_overlap=50）、同样的嵌入模型、同样的召回条数（TOP_K=3）。

    **参数必须与演示脚本一致**，否则评测测的就不是课程里展示的那套系统了。

    Args:
        persist_directory: 向量库持久化目录；None 表示使用内存库（评测用，
            避免污染项目的 ``chroma_db/`` 目录）。

    Returns:
        ``VectorStore`` 实例，可调用 ``.as_retriever()``。
    """
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from langchain_core.documents import Document

    logger.info("构建 RAG 向量库（评测模式，内存存储）")

    # 与 RAGDemo.py 保持一致的切分参数
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_text(RAG_DOCUMENT)
    docs = [
        Document(page_content=chunk, metadata={"source": "ai_overview", "chunk": i})
        for i, chunk in enumerate(chunks)
    ]
    logger.info("文档切分完成：%d 个 chunk", len(docs))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    if persist_directory is None:
        # 评测用内存库：进程结束即释放，不污染项目目录
        from langchain_core.vectorstores import InMemoryVectorStore

        return InMemoryVectorStore.from_documents(docs, embeddings)

    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="eval_rag",
    )


def build_rag_agent(retriever):
    """构建 RAG Agent（对应 ``RAGDemo.py`` 的 ``build_agent``）。

    Args:
        retriever: 向量库检索器。

    Returns:
        ``create_agent`` 返回的 Agent 实例。
    """
    from langchain.agents import create_agent
    from langchain.chat_models import init_chat_model
    from langchain_core.tools import create_retriever_tool

    api_key = _require_api_key()

    model = init_chat_model("deepseek:deepseek-chat", temperature=0.0, api_key=api_key)

    search_tool = create_retriever_tool(
        retriever,
        name="search_documents",
        description=(
            "Search the AI overview knowledge base for information about "
            "machine learning, deep learning, reinforcement learning, "
            "natural language processing, and computer vision."
        ),
    )

    # 与 RAGDemo.py 的 SYSTEM_PROMPT 保持一致
    system_prompt = (
        "You are a helpful assistant answering questions about AI.\n"
        "Use the `search_documents` tool to look up information in the knowledge base "
        "before answering. Base your answer only on the retrieved context. If the "
        "context does not contain the answer, say that you don't know instead of "
        "making something up.\n"
        "Focus only on what the question actually asks; do not restate unrelated "
        "parts of the context, and do not add meta-commentary about what the "
        "context does or does not cover."
    )

    logger.info("RAG Agent 构建完成，已注册工具：search_documents")
    return create_agent(model=model, tools=[search_tool], system_prompt=system_prompt)


def answer_with_rag(agent, question: str) -> str:
    """向 RAG Agent 提问并返回最终答案。

    Args:
        agent: ``build_rag_agent`` 返回的 Agent。
        question: 用户问题。

    Returns:
        Agent 的最终回答文本。
    """
    from langchain_core.messages import HumanMessage

    logger.info("向 RAG Agent 提问：%s", question)
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    answer = result["messages"][-1].content
    logger.info("RAG Agent 回答完成，长度=%d 字符", len(answer))
    return answer


def retrieve_context(retriever, question: str, k: int | None = None) -> list[str]:
    """检索并返回上下文文本列表。

    **关键**：这里的 k 必须与 Agent 内部检索器的 k 一致。

    否则会出现「评的上下文」和「模型实际看到的上下文」不是同一批 chunk，
    导致 Faithfulness / ContextualRecall 指标失真——
    模型没被检索到的内容，却被算作「凭空编造」。

    Args:
        retriever: 向量库检索器。
        question: 查询问题。
        k: 召回条数；None 表示沿用检索器自身的配置。

    Returns:
        命中的 chunk 文本列表（顺序即相似度从高到低）。
    """
    if k is None:
        # 沿用 retriever 自身的 search_kwargs（与 Agent 内部保持一致）
        docs = retriever.invoke(question)
    else:
        docs = retriever.invoke(question, **{"k": k})
    logger.info("检索完成：命中 %d 个 chunk", len(docs))
    return [doc.page_content for doc in docs]


# --------------------------------------------------------------------------- #
# 提示工程（对应 src/03-prompt-engineering-demo/）
# --------------------------------------------------------------------------- #
# 约束式输出的提示词。与 ConstrainedOutputPromptingDemo.py 保持一致，
# 评测关注的是「输出是否满足硬性格式约束」。
CONSTRAINED_SYSTEM_PROMPT = """You are a professional customer service representative.
Respond to the customer message using EXACTLY 3 paragraphs.

Mandatory rules:
- Do NOT use any exclamation marks (!).
- Say "I understand" exactly ONCE in the entire response.
- Apologize exactly ONCE, starting the first paragraph with "I'm sorry".
- The third paragraph MUST end with a question.
- Keep the entire response under 200 words."""


def answer_customer_reply(message: str) -> str:
    """用约束式提示词生成客服回复。

    Args:
        message: 客户消息原文。

    Returns:
        模型生成的回答文本。

    Raises:
        RuntimeError: 未配置 API Key。
    """
    from langchain.chat_models import init_chat_model

    api_key = _require_api_key()
    model = init_chat_model("deepseek:deepseek-chat", temperature=0.0, api_key=api_key)

    logger.info("生成约束式客服回复，消息长度=%d 字符", len(message))
    response = model.invoke(
        [
            {"role": "system", "content": CONSTRAINED_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]
    )
    return response.content


def check_constrained_output(text: str) -> dict[str, bool]:
    """检查客服回复是否满足全部硬性约束。

    这是**确定性校验**：不依赖 LLM 评判，可以离线跑、结果稳定。

    Args:
        text: 模型生成的回答。

    Returns:
        约束名 -> 是否满足 的字典。
    """
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    exclamation_count = text.count("!")
    apology_count = text.lower().count("i'm sorry") + text.lower().count("i am sorry")
    understand_count = text.lower().count("i understand")

    return {
        "paragraphs": len(paragraphs) == 3,
        "no_exclamation_marks": exclamation_count == 0,
        "apology_count": apology_count == 1,
        "i_understand_count": understand_count == 1,
        "ends_with_question": bool(text.strip().endswith("?")),
    }


# --------------------------------------------------------------------------- #
# 文档加载（对应 src/04-document-loaders-demo/）
# --------------------------------------------------------------------------- #
def load_csv_documents():
    """用 CSVLoader 加载示例 CSV。

    Returns:
        ``List[Document]``。
    """
    from langchain_community.document_loaders import CSVLoader

    csv_path = PROJECT_ROOT / "src" / "04-document-loaders-demo" / "data" / "products.csv"
    logger.info("加载 CSV：%s", csv_path)
    return CSVLoader(file_path=str(csv_path)).load()


def load_chunk_documents(chunk_size: int = 100, chunk_overlap: int = 20) -> list[str]:
    """按给定参数切分 RAG 示例文档。

    对应 ``src/02-chunking-methods-demo/RecursiveCharacterTextSplitterDemo.py``。

    Args:
        chunk_size: 分块大小。
        chunk_overlap: 重叠大小。

    Returns:
        切分后的 chunk 文本列表。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(RAG_DOCUMENT)
    logger.info("切分完成：%d 个 chunk（chunk_size=%d）", len(chunks), chunk_size)
    return chunks


__all__ = [
    "CONSTRAINED_SYSTEM_PROMPT",
    "RAG_DOCUMENT",
    "answer_customer_reply",
    "answer_with_rag",
    "build_rag_agent",
    "build_rag_pipeline",
    "check_constrained_output",
    "load_chunk_documents",
    "load_csv_documents",
    "retrieve_context",
]
