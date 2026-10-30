"""RAG（检索增强生成）演示，使用最新的 LangChain v1 API。

本文件展示了一个完整的 RAG（Retrieval-Augmented Generation，检索增强生成）流程：
    准备文档 -> 文本切分(chunking) -> 向量化(embeddings) -> 存入向量库 -> 相似度检索 -> Agent 编排问答

LangChain v1 的主要变化（本文件均已采用）：
- ``langchain.chat_models.init_chat_model``：与厂商无关的聊天模型初始化入口，
  换模型只需改一个字符串（如 "deepseek:deepseek-chat" -> "anthropic:claude-sonnet-4-5"）。
- ``langchain.embeddings.init_embeddings``：与厂商无关的向量模型初始化入口。
- ``langchain.agents.create_agent``：基于 LangGraph 的 Agent 工厂函数，取代了旧版的
  ``create_retrieval_chain`` / ``create_react_agent`` 等辅助函数。
- ``langchain_core.tools.create_retriever_tool``：把一个 retriever 包装成 Agent 可调用的工具。

运行方式：
    cd src/01-rag-demo && python RAGDemo.py
"""

import os
import sys
from pathlib import Path

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具。
# 这样无论从哪个工作目录运行 `python src/01-rag-demo/RAGDemo.py`，
# 都能找到 src/utils/logging_utils.py。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.logging_utils import get_logger, log_exception

# 初始化 logger：日志同时输出到控制台和 logs/course.log
logger = get_logger(__name__)

# 读取项目根目录（或当前目录）下的 .env 文件，把环境变量加载进 os.environ。
# 这样 DEEPSEEK_API_KEY 之类的密钥就不需要硬编码在代码里。
load_dotenv()

# 向量库中的集合名：相当于一张"表"的名字，同名集合会复用同一个存储空间。
COLLECTION_NAME = "rag_demo"

# 向量库的持久化目录。用 __file__ 定位，保证无论从哪个工作目录运行脚本，
# 数据都写在脚本同级目录，而不是依赖当前工作目录（相对路径 "./chroma_db" 的老问题）。
PERSIST_DIRECTORY = Path(__file__).parent / "chroma_db"

# 每次检索返回给模型的上下文片段数量（Top-K）。K 越大召回越全，但 token 成本和噪声也越高。
TOP_K = 3

# Agent 的系统提示词：规定角色、强制先检索再回答、禁止编造（幻觉防护）。
# system_prompt 会被作为 SystemMessage 放在对话开头，作用类似于"人设与规则说明"。
#
# 「Focus only on what the question asks」这一句是评测驱动出来的：
# 没有它时，模型会把检索到的每个 chunk 都复述一遍
# （连「上下文没覆盖 Q-learning 等算法」这种元评论也会写进去），
# 导致回答切题分数偏低。加了之后回答才真正聚焦。
SYSTEM_PROMPT = """You are a helpful assistant answering questions about AI.
Use the `search_documents` tool to look up information in the knowledge base before
answering. Base your answer only on the retrieved context. If the context does not
contain the answer, say that you don't know instead of making something up.
Focus only on what the question actually asks; do not restate unrelated parts of
the context, and do not add meta-commentary about what the context does or does
not cover."""


def print_header(step: str, title: str) -> None:
    """打印分步骤标题，让演示输出更易读。

    同时写入一条日志，方便在日志文件里按步骤检索整个流程。

    Args:
        step: 步骤编号文案，如 "Step 1"。
        title: 步骤标题，如 "Preparing our document"。
    """
    print(f"\n{step}: {title}")
    print("-" * 50)
    logger.info("========== %s: %s ==========", step, title)


# --------------------------------------------------------------------------- #
# Step 1: 准备原始文档
# --------------------------------------------------------------------------- #
def load_document() -> str:
    """加载待检索的原始文本。

    真实项目里这里会替换成 PDF / Markdown / 数据库 / 网页爬取等任意数据源，
    但无论来源如何，后续流程（RAG 的三步）都是一样的。

    Returns:
        完整的原始文档字符串。
    """
    print_header("Step 1", "Preparing our document")
    # 示例文档：涵盖 AI、机器学习、深度学习、NLP、计算机视觉、强化学习六个知识点，
    # 便于后面用不同问题验证检索是否命中正确片段。
    document = """Artificial Intelligence (AI) is transforming the way we live and work. Machine learning,
a subset of AI, enables computers to learn from data without explicit programming.
Deep learning, a type of machine learning, uses neural networks inspired by the human brain.

Natural Language Processing (NLP) is a branch of AI that helps computers understand and
process human language. It's used in applications like translation, chatbots, and text analysis.

Computer Vision is another important field in AI. It enables machines to understand and
process visual information from the world, like images and videos. Applications include
facial recognition, autonomous vehicles, and medical image analysis.

Reinforcement Learning is a type of machine learning where agents learn by interacting
with an environment. They receive rewards for good actions and penalties for bad ones.
This is used in game playing, robotics, and autonomous systems.
"""
    logger.info("文档加载完成，共 %d 字符", len(document))
    print("Document loaded. Length:", len(document), "characters")
    print("\nPreview of the document:")
    print(document[:200], "...\n")
    return document


# --------------------------------------------------------------------------- #
# Step 2: 文本切分（Chunking）
# --------------------------------------------------------------------------- #
def split_document(document: str) -> list[Document]:
    """把长文档切成若干语义完整的小片段，并包装成 Document 对象。

    为什么要切分？
      1. 大模型的上下文窗口有限，且整篇文档塞进去会稀释注意力、抬高成本；
      2. 检索粒度越细，命中的片段越精确，检索效果（召回率/精确率）越好；
      3. 向量模型对超长文本的语义表征会变差。

    参数说明：
      - chunk_size=200：每个片段的目标字符数上限。
      - chunk_overlap=50：相邻片段之间重叠的字符数。重叠的作用是防止句子正好被
        切断在边界上，导致语义被割裂（同一句话的前后半部分落在不同片段里）。
      - length_function=len：按"字符数"而不是"token 数"来衡量长度。
      - separators：递归切分的优先级，依次尝试用这些分隔符把文本拆开，
        优先保留段落/换行等结构信息，最后才退化为按空格硬切。

    Args:
        document: 原始文档全文。

    Returns:
        Document 列表，每个 Document 含 page_content（正文）和 metadata（元数据）。
    """
    print_header("Step 2", "Chunking the document")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " "],
    )
    # split_text 返回纯字符串列表；真实项目里也可以用 create_documents
    # （配合 splitter 对应的 splitters 元数据）自动生成 Document。
    chunks = text_splitter.split_text(document)

    # 把字符串转成 LangChain 统一的 Document 结构，并附带元数据：
    # source 用于标明文档来源（多知识库场景下做溯源），chunk 用于标记片段序号。
    docs = [
        Document(page_content=chunk, metadata={"source": "ai_overview", "chunk": i})
        for i, chunk in enumerate(chunks)
    ]

    # 统计每个片段的长度分布，用于观察 chunk_size / chunk_overlap 的实际效果
    lengths = [len(d.page_content) for d in docs]
    logger.info(
        "文档切分完成：共 %d 个片段，最短 %d 字符，最长 %d 字符，平均 %.1f 字符",
        len(docs),
        min(lengths) if lengths else 0,
        max(lengths) if lengths else 0,
        (sum(lengths) / len(lengths)) if lengths else 0.0,
    )

    print(f"Document has been split into {len(docs)} chunks.")
    print("\nExample chunks:")
    for i, doc in enumerate(docs[:3], start=1):
        print(f"\nChunk {i}:")
        print(doc.page_content)
        print("-" * 30)
    return docs


# --------------------------------------------------------------------------- #
# Step 3: 向量化（Embeddings）
# --------------------------------------------------------------------------- #
def create_embeddings():
    """初始化向量模型，把文本转换成稠密向量（dense vector）。

    这里用的是 all-MiniLM-L6-v2：体量小、速度快、本地即可运行，适合教学示例；
    生产环境可以换成 OpenAI / 通义 / BGE 等更强的模型（只需改 init_embeddings 的参数）。

    Returns:
        Embeddings 实例，遵循 langchain_core 的 Embeddings 接口。
    """
    print_header("Step 3", "Creating embeddings")
    # init_embeddings 的参数格式为 "厂商:模型名"，厂商可选：
    # openai / huggingface / azure_openai / bedrock / google_vertexai 等。
    embeddings = init_embeddings("huggingface:sentence-transformers/all-MiniLM-L6-v2")
    logger.info("嵌入模型初始化完成：all-MiniLM-L6-v2")
    # 打印向量维度：all-MiniLM-L6-v2 输出 384 维；这决定了向量库里每条记录的长度。
    dimension = len(embeddings.embed_query("test"))
    logger.info("嵌入向量维度=%d（向量库中每条记录的长度）", dimension)
    print("Embedding dimension:", dimension)
    return embeddings


# --------------------------------------------------------------------------- #
# Step 4: 构建向量库
# --------------------------------------------------------------------------- #
def create_vector_store(docs: list[Document], embeddings):
    """创建/复用 Chroma 向量库并写入文档。

    向量库做的事：保存每个片段的向量 + 原文 + 元数据，
    查询时把问题也转成向量，用余弦相似度找出最接近的 Top-K 片段。

    Args:
        docs: 切分后的 Document 列表。
        embeddings: 向量模型实例。

    Returns:
        Chroma 向量库实例。
    """
    print_header("Step 4", "Creating vector store")
    from langchain_chroma import Chroma

    # 确保持久化目录存在，否则 Chroma 首次写入会报错。
    PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # LangChain v1 推荐直接用构造函数创建向量库（from_texts/from_documents
    # 属于便捷类方法）。collection_name 相当于表名，persist_directory 落盘位置。
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIRECTORY),
    )

    # 幂等处理：先检查集合里是否已有数据。
    # 直接用 from_texts 每次运行都会重复写入，导致向量库膨胀、检索结果出现重复。
    existing = vectorstore.get()["ids"]
    if existing:
        logger.info("集合 '%s' 已存在（%d 条文档），跳过写入直接复用", COLLECTION_NAME, len(existing))
        print(
            f"Collection '{COLLECTION_NAME}' already exists ({len(existing)} docs). Reusing it."
        )
    else:
        vectorstore.add_documents(docs)
        logger.info("已向集合 '%s' 写入 %d 个文档", COLLECTION_NAME, len(docs))
        print(f"Added {len(docs)} documents to the collection.")

    total = len(vectorstore.get()["ids"])
    logger.info("向量库就绪：集合=%s，总文档数=%d，存储目录=%s", COLLECTION_NAME, total, PERSIST_DIRECTORY)
    print("Vector store created with following details:")
    print(f"- Number of texts: {total}")
    print(f"- Database location: {PERSIST_DIRECTORY}")
    return vectorstore


# --------------------------------------------------------------------------- #
# Step 5: 相似度检索测试
# --------------------------------------------------------------------------- #
def test_similarity_search(vectorstore) -> None:
    """验证"检索"这一步本身是否有效（不经过大模型）。

    这是 RAG 的核心环节：把用户问题向量化后，在向量库中找语义最相近的片段。
    只有检索准，后面让大模型基于这些片段作答才有意义。

    Args:
        vectorstore: 向量库实例。
    """
    print_header("Step 5", "Testing similarity search")
    query = "What is reinforcement learning?"
    # similarity_search：返回最相似的 k 个 Document（内部做 embedding + 相似度排序）。
    results = vectorstore.similarity_search(query, k=2)
    logger.info("相似度检索完成：query=%r, k=2, 命中 %d 个片段", query, len(results))

    print(f"Query: {query}")
    print("\nTop 2 most relevant chunks:")
    for i, doc in enumerate(results, start=1):
        print(f"\nResult {i}:")
        print(doc.page_content)
        print("-" * 30)


# --------------------------------------------------------------------------- #
# Step 6: Agentic RAG（智能体式检索增强生成）
# --------------------------------------------------------------------------- #
def build_agent(vectorstore):
    """构建基于 Agent 的 RAG 系统。

    与传统"固定管线"（检索 -> 塞进 prompt -> 生成）不同，Agentic RAG 让模型
    自己决定要不要检索、检索什么、检索几次，甚至在检索结果不相关时换个说法再检索，
    从而减少无效检索、降低 token 消耗。

    Args:
        vectorstore: 向量库实例。

    Returns:
        编译好的 Agent（基于 LangGraph 的状态图），可直接 .invoke()。
    """
    print_header("Step 6", "Setting up the agentic RAG pipeline")
    # 提前校验密钥，避免在调用时才抛出难以定位的鉴权错误。
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is not set. Add it to your .env file.")
    logger.info("检测到 DEEPSEEK_API_KEY，开始初始化对话模型")

    # init_chat_model("厂商:模型名") 是 v1 推荐的统一入口。
    # 换成 OpenAI 就是 "openai:gpt-4.1"，换成 Anthropic 就是 "anthropic:claude-sonnet-4-5"，
    # 业务代码完全不用改。temperature 越低输出越稳定，越高越发散。
    model = init_chat_model("deepseek:deepseek-chat", temperature=0.5)
    logger.info("对话模型初始化完成：deepseek:deepseek-chat, temperature=0.5")

    # as_retriever 把向量库包装成标准 Retriever；search_kwargs={"k": N} 指定召回条数。
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # create_retriever_tool 把 Retriever 包装成一个 Agent 可调用的"工具"（Tool）：
    # 模型看到工具名 + 描述后，会生成 tool_call 去调用它，
    # 工具内部执行检索并把结果作为 ToolMessage 回传给模型。
    # name/description 是模型"看到"的部分，写得越清楚，模型越知道什么时候该调用它。
    search_tool = create_retriever_tool(
        retriever,
        name="search_documents",
        description=(
            "Search the AI knowledge base for information about artificial intelligence, "
            "machine learning, deep learning, NLP, computer vision and reinforcement "
            "learning. Use this before answering domain questions."
        ),
    )

    # create_agent 组装出完整的 ReAct 循环：
    #   模型思考 -> 需要工具则发起 tool_call -> 执行工具 -> 结果回填 -> 直到给出最终答案。
    agent = create_agent(
        model=model,
        tools=[search_tool],
        system_prompt=SYSTEM_PROMPT,
    )
    logger.info("Agent 构建完成，已注册工具：%s", [search_tool.name])
    print("Agent created with tools:", [search_tool.name])
    return agent


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    """串起完整流程：切分 -> 向量化 -> 入库 -> 检索验证 -> Agent 问答。"""
    # Step 1-2：准备文档并切分
    document = load_document()
    docs = split_document(document)

    # Step 3-4：向量化并写入向量库
    embeddings = create_embeddings()
    vectorstore = create_vector_store(docs, embeddings)

    # Step 5：先单独验证检索效果
    test_similarity_search(vectorstore)

    # Step 6：构建 Agent
    agent = build_agent(vectorstore)

    # 三个测试问题，分别对应强化学习 / 计算机视觉 / NLP 三个知识点。
    questions = [
        "What is reinforcement learning and how does it work?",
        "What are the main applications of computer vision?",
        "How is NLP used in real-world applications?",
    ]

    print("\nAsking questions to our RAG system:")
    for idx, question in enumerate(questions, start=1):
        print("\nQuestion:", question)
        logger.info("[%d/%d] 向 Agent 提问：%s", idx, len(questions), question)
        try:
            # Agent 的输入是消息列表；返回值同样是消息列表，
            # 最后一条即为模型的最终回答（AIMessage）。
            result = agent.invoke({"messages": [HumanMessage(content=question)]})
            answer = result["messages"][-1].content
            print("Answer:", answer)
            # 记录本次调用的 token 用量，便于统计成本
            usage = getattr(result["messages"][-1], "usage_metadata", None)
            if usage:
                logger.info(
                    "[%d/%d] 回答完成，token 用量：输入=%s，输出=%s，合计=%s",
                    idx, len(questions),
                    usage.get("input_tokens"), usage.get("output_tokens"),
                    usage.get("total_tokens"),
                )
            else:
                logger.info("[%d/%d] 回答完成，长度=%d 字符（未返回 token 用量）",
                            idx, len(questions), len(answer))
        except Exception as exc:  # noqa: BLE001 - 演示脚本，单个问题失败不影响其余
            # 单个问题失败不应中断整个演示：记录完整堆栈后继续下一个问题
            log_exception(logger, "[%d/%d] 回答问题失败：%s", idx, len(questions), question)
            print("Error getting answer:", exc)

    logger.info("RAG 演示流程全部结束")


if __name__ == "__main__":
    main()
