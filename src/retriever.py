"""检索模块，支持语义检索、分数检索和 MMR 多样化检索"""

from typing import List, Optional, Union, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from src.config import config


class Retriever:
    """检索器封装"""

    def __init__(self, vectorstore: Chroma, top_k: Optional[int] = None):
        self.vectorstore = vectorstore
        self.top_k = top_k or config.retrieval.default_top_k

    def similarity_search(
        self, query: str, top_k: Optional[int] = None, **kwargs
    ) -> List[Document]:
        """语义相似度检索"""
        k = top_k or self.top_k
        return self.vectorstore.similarity_search(query, k=k, **kwargs)

    def similarity_search_with_score(
        self, query: str, top_k: Optional[int] = None, **kwargs
    ) -> List[Tuple[Document, float]]:
        """带相关性分数的检索，分数越低相似度越高"""
        k = top_k or self.top_k
        return self.vectorstore.similarity_search_with_score(query, k=k, **kwargs)

    def mmr_search(
        self,
        query: str,
        top_k: int = 20,
        fetch_k: int = 50,
        lambda_mult: float = 0.5,
        **kwargs,
    ) -> List[Document]:
        """
        Max Marginal Relevance 多样化检索

        参数:
            query: 查询文本
            top_k: 最终返回的文档数量
            fetch_k: 从向量库中初筛的数量
            lambda_mult: 多样性权重，0=最大多样性，1=只考虑相关性
                        推荐 0.3-0.7 之间平衡相关性和多样性
        """
        return self.vectorstore.max_marginal_relevance_search(
            query, k=top_k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )

    def get_relevant_documents_with_context(
        self, query: str, top_k: Optional[int] = None
    ) -> List[dict]:
        """返回带上下文字典格式的检索结果"""
        docs = self.similarity_search(query, top_k=top_k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("source_file", "unknown"),
            }
            for doc in docs
        ]


class MMRRetriever(Retriever):
    """专门使用 MMR 多样化检索的检索器"""

    def __init__(
        self,
        vectorstore: Chroma,
        top_k: int = 5,
        fetch_k: int = 30,
        lambda_mult: float = 0.5,
    ):
        super().__init__(vectorstore, top_k=top_k)
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult

    def search(self, query: str) -> List[Document]:
        return self.mmr_search(
            query,
            top_k=self.top_k,
            fetch_k=self.fetch_k,
            lambda_mult=self.lambda_mult,
        )


def create_retriever(
    vectorstore: Chroma,
    search_type: str = "similarity",
    top_k: int = 5,
    **kwargs,
) -> Retriever:
    """工厂函数：创建检索器"""
    if search_type == "mmr":
        return MMRRetriever(
            vectorstore,
            top_k=top_k,
            fetch_k=kwargs.get("fetch_k", 30),
            lambda_mult=kwargs.get("lambda_mult", 0.5),
        )
    return Retriever(vectorstore, top_k=top_k)


def build_context_from_docs(
    documents: List[Document],
    max_chars: int = 3000,
) -> str:
    """将检索到的文档构建为上下文字符串"""
    context_parts = []
    total_chars = 0

    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source_file", "unknown")
        content = doc.page_content.strip()
        part = f"[文档 {i}] 来源: {source}\n{content}"

        if total_chars + len(part) > max_chars:
            break
        context_parts.append(part)
        total_chars += len(part)

    return "\n\n---\n\n".join(context_parts)
