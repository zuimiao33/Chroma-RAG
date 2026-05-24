"""完整 RAG 流程，整合文档加载、分割、嵌入、存储、检索和生成"""

from typing import List, Optional, Union, Dict, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import config
from src.loader import load_documents
from src.splitter import split_and_process, ChunkSplitter
from src.embedder import get_embedder, LocalEmbeddings, EmbeddingModel
from src.vectorstore import VectorStoreManager, get_vectorstore
from src.retriever import Retriever, MMRRetriever, create_retriever, build_context_from_docs
from src.generator import RAGGenerator, get_generator


class RAGPipeline:
    """完整的 RAG 流程管理器"""

    def __init__(
        self,
        use_local_embed: bool = True,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
    ):
        self.embedder = get_embedder(use_local=use_local_embed)
        self.vectorstore_manager = VectorStoreManager(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embeddings=self.embedder.embeddings,
        )
        self.generator = get_generator()
        self.retriever: Optional[Retriever] = None
        self._indexed_doc_count = 0

    def ingest(
        self,
        source: Union[str, Path],
        recursive: bool = False,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        merge_short: bool = True,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        文档入库流程：加载 -> 分块 -> 嵌入 -> 存储
        """
        source_path = Path(source)
        if show_progress:
            print(f"\n{'='*50}")
            print(f"开始文档入库: {source_path}")
            print(f"{'='*50}")

        if show_progress:
            print("\n[1/4] 加载文档...")
        docs = load_documents(source_path, recursive=recursive)
        if show_progress:
            print(f"  共加载 {len(docs)} 个文档")

        if show_progress:
            print("\n[2/4] 文本分块...")
        chunks = split_and_process(
            docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            merge_short=merge_short,
        )

        if show_progress:
            print("\n[3/4] 生成嵌入向量...")
        self.vectorstore_manager.add_documents(chunks)

        if show_progress:
            print(f"\n[4/4] 存储完成！")
            info = self.vectorstore_manager.get_collection_info()
            print(f"  集合名称: {info['name']}")
            print(f"  总文档数: {info['count']}")
            print(f"  存储路径: {info['persist_directory']}")

        self._indexed_doc_count = len(chunks)
        self._init_retriever()
        return {
            "chunks_count": len(chunks),
            "original_docs": len(docs),
            "collection_info": self.vectorstore_manager.get_collection_info(),
        }

    def _init_retriever(self):
        self.retriever = Retriever(self.vectorstore_manager.vectorstore)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_context: bool = False,
        search_type: str = "similarity",
        mmr_lambda: float = 0.5,
    ) -> Union[str, Dict[str, Any]]:
        """
        问答流程：检索 -> 生成
        """
        if self.retriever is None:
            self._init_retriever()

        if search_type == "mmr":
            retriever = MMRRetriever(
                self.vectorstore_manager.vectorstore,
                top_k=top_k or config.retrieval.default_top_k,
                lambda_mult=mmr_lambda,
            )
            docs = retriever.mmr_search(question)
        else:
            docs = self.retriever.similarity_search(question, top_k=top_k)

        if not docs:
            return "抱歉，知识库中没有找到相关内容。"

        answer = self.generator.generate_with_docs(question, docs)

        if return_context:
            return {
                "answer": answer,
                "sources": build_context_from_docs(docs),
                "doc_count": len(docs),
            }
        return answer

    def query_with_scores(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """带分数的检索，用于调试和分析"""
        if self.retriever is None:
            self._init_retriever()

        results = self.retriever.similarity_search_with_score(question, top_k=top_k)
        answer = self.generator.generate_with_docs(
            question, [doc for doc, _ in results]
        )
        return {
            "answer": answer,
            "retrieval_results": [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source_file", "unknown"),
                    "score": score,
                }
                for doc, score in results
            ],
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        try:
            info = self.vectorstore_manager.get_collection_info()
            return {
                "collection_name": info["name"],
                "total_documents": info["count"],
                "persist_directory": info["persist_directory"],
                "embedding_model": config.embedding.model_name,
                "embedding_dimension": self.embedder.get_dimension(),
                "chunk_size": config.chunking.chunk_size,
                "chunk_overlap": config.chunking.chunk_overlap,
            }
        except Exception as e:
            return {"error": str(e)}

    def delete_by_ids(self, ids: List[str]) -> None:
        self.vectorstore_manager.delete_by_id(ids)

    def clear_knowledge_base(self) -> None:
        self.vectorstore_manager.clear()
        self._indexed_doc_count = 0
        self.retriever = None


def load_existing_pipeline(
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
) -> RAGPipeline:
    """加载已有的知识库 pipeline"""
    pipeline = RAGPipeline(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    pipeline._init_retriever()
    return pipeline
