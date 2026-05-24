"""文本分割模块，支持递归字符分割和 Markdown 语义分割"""

from typing import List, Optional, Callable

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)

from src.config import config


class ChunkSplitter:
    """文本分割器，支持多种分割策略"""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or config.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunking.chunk_overlap

        self.char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )

    def split_documents(
        self, documents: List[Document]
    ) -> List[Document]:
        """分割文档列表，返回 chunk 列表"""
        return self.char_splitter.split_documents(documents)

    def split_text(self, text: str) -> List[str]:
        """分割单段文本"""
        return self.char_splitter.split_text(text)

    def merge_short_chunks(
        self, documents: List[Document], min_chunk_size: int = None
    ) -> List[Document]:
        """合并过短的 chunks，减少碎片"""
        min_size = min_chunk_size or (self.chunk_size // 2)
        merged: List[Document] = []
        buffer: Optional[Document] = None

        for doc in documents:
            if buffer is None:
                buffer = doc
                continue

            if len(doc.page_content) < min_size:
                buffer.page_content += doc.page_content
                buffer.metadata.update(doc.metadata)
            else:
                merged.append(buffer)
                buffer = doc

        if buffer is not None:
            merged.append(buffer)

        return merged

    def get_chunk_stats(self, documents: List[Document]) -> dict:
        """获取分块统计信息"""
        if not documents:
            return {"count": 0, "avg_length": 0, "min_length": 0, "max_length": 0}

        lengths = [len(doc.page_content) for doc in documents]
        return {
            "count": len(documents),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }


class MarkdownSplitter:
    """Markdown 专用分割器，按标题层级结构分割"""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or config.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunking.chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n## ",
                "\n### ",
                "\n#### ",
                "\n\n",
                "\n",
                "。",
                " ",
                "",
            ],
        )

    def split_markdown(self, text: str, metadata: dict = None) -> List[Document]:
        """分割 Markdown 文本，保留标题层级信息"""
        docs = self.splitter.create_documents([text], metadatas=[metadata or {}])
        return docs


def split_and_process(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    merge_short: bool = True,
) -> List[Document]:
    """便捷函数：分割并可选合并短块"""
    splitter = ChunkSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    stats = splitter.get_chunk_stats(chunks)
    print(f"  分块完成: {stats['count']} 个 chunks, "
          f"平均长度 {stats['avg_length']:.1f} 字符, "
          f"范围 [{stats['min_length']}, {stats['max_length']}]")

    if merge_short:
        chunks = splitter.merge_short_chunks(chunks)
        print(f"  合并短块后: {len(chunks)} 个 chunks")

    return chunks
