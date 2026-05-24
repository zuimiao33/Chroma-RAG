"""向量存储模块，基于 Chroma 实现文档的持久化向量存储"""

from typing import List, Optional, Union, Dict, Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from src.config import config


class VectorStoreManager:
    """Chroma 向量库管理器"""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
    ):
        self.persist_directory = persist_directory or config.chroma.persist_directory
        self.collection_name = collection_name or config.chroma.collection_name
        self._embeddings = embeddings
        self._vectorstore: Optional[Chroma] = None

    @property
    def vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self._embeddings,
                collection_name=self.collection_name,
            )
        return self._vectorstore

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        """添加文档到向量库"""
        return self.vectorstore.add_documents(
            documents=documents, ids=ids, **kwargs
        )

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        """添加文本到向量库"""
        return self.vectorstore.add_texts(
            texts=texts, metadatas=metadatas, ids=ids, **kwargs
        )

    def get_by_id(self, ids: List[str]) -> List[Document]:
        """根据 ID 获取文档"""
        return self.vectorstore.get(ids=ids)

    def delete_by_id(self, ids: List[str]) -> None:
        """根据 ID 删除文档"""
        self.vectorstore.delete(ids=ids)

    def delete_collection(self) -> None:
        """删除整个集合"""
        self.vectorstore.delete_collection()
        self._vectorstore = None

    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        return {
            "name": self.vectorstore._collection.name,
            "count": self.vectorstore._collection.count(),
            "persist_directory": self.persist_directory,
        }

    def clear(self) -> None:
        """清空向量库"""
        self.vectorstore.delete_collection()
        self.vectorstore.reset_collection()
        self._vectorstore = None

    def save_to_disk(self) -> None:
        """持久化到磁盘（Chroma 自动持久化，但可手动触发）"""
        if self._vectorstore is not None:
            self._vectorstore.persist()


class PersistentVectorStore(VectorStoreManager):
    """持久化向量库，自动保存"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def upsert_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """upsert：存在则更新，不存在则添加"""
        self.add_documents(documents=documents, ids=ids)
        self.save_to_disk()
        return ids or []


def get_vectorstore(
    embeddings: Optional[Embeddings] = None,
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Chroma:
    """便捷函数：获取 Chroma 向量库实例"""
    return Chroma(
        persist_directory=persist_directory or config.chroma.persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name or config.chroma.collection_name,
    )
