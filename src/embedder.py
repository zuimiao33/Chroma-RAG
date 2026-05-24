"""嵌入模型模块，支持本地 sentence-transformers 和 OpenAI 嵌入"""

from typing import List, Optional, Union
import numpy as np

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

from src.config import config


class LocalEmbeddings(Embeddings):
    """本地 sentence-transformers 嵌入封装"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name or config.embedding.model_name
        self.device = device or config.embedding.device
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            print(f"  加载嵌入模型: {self.model_name} (device={self.device})")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=False
        )
        return [emb.tolist() for emb in embeddings]

    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


class EmbeddingModel:
    """嵌入模型管理器，支持本地和远程切换"""

    def __init__(self, use_local: bool = True):
        self.use_local = use_local
        self._embeddings: Optional[Embeddings] = None

    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            if self.use_local:
                self._embeddings = LocalEmbeddings()
            else:
                self._embeddings = OpenAIEmbeddings(
                    model="text-embedding-ada-002",
                    api_key=config.llm.api_key,
                )
        return self._embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def get_dimension(self) -> int:
        return self.embeddings.get_embedding_dimension()


def get_embedder(use_local: bool = True) -> EmbeddingModel:
    """便捷函数：获取嵌入模型实例"""
    return EmbeddingModel(use_local=use_local)
