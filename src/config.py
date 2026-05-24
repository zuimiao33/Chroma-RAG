"""配置管理模块，统一管理所有配置项"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class EmbeddingConfig(BaseModel):
    """嵌入模型配置"""
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    encode_kwargs: dict = {"normalize_embeddings": True}


class ChromaConfig(BaseModel):
    """Chroma 向量库配置"""
    persist_directory: str = "./vector_store"
    collection_name: str = "rag_docs"


class LLMConfig(BaseModel):
    """大模型配置"""
    model_name: str = "deepseek-v4-pro"
    api_key: str
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.7
    max_tokens: int = 2048


class ChunkingConfig(BaseModel):
    """文本分块配置"""
    chunk_size: int = 800
    chunk_overlap: int = 100


class RetrievalConfig(BaseModel):
    """检索配置"""
    default_top_k: int = 5


class Config:
    """全局配置类，统一管理所有子配置"""

    def __init__(self):
        self.embedding = EmbeddingConfig(
            model_name=os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
            device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        )
        self.chroma = ChromaConfig(
            persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./vector_store"),
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "rag_docs"),
        )
        self.llm = LLMConfig(
            model_name=os.getenv("LLM_MODEL_NAME", "deepseek-v4-pro"),
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
        )
        self.chunking = ChunkingConfig(
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
        )
        self.retrieval = RetrievalConfig(
            default_top_k=int(os.getenv("DEFAULT_TOP_K", "5")),
        )

        self._validate()

    def _validate(self):
        if not self.llm.api_key:
            raise ValueError("DEEPSEEK_API_KEY 未设置，请检查 .env 文件")

    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def processed_dir(self) -> Path:
        return self.project_root / "processed"


config = Config()
