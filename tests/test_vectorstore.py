"""测试向量存储模块"""

import pytest
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from src.vectorstore import VectorStoreManager, PersistentVectorStore, get_vectorstore


@pytest.fixture
def temp_vector_dir():
    with tempfile.TemporaryDirectory() as td:
        yield td


@pytest.fixture
def mock_embeddings():
    emb = MagicMock()
    emb.embed_query.return_value = [0.1] * 384
    emb.embed_documents.return_value = [[0.1] * 384] * 10
    emb.get_embedding_dimension.return_value = 384
    return emb


@pytest.fixture
def sample_documents():
    return [
        Document(page_content=f"这是第 {i} 个文档的内容。", metadata={"id": i})
        for i in range(5)
    ]


class TestVectorStoreManager:
    def test_init_with_params(self, temp_vector_dir, mock_embeddings):
        manager = VectorStoreManager(
            persist_directory=temp_vector_dir,
            collection_name="test_collection",
            embeddings=mock_embeddings,
        )
        assert manager.persist_directory == temp_vector_dir
        assert manager.collection_name == "test_collection"
        assert manager._embeddings == mock_embeddings

    def test_vectorstore_property_creates_instance(self, temp_vector_dir, mock_embeddings):
        with patch("src.vectorstore.Chroma") as mock_chroma:
            mock_vs = MagicMock()
            mock_vs._collection = MagicMock()
            mock_vs._collection.count.return_value = 0
            mock_chroma.return_value = mock_vs

            manager = VectorStoreManager(
                persist_directory=temp_vector_dir,
                embeddings=mock_embeddings,
            )
            _ = manager.vectorstore
            mock_chroma.assert_called_once()

    def test_add_documents_calls_vectorstore(self, temp_vector_dir, mock_embeddings, sample_documents):
        with patch("src.vectorstore.Chroma") as mock_chroma_cls:
            mock_vs = MagicMock()
            mock_vs._collection = MagicMock()
            mock_vs._collection.count.return_value = 0
            mock_vs.add_documents.return_value = ["id1", "id2", "id3", "id4", "id5"]
            mock_chroma_cls.return_value = mock_vs

            manager = VectorStoreManager(
                persist_directory=temp_vector_dir,
                embeddings=mock_embeddings,
            )
            ids = manager.add_documents(sample_documents)
            assert len(ids) == 5
            mock_vs.add_documents.assert_called_once()

    def test_delete_by_id(self, temp_vector_dir, mock_embeddings):
        with patch("src.vectorstore.Chroma") as mock_chroma_cls:
            mock_vs = MagicMock()
            mock_vs._collection = MagicMock()
            mock_vs._collection.count.return_value = 0
            mock_chroma_cls.return_value = mock_vs

            manager = VectorStoreManager(
                persist_directory=temp_vector_dir,
                embeddings=mock_embeddings,
            )
            manager.delete_by_id(["id1", "id2"])
            mock_vs.delete.assert_called_once()

    def test_get_collection_info(self, temp_vector_dir, mock_embeddings):
        with patch("src.vectorstore.Chroma") as mock_chroma_cls:
            mock_vs = MagicMock()
            mock_vs._collection = MagicMock()
            mock_vs._collection.name = "test"
            mock_vs._collection.count.return_value = 42
            mock_chroma_cls.return_value = mock_vs

            manager = VectorStoreManager(
                persist_directory=temp_vector_dir,
                embeddings=mock_embeddings,
            )
            info = manager.get_collection_info()
            assert info["name"] == "test"
            assert info["count"] == 42


class TestPersistentVectorStore:
    def test_upsert_documents(self, temp_vector_dir, mock_embeddings, sample_documents):
        with patch("src.vectorstore.Chroma") as mock_chroma_cls:
            mock_vs = MagicMock()
            mock_vs._collection = MagicMock()
            mock_vs._collection.count.return_value = 0
            mock_vs.add_documents.return_value = ["id1"]
            mock_chroma_cls.return_value = mock_vs

            store = PersistentVectorStore(
                persist_directory=temp_vector_dir,
                embeddings=mock_embeddings,
            )
            store.upsert_documents(sample_documents)
            mock_vs.persist.assert_called()


class TestGetVectorstore:
    def test_returns_chroma_instance(self, mock_embeddings):
        with patch("src.vectorstore.Chroma") as mock_chroma_cls:
            mock_vs = MagicMock()
            mock_vs._collection = MagicMock()
            mock_vs._collection.count.return_value = 0
            mock_chroma_cls.return_value = mock_vs

            vs = get_vectorstore(embeddings=mock_embeddings)
            mock_chroma_cls.assert_called_once()
