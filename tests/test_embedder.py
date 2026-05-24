"""测试嵌入模型模块"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.embedder import LocalEmbeddings, EmbeddingModel, get_embedder


class TestLocalEmbeddings:
    def test_embed_query_returns_list(self):
        with patch("src.embedder.SentenceTransformer") as mock_model:
            mock_instance = MagicMock()
            mock_instance.encode.return_value = np.array([0.1] * 384)
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_model.return_value = mock_instance

            emb = LocalEmbeddings(model_name="all-MiniLM-L6-v2", device="cpu")
            result = emb.embed_query("这是一个测试")
            assert isinstance(result, list)
            assert len(result) == 384

    def test_embed_documents_returns_list_of_lists(self):
        with patch("src.embedder.SentenceTransformer") as mock_model:
            mock_instance = MagicMock()
            mock_instance.encode.return_value = np.array([[0.1] * 384] * 3)
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_model.return_value = mock_instance

            emb = LocalEmbeddings()
            texts = ["text1", "text2", "text3"]
            result = emb.embed_documents(texts)
            assert isinstance(result, list)
            assert len(result) == 3
            assert all(isinstance(r, list) for r in result)
            assert all(len(r) == 384 for r in result)

    def test_embeddings_are_normalized(self):
        with patch("src.embedder.SentenceTransformer") as mock_model:
            mock_instance = MagicMock()
            vec = np.array([0.5, 0.5, 0.5, 0.5])
            mock_instance.encode.return_value = vec
            mock_instance.get_sentence_embedding_dimension.return_value = 4
            mock_model.return_value = mock_instance

            emb = LocalEmbeddings()
            result = emb.embed_query("test")
            norm = np.linalg.norm(result)
            assert abs(norm - 1.0) < 0.01

    def test_same_text_same_embedding(self):
        with patch("src.embedder.SentenceTransformer") as mock_model:
            mock_instance = MagicMock()
            mock_instance.encode.return_value = np.array([0.1] * 384)
            mock_model.return_value = mock_instance

            emb = LocalEmbeddings()
            r1 = emb.embed_query("hello world")
            r2 = emb.embed_query("hello world")
            assert r1 == r2

    def test_get_embedding_dimension(self):
        with patch("src.embedder.SentenceTransformer") as mock_model:
            mock_instance = MagicMock()
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_model.return_value = mock_instance

            emb = LocalEmbeddings()
            assert emb.get_embedding_dimension() == 384


class TestEmbeddingModel:
    def test_embedder_lazy_loading(self):
        with patch("src.embedder.SentenceTransformer") as mock_model:
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance

            em = EmbeddingModel(use_local=True)
            assert em._embeddings is None
            _ = em.embeddings
            assert em._embeddings is not None

    def test_embed_query_delegates(self):
        with patch("src.embedder.SentenceTransformer") as mock_model:
            mock_instance = MagicMock()
            mock_instance.encode.return_value = np.array([0.1] * 384)
            mock_model.return_value = mock_instance

            em = EmbeddingModel(use_local=True)
            result = em.embed_query("test")
            assert isinstance(result, list)
            assert len(result) == 384

    def test_get_embedder_function(self):
        embedder = get_embedder(use_local=True)
        assert isinstance(embedder, EmbeddingModel)
