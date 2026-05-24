"""测试检索模块"""

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.retriever import (
    Retriever,
    MMRRetriever,
    create_retriever,
    build_context_from_docs,
)


@pytest.fixture
def mock_vectorstore():
    vs = MagicMock()
    vs.similarity_search.return_value = [
        Document(page_content="文档内容1", metadata={"source_file": "doc1.txt"}),
        Document(page_content="文档内容2", metadata={"source_file": "doc2.txt"}),
    ]
    vs.similarity_search_with_score.return_value = [
        (Document(page_content="相关文档", metadata={"source_file": "doc1.txt"}), 0.1),
        (Document(page_content="有点相关", metadata={"source_file": "doc2.txt"}), 0.4),
    ]
    vs.max_marginal_relevance_search.return_value = [
        Document(page_content="MMR结果1", metadata={"source": "mmr1.txt"}),
    ]
    return vs


@pytest.fixture
def sample_docs():
    return [
        Document(page_content="这是第一个文档的完整内容。", metadata={"source_file": "doc1.txt"}),
        Document(page_content="这是第二个文档的完整内容。", metadata={"source_file": "doc2.txt"}),
        Document(page_content="这是第三个文档的完整内容。", metadata={"source_file": "doc3.txt"}),
    ]


class TestRetriever:
    def test_similarity_search(self, mock_vectorstore):
        retriever = Retriever(mock_vectorstore, top_k=2)
        results = retriever.similarity_search("测试查询")
        mock_vectorstore.similarity_search.assert_called_once()
        assert len(results) == 2

    def test_similarity_search_with_score(self, mock_vectorstore):
        retriever = Retriever(mock_vectorstore)
        results = retriever.similarity_search_with_score("测试查询")
        assert len(results) == 2
        assert all(isinstance(r, tuple) and isinstance(r[0], Document) for r in results)

    def test_similarity_search_custom_top_k(self, mock_vectorstore):
        retriever = Retriever(mock_vectorstore, top_k=1)
        retriever.similarity_search("query")
        call_args = mock_vectorstore.similarity_search.call_args
        assert call_args[1]["k"] == 1

    def test_get_relevant_documents_with_context(self, mock_vectorstore):
        retriever = Retriever(mock_vectorstore)
        results = retriever.get_relevant_documents_with_context("测试")
        assert len(results) == 2
        assert "content" in results[0]
        assert "metadata" in results[0]
        assert "source" in results[0]


class TestMMRRetriever:
    def test_mmr_search(self, mock_vectorstore):
        retriever = MMRRetriever(mock_vectorstore, top_k=1, lambda_mult=0.5)
        results = retriever.mmr_search("测试查询")
        mock_vectorstore.max_marginal_relevance_search.assert_called_once()
        assert len(results) == 1

    def test_mmr_search_params(self, mock_vectorstore):
        retriever = MMRRetriever(mock_vectorstore, top_k=3, fetch_k=10, lambda_mult=0.3)
        retriever.mmr_search("query", top_k=3, fetch_k=10, lambda_mult=0.3)
        mock_vectorstore.max_marginal_relevance_search.assert_called_once()
        call_args = mock_vectorstore.max_marginal_relevance_search.call_args
        assert call_args.kwargs["k"] == 3
        assert call_args.kwargs["fetch_k"] == 10
        assert call_args.kwargs["lambda_mult"] == 0.3

    def test_search_delegates_to_mmr(self, mock_vectorstore):
        retriever = MMRRetriever(mock_vectorstore)
        results = retriever.search("query")
        assert len(results) == 1


class TestCreateRetriever:
    def test_create_similarity_retriever(self, mock_vectorstore):
        retriever = create_retriever(mock_vectorstore, search_type="similarity", top_k=3)
        assert isinstance(retriever, Retriever)
        assert retriever.top_k == 3

    def test_create_mmr_retriever(self, mock_vectorstore):
        retriever = create_retriever(mock_vectorstore, search_type="mmr", top_k=2)
        assert isinstance(retriever, MMRRetriever)


class TestBuildContextFromDocs:
    def test_build_context_from_docs(self, sample_docs):
        context = build_context_from_docs(sample_docs)
        assert isinstance(context, str)
        assert "doc1.txt" in context
        assert "doc2.txt" in context

    def test_build_context_respects_max_chars(self, sample_docs):
        context = build_context_from_docs(sample_docs, max_chars=50)
        assert len(context) <= 50 + 100

    def test_build_context_empty_list(self):
        context = build_context_from_docs([])
        assert context == ""

    def test_build_context_includes_source(self, sample_docs):
        context = build_context_from_docs(sample_docs)
        assert "来源:" in context
