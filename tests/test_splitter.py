"""测试文本分割模块"""

import pytest

from langchain_core.documents import Document

from src.splitter import ChunkSplitter, MarkdownSplitter, split_and_process


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="这是一个很长的段落。" * 50,
            metadata={"source": "test1.txt"},
        ),
        Document(
            page_content="另一个文档段落内容。" * 30,
            metadata={"source": "test2.txt"},
        ),
    ]


@pytest.fixture
def short_docs():
    return [
        Document(page_content="短", metadata={"source": "s1"}),
        Document(page_content="更短", metadata={"source": "s2"}),
        Document(page_content="非常短的文本", metadata={"source": "s3"}),
    ]


class TestChunkSplitter:
    def test_init_default_params(self):
        splitter = ChunkSplitter()
        assert splitter.chunk_size == 800
        assert splitter.chunk_overlap == 100

    def test_init_custom_params(self):
        splitter = ChunkSplitter(chunk_size=500, chunk_overlap=50)
        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 50

    def test_split_documents_returns_list(self, sample_documents):
        splitter = ChunkSplitter(chunk_size=200, chunk_overlap=30)
        chunks = splitter.split_documents(sample_documents)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Document) for c in chunks)

    def test_split_increases_count(self, sample_documents):
        splitter = ChunkSplitter(chunk_size=200, chunk_overlap=30)
        chunks = splitter.split_documents(sample_documents)
        assert len(chunks) >= len(sample_documents)

    def test_split_text(self):
        splitter = ChunkSplitter(chunk_size=50, chunk_overlap=10)
        text = "这是第一句。" * 10 + "这是第二句。" * 10
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2
        assert all(isinstance(c, str) for c in chunks)

    def test_chunk_size_respected(self):
        splitter = ChunkSplitter(chunk_size=100, chunk_overlap=20)
        doc = Document(page_content="a" * 500, metadata={"source": "test"})
        chunks = splitter.split_documents([doc])
        for chunk in chunks:
            assert len(chunk.page_content) <= 100 + 20

    def test_overlap_mechanism(self):
        splitter = ChunkSplitter(chunk_size=50, chunk_overlap=20)
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 3
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2
        if len(chunks) >= 2:
            assert chunks[0][-20:] == chunks[1][:20]

    def test_merge_short_chunks(self, short_docs):
        splitter = ChunkSplitter(chunk_size=200, chunk_overlap=20)
        chunks = splitter.split_documents(short_docs)
        merged = splitter.merge_short_chunks(chunks, min_chunk_size=20)
        total_chars = sum(len(c.page_content) for c in merged)
        assert total_chars == len("短") + len("更短") + len("非常短的文本")

    def test_get_chunk_stats(self, sample_documents):
        splitter = ChunkSplitter(chunk_size=200, chunk_overlap=30)
        chunks = splitter.split_documents(sample_documents)
        stats = splitter.get_chunk_stats(chunks)
        assert stats["count"] == len(chunks)
        assert stats["avg_length"] > 0
        assert stats["min_length"] <= stats["max_length"]

    def test_get_chunk_stats_empty(self):
        splitter = ChunkSplitter()
        stats = splitter.get_chunk_stats([])
        assert stats["count"] == 0
        assert stats["avg_length"] == 0


class TestMarkdownSplitter:
    def test_split_markdown(self):
        md_text = "# 大标题\n\n段落一内容。" * 10
        splitter = MarkdownSplitter(chunk_size=100, chunk_overlap=20)
        docs = splitter.split_markdown(md_text, metadata={"doc_type": "test"})
        assert len(docs) >= 1
        assert docs[0].metadata["doc_type"] == "test"


class TestSplitAndProcess:
    def test_returns_document_list(self, sample_documents):
        result = split_and_process(sample_documents, merge_short=True)
        assert isinstance(result, list)
        assert all(isinstance(d, Document) for d in result)

    def test_merge_short_reduces_count(self, short_docs):
        result_no_merge = split_and_process(short_docs, merge_short=False)
        result_merge = split_and_process(short_docs, merge_short=True)
        assert len(result_merge) <= len(result_no_merge)
