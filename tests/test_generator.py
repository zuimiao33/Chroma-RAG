"""测试生成模块"""

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.generator import (
    RAGGenerator,
    SimpleRAGGenerator,
    build_context_from_docs,
    DEFAULT_USER_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
)


@pytest.fixture
def sample_docs():
    return [
        Document(page_content="人工智能是计算机科学的一个分支。", metadata={"source": "ai.txt"}),
        Document(page_content="机器学习是人工智能的子领域。", metadata={"source": "ml.txt"}),
    ]


@pytest.fixture
def mock_llm():
    mock = MagicMock()
    response = MagicMock()
    response.content = "这是 LLM 生成的回答。"
    mock.invoke.return_value = response
    return mock


class TestRAGGenerator:
    def test_init_with_defaults(self):
        with patch("src.generator.ChatOpenAI") as mock_chat:
            mock_instance = MagicMock()
            mock_chat.return_value = mock_instance

            gen = RAGGenerator()
            assert gen.temperature == 0.7
            assert gen.max_tokens == 2048
            mock_chat.assert_called_once()

    def test_init_with_custom_params(self):
        with patch("src.generator.ChatOpenAI") as mock_chat:
            mock_instance = MagicMock()
            mock_chat.return_value = mock_instance

            gen = RAGGenerator(temperature=0.5, max_tokens=1024, model_name="deepseek-chat")
            assert gen.temperature == 0.5
            assert gen.max_tokens == 1024
            assert gen.model_name == "deepseek-chat"

    def test_chain_is_built(self):
        with patch("src.generator.ChatOpenAI") as mock_chat:
            mock_instance = MagicMock()
            mock_chat.return_value = mock_instance

            gen = RAGGenerator()
            assert gen.chain is not None

    def test_generate_calls_chain(self, sample_docs):
        gen = RAGGenerator()
        with patch.object(gen, "generate", return_value="生成的回答"):
            result = gen.generate("什么是人工智能？", "人工智能是研究。")
            assert result == "生成的回答"

    def test_generate_with_docs(self, sample_docs):
        gen = RAGGenerator()
        with patch.object(gen, "generate_with_docs", return_value="回答"):
            result = gen.generate_with_docs("问题", sample_docs)
            assert result == "回答"

    def test_custom_prompt_templates(self):
        with patch("src.generator.ChatOpenAI") as mock_chat:
            mock_instance = MagicMock()
            mock_chat.return_value = mock_instance

            custom_system = "你是一个助手。"
            custom_user = "上下文: {context}\n问题: {question}"
            gen = RAGGenerator(system_prompt=custom_system, user_prompt=custom_user)
            assert gen.system_prompt == custom_system
            assert gen.user_prompt == custom_user


class TestSimpleRAGGenerator:
    def test_answer_calls_llm(self, sample_docs, mock_llm):
        with patch("src.generator.ChatOpenAI") as mock_chat:
            mock_chat.return_value = mock_llm
            gen = SimpleRAGGenerator(llm=mock_llm)
            result = gen.answer("问题", sample_docs)
            assert "回答" in result


class TestPromptConstants:
    def test_default_system_prompt_exists(self):
        assert DEFAULT_SYSTEM_PROMPT is not None
        assert len(DEFAULT_SYSTEM_PROMPT) > 0
        assert "参考文档" in DEFAULT_SYSTEM_PROMPT

    def test_default_user_prompt_has_placeholders(self):
        assert "{context}" in DEFAULT_USER_PROMPT
        assert "{question}" in DEFAULT_USER_PROMPT


class TestBuildContextFromDocs:
    def test_context_construction(self, sample_docs):
        from src.retriever import build_context_from_docs
        context = build_context_from_docs(sample_docs)
        assert isinstance(context, str)
        assert len(context) > 0
