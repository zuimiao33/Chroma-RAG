"""生成模块，对接 DeepSeek-V4 Pro LLM 实现 RAG 问答"""

from typing import Optional, List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import config
from src.retriever import build_context_from_docs


DEFAULT_SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请根据以下参考文档内容，准确、简洁地回答用户的问题。

要求：
1. 如果参考文档中有相关信息，请基于文档内容回答
2. 如果文档中没有相关内容，请如实说明"根据提供的文档，我无法回答这个问题"
3. 回答要条理清晰，用中文回复
4. 如果需要，可以引用文档中的具体内容

"""

DEFAULT_USER_PROMPT = """参考文档：
{context}

用户问题：{question}

请基于上述参考文档回答问题："""


class RAGGenerator:
    """RAG 生成器，对接 DeepSeek-V4 Pro"""

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt: str = DEFAULT_USER_PROMPT,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
    ):
        self.temperature = temperature or config.llm.temperature
        self.max_tokens = max_tokens or config.llm.max_tokens
        self.model_name = model_name or config.llm.model_name

        self._llm = ChatOpenAI(
            model=self.model_name,
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self._build_chain()

    def _build_chain(self):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", self.user_prompt),
        ])

        def format_inputs(question: str, context: str) -> dict:
            return {"question": question, "context": context}

        self.chain = (
            RunnablePassthrough()
            | (lambda inputs: format_inputs(**inputs))
            | self.prompt
            | self._llm
            | StrOutputParser()
        )

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        """基于上下文和问题生成回答"""
        return self.chain.invoke({"question": question, "context": context})

    def generate_with_docs(
        self,
        question: str,
        documents: List[Document],
        max_context_chars: int = 3000,
    ) -> str:
        """一步完成：传入文档列表，自动构建上下文并生成"""
        context = build_context_from_docs(documents, max_chars=max_context_chars)
        return self.generate(question, context)

    def batch_generate(
        self,
        questions: List[str],
        contexts: List[str],
    ) -> List[str]:
        """批量生成（并发）"""
        return [
            self.generate(q, c)
            for q, c in zip(questions, contexts)
        ]


class SimpleRAGGenerator:
    """简化版生成器，仅保留核心生成逻辑"""

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        system_prompt: Optional[str] = None,
    ):
        self._llm = llm or ChatOpenAI(
            model=config.llm.model_name,
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.user_template = DEFAULT_USER_PROMPT

    def answer(
        self,
        question: str,
        retrieved_docs: List[Document],
    ) -> str:
        context = build_context_from_docs(retrieved_docs)
        prompt = self.system_prompt + self.user_template.format(
            context=context, question=question
        )
        return self._llm.invoke(prompt).content


def get_generator(
    system_prompt: Optional[str] = None,
    **kwargs,
) -> RAGGenerator:
    """便捷函数：获取生成器实例"""
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    return RAGGenerator(**kwargs)
