"""RAG 评估体系模块，支持检索指标和生成质量评估"""

import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

from langchain_core.documents import Document

from src.config import config
from src.retriever import build_context_from_docs


class RAGEvaluator:
    """RAG 系统评估器"""

    def __init__(self, rag_pipeline, llm=None):
        self.pipeline = rag_pipeline
        self.llm = llm

    # ========== 检索指标 ==========

    def compute_retrieval_metrics(
        self,
        retrieved_docs: List[Document],
        relevant_sources: List[str],
    ) -> Dict[str, float]:
        """
        计算检索召回率和精确率

        参数:
            retrieved_docs: 检索返回的文档列表
            relevant_sources: 预先标注的相关文档来源列表（Ground Truth，如 L1.CF总结.pdf）
        """
        retrieved_sources = {doc.metadata.get("source_file") for doc in retrieved_docs}
        relevant_sources = set(relevant_sources)

        tp = len(retrieved_sources & relevant_sources)
        fp = len(retrieved_sources - relevant_sources)
        fn = len(relevant_sources - retrieved_sources)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    def evaluate_retrieval(
        self,
        question: str,
        relevant_sources: List[str],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """评估单次检索的质量"""
        if self.pipeline.retriever is None:
            self.pipeline._init_retriever()

        docs = self.pipeline.retriever.similarity_search(question, top_k=top_k)
        metrics = self.compute_retrieval_metrics(docs, relevant_sources)

        return {
            "question": question,
            "retrieved_count": len(docs),
            "relevant_count": len(relevant_sources),
            "retrieval_metrics": metrics,
            "retrieved_docs": [
                {"content": d.page_content[:100] + "...", "source": d.metadata.get("source_file", "unknown")}
                for d in docs
            ],
        }

    # ========== 生成质量评估（LLM） ==========

    def _evaluate_faithfulness(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """评估回答是否忠实于检索文档"""
        prompt = f"""你是一个评估专家。请判断以下回答是否忠实于提供的参考文档。

参考文档：
{context}

回答：
{answer}

请评估回答的忠实度，从以下角度：
1. 回答中的陈述是否可以从参考文档推导出来？
2. 是否有回答凭空编造的内容？

请以以下 JSON 格式返回（只返回 JSON，不要其他内容）：
{{"score": 0-5 的整数分数, "reason": "简短理由"}}

评分标准：
5分：完全忠实，所有内容均可从文档推导
4分：基本忠实，极少偏差
3分：部分忠实，有一些无依据内容
2分：不太忠实，有较多无依据内容
1分：基本不忠实
0分：完全编造，与文档无关
"""
        try:
            response = self.pipeline.generator._llm.invoke(prompt)
            result = json.loads(response.content)
            return {"score": result.get("score", 0), "reason": result.get("reason", "")}
        except Exception:
            return {"score": 0, "reason": "评估失败"}

    def _evaluate_answer_relevance(self, question: str, answer: str) -> Dict[str, Any]:
        """评估回答与问题的相关性"""
        prompt = f"""你是一个评估专家。请判断以下回答与问题的相关性。

问题：{question}

回答：{answer}

请评估回答是否切题。从以下角度：
1. 回答是否针对问题进行了解答？
2. 回答是否覆盖了问题的核心？

请以以下 JSON 格式返回（只返回 JSON，不要其他内容）：
{{"score": 0-5 的整数分数, "reason": "简短理由"}}

评分标准：
5分：完全切题，精准回答问题
4分：基本切题，有轻微偏离
3分：部分切题，有些答非所问
2分：不太切题
1分：基本不切题
0分：完全跑题
"""
        try:
            response = self.pipeline.generator._llm.invoke(prompt)
            result = json.loads(response.content)
            return {"score": result.get("score", 0), "reason": result.get("reason", "")}
        except Exception:
            return {"score": 0, "reason": "评估失败"}

    def _evaluate_context_relevance(self, question: str, docs: List[Document]) -> Dict[str, Any]:
        """评估检索文档与问题的相关性"""
        context_text = "\n".join([d.page_content for d in docs])
        prompt = f"""你是一个评估专家。请判断以下检索到的文档与问题的相关性。

问题：{question}

检索到的文档：
{context_text}

请评估这些文档是否对回答该问题有帮助。

请以以下 JSON 格式返回（只返回 JSON，不要其他内容）：
{{"score": 0-5 的整数分数, "reason": "简短理由"}}

评分标准：
5分：完全相关，所有文档都直接相关
4分：大部分相关
3分：部分相关，有一些噪音
2分：大部分噪音
1分：几乎不相关
0分：完全不相关
"""
        try:
            response = self.pipeline.generator._llm.invoke(prompt)
            result = json.loads(response.content)
            return {"score": result.get("score", 0), "reason": result.get("reason", "")}
        except Exception:
            return {"score": 0, "reason": "评估失败"}

    def evaluate_generation(
        self,
        question: str,
        answer: str,
        docs: List[Document],
    ) -> Dict[str, Any]:
        """评估生成质量（需要 LLM）"""
        context = build_context_from_docs(docs)

        faithfulness = self._evaluate_faithfulness(question, answer, context)
        answer_relevance = self._evaluate_answer_relevance(question, answer)
        context_relevance = self._evaluate_context_relevance(question, docs)

        return {
            "question": question,
            "faithfulness": faithfulness,
            "answer_relevance": answer_relevance,
            "context_relevance": context_relevance,
            "avg_score": round(
                (faithfulness["score"] + answer_relevance["score"] + context_relevance["score"]) / 3, 2
            ),
        }

    # ========== 完整端到端评估 ==========

    def evaluate_single(
        self,
        question: str,
        relevant_sources: List[str],
        ground_truth: str = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """单次完整评估（检索 + 生成）"""
        retrieval_result = self.evaluate_retrieval(question, relevant_sources, top_k)
        docs = self.pipeline.retriever.similarity_search(question, top_k=top_k)
        answer = self.pipeline.generator.generate_with_docs(question, docs)

        generation_result = {}
        if self.llm:
            generation_result = self.evaluate_generation(question, answer, docs)

        return {
            "question": question,
            "retrieval": retrieval_result,
            "generation": {
                "answer": answer,
                "ground_truth": ground_truth,
                "llm_evaluation": generation_result,
            },
        }

    def evaluate_batch(
        self,
        test_data: List[Dict[str, Any]],
        top_k: int = 5,
        use_llm_eval: bool = False,
    ) -> Dict[str, Any]:
        """
        批量评估

        参数:
            test_data: 测试数据集，每个元素包含 question, relevant_doc_ids, ground_truth
            top_k: 检索的文档数量
            use_llm_eval: 是否使用 LLM 评估生成质量
        """
        results = []
        retrieval_metrics_list = []
        generation_scores = []
        total_time = 0

        for item in test_data:
            start = time.time()
            result = self.evaluate_single(
                question=item["question"],
                relevant_sources=item.get("relevant_sources", []),
                ground_truth=item.get("ground_truth"),
                top_k=top_k,
            )
            elapsed = time.time() - start
            total_time += elapsed

            results.append(result)

            if item.get("relevant_sources"):
                retrieval_metrics_list.append(result["retrieval"]["retrieval_metrics"])

            if result["generation"]["llm_evaluation"]:
                avg = result["generation"]["llm_evaluation"]["avg_score"]
                generation_scores.append(avg)

        summary = {
            "total_questions": len(test_data),
            "avg_latency_ms": round(total_time / len(test_data) * 1000, 2) if test_data else 0,
            "retrieval": {},
            "generation": {},
        }

        if retrieval_metrics_list:
            summary["retrieval"] = {
                "avg_precision": round(sum(m["precision"] for m in retrieval_metrics_list) / len(retrieval_metrics_list), 4),
                "avg_recall": round(sum(m["recall"] for m in retrieval_metrics_list) / len(retrieval_metrics_list), 4),
                "avg_f1": round(sum(m["f1"] for m in retrieval_metrics_list) / len(retrieval_metrics_list), 4),
                "evaluated_count": len(retrieval_metrics_list),
            }

        if generation_scores:
            summary["generation"] = {
                "avg_llm_score": round(sum(generation_scores) / len(generation_scores), 2),
                "max_score": max(generation_scores),
                "min_score": min(generation_scores),
                "evaluated_count": len(generation_scores),
            }

        return {"summary": summary, "details": results}


def load_test_data(path: str) -> List[Dict[str, Any]]:
    """加载测试数据集"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"测试数据文件不存在: {path}")

    if p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("test_cases", [])

    raise ValueError(f"不支持的文件格式: {p.suffix}")
