"""RAG 知识库命令行入口"""

import sys
import argparse
from pathlib import Path

from src.config import config
from src.rag_pipeline import RAGPipeline, load_existing_pipeline
from src.eval_pipeline import RAGEvaluator, load_test_data


def cmd_ingest(args):
    """入库命令"""
    source = Path(args.source)
    if not source.exists():
        print(f"错误：路径不存在 {source}")
        return

    pipeline = RAGPipeline()
    result = pipeline.ingest(
        source=source,
        recursive=args.recursive,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"\n入库完成！共处理 {result['original_docs']} 份文档，"
          f"生成 {result['chunks_count']} 个 chunks")


def cmd_query(args):
    """问答命令"""
    pipeline = load_existing_pipeline()

    if args.interactive:
        print("\nRAG 知识库问答系统 (输入 'quit' 或 'exit' 退出)")
        print(f"当前知识库: {pipeline.get_stats()}")
        print("-" * 50)

        history = []
        while True:
            try:
                question = input("\n你: ").strip()
                if question.lower() in ("quit", "exit", "q"):
                    print("再见!")
                    break
                if not question:
                    continue

                if args.show_context:
                    result = pipeline.query(question, return_context=True)
                    print(f"\n回答: {result['answer']}")
                    print(f"\n参考文档 ({result['doc_count']} 篇):")
                    print(result["sources"])
                else:
                    answer = pipeline.query(question)
                    print(f"\n回答: {answer}")

            except KeyboardInterrupt:
                print("\n\n再见!")
                break
    else:
        answer = pipeline.query(args.question, return_context=args.show_context)
        if args.show_context:
            print(f"回答: {answer['answer']}")
            print(f"\n参考文档 ({answer['doc_count']} 篇):")
            print(answer["sources"])
        else:
            print(answer)


def cmd_stats(args):
    """统计命令"""
    pipeline = load_existing_pipeline()
    stats = pipeline.get_stats()
    if "error" in stats:
        print(f"错误：{stats['error']}")
        return
    print("\n知识库统计信息：")
    print("-" * 40)
    for key, value in stats.items():
        print(f"  {key}: {value}")


def cmd_clear(args):
    """清空知识库"""
    pipeline = load_existing_pipeline()
    pipeline.clear_knowledge_base()
    print("知识库已清空。")


def cmd_eval(args):
    """评估命令"""
    import json
    pipeline = load_existing_pipeline()
    evaluator = RAGEvaluator(pipeline)

    test_data_path = args.test_data or "eval_data.json"
    print(f"\n加载测试数据: {test_data_path}")
    test_data = load_test_data(test_data_path)
    print(f"共 {len(test_data)} 条测试用例")
    print(f"检索 top_k={args.top_k}, LLM评估={'开启' if args.use_llm else '关闭'}")
    print()

    results = evaluator.evaluate_batch(
        test_data,
        top_k=args.top_k,
        use_llm_eval=args.use_llm,
    )

    summary = results["summary"]
    print("=" * 60)
    print("评估结果汇总")
    print("=" * 60)
    print(f"测试用例数: {summary['total_questions']}")
    print(f"平均延迟:   {summary['avg_latency_ms']} ms/次")
    print()

    if summary.get("retrieval"):
        print("--- 检索指标 ---")
        print(f"  平均精确率(Precision): {summary['retrieval']['avg_precision']:.4f}")
        print(f"  平均召回率(Recall):    {summary['retrieval']['avg_recall']:.4f}")
        print(f"  平均F1分数:           {summary['retrieval']['avg_f1']:.4f}")
        print(f"  评估样本数:           {summary['retrieval']['evaluated_count']}")
        print()

    if summary.get("generation"):
        print("--- 生成质量（LLM评估）---")
        print(f"  平均得分: {summary['generation']['avg_llm_score']:.2f}/5.0")
        print(f"  最高得分: {summary['generation']['max_score']:.2f}")
        print(f"  最低得分: {summary['generation']['min_score']:.2f}")
        print()

    if args.detailed:
        print("--- 详细结果 ---")
        print("=" * 60)
        for detail in results["details"]:
            q = detail["question"]
            print(f"\n问题: {q}")
            if "retrieval_metrics" in detail["retrieval"]:
                m = detail["retrieval"]["retrieval_metrics"]
                print(f"  检索 - P: {m['precision']:.2f}  R: {m['recall']:.2f}  F1: {m['f1']:.2f}")
            ans = detail["generation"]["answer"]
            print(f"  回答: {ans[:200]}{'...' if len(ans) > 200 else ''}")
            if detail["generation"].get("llm_evaluation"):
                ev = detail["generation"]["llm_evaluation"]
                print(f"  LLM评估: 忠实度={ev['faithfulness']['score']} 相关性={ev['answer_relevance']['score']} 上下文={ev['context_relevance']['score']}")

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RAG 全流程知识库")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    p_ingest = subparsers.add_parser("ingest", help="文档入库")
    p_ingest.add_argument("source", help="文档路径或目录")
    p_ingest.add_argument("-r", "--recursive", action="store_true", help="递归扫描子目录")
    p_ingest.add_argument("--chunk-size", type=int, default=None, help="chunk 大小")
    p_ingest.add_argument("--chunk-overlap", type=int, default=None, help="chunk 重叠大小")

    p_query = subparsers.add_parser("query", help="知识问答")
    p_query.add_argument("-q", "--question", type=str, help="问题内容")
    p_query.add_argument("-i", "--interactive", action="store_true", help="交互式问答")
    p_query.add_argument("-c", "--show-context", action="store_true", help="显示参考文档")

    subparsers.add_parser("stats", help="查看知识库统计")

    p_clear = subparsers.add_parser("clear", help="清空知识库")

    p_eval = subparsers.add_parser("eval", help="RAG 评估")
    p_eval.add_argument("-d", "--test-data", type=str, default=None, help="测试数据文件路径")
    p_eval.add_argument("-k", "--top-k", type=int, default=5, help="检索返回的文档数量")
    p_eval.add_argument("--use-llm", action="store_true", help="开启 LLM 生成质量评估")
    p_eval.add_argument("--detailed", action="store_true", help="显示每条测试用例的详细结果")
    p_eval.add_argument("-o", "--output", type=str, default=None, help="结果输出文件路径(JSON)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print("\n\n快速开始:")
        print("  1. 将文档放入 data/ 目录")
        print("  2. 运行: python main.py ingest data/")
        print("  3. 运行: python main.py query -i")
        print("  4. 运行: python main.py eval            # 快速评估（仅检索指标）")
        print("  5. 运行: python main.py eval --use-llm  # 完整评估（含生成质量）")
        return

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "clear":
        cmd_clear(args)
    elif args.command == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
