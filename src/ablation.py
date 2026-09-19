import os
import json
from typing import List, Dict, Any
from src.rag_pipeline import RAGPipeline

def run_chunk_size_ablation(dataset_items: List[Dict[str, Any]], chunk_sizes: List[int] = [256, 512, 1024], llm=None) -> Dict[str, Any]:
    """
    Executes a chunk-size ablation study comparing chunk sizes 256, 512, and 1024 on a subset of data.
    Returns dictionary with metrics per chunk size.
    """
    print("Running Chunk-Size Ablation Study (sizes: 256, 512, 1024)...")
    ablation_sample = dataset_items[:20]  # Representative sample for speed
    results = {}

    for cs in chunk_sizes:
        print(f"Testing chunk size: {cs}...")
        pipeline = RAGPipeline(chunk_size=cs, chunk_overlap=int(cs * 0.1), top_k=5, llm=llm)
        pipeline_results, perf_stats = pipeline.run_benchmark(ablation_sample, save_path=os.devnull)

        # Quick evaluation
        from src.evaluator import BenchmarkEvaluator
        evaluator = BenchmarkEvaluator()
        preds = [r["generated_answer"] for r in pipeline_results]
        refs = [r["reference_answer"] for r in pipeline_results]
        r1, rl = evaluator.calculate_rouge_scores(preds, refs)
        bert_avg, _ = evaluator.calculate_bert_scores(preds, refs)

        results[str(cs)] = {
            "chunk_size": cs,
            "rouge_1": round(r1, 4),
            "rouge_l": round(rl, 4),
            "bert_score_f1": round(bert_avg, 4),
            "median_latency_s": perf_stats["median_query_latency_s"],
            "tokens_per_s": perf_stats["generation_tokens_per_s"]
        }

    print("Chunk-size ablation study completed.")
    return results
