import os
import json
import time
import torch
from src.config import (
    SAMPLE_SIZE,
    RESULTS_DIR,
    RAG_ANSWERS_PATH,
    LONG_CONTEXT_ANSWERS_PATH,
    EVALUATION_SCORES_PATH,
    POSITION_SENSITIVITY_PATH,
    POSITION_SENSITIVITY_CHART_PATH,
    PERFORMANCE_METRICS_PATH
)
from src.data_loader import load_narrativeqa_sample
from src.models import LLMInterface
from src.rag_pipeline import RAGPipeline
from src.long_context_pipeline import LongContextPipeline
from src.evaluator import BenchmarkEvaluator
from src.ablation import run_chunk_size_ablation

def main():
    print("=" * 70)
    print("STARTING RAG VS. LONG-CONTEXT PROMPTING BENCHMARK PIPELINE")
    print("=" * 70)

    # Step 1: Load Dataset (first 200 items from test set)
    dataset_items = load_narrativeqa_sample(sample_size=SAMPLE_SIZE, split="test")

    # Step 2: Initialize Shared LLM Interface
    llm = LLMInterface()

    # Step 3: Run RAG Pipeline
    print("\n--- Method 1: Running RAG Pipeline ---")
    rag_pipeline = RAGPipeline(chunk_size=512, chunk_overlap=50, top_k=5, llm=llm)
    rag_results, rag_perf = rag_pipeline.run_benchmark(dataset_items, save_path=RAG_ANSWERS_PATH)

    # Step 4: Run Long-Context Pipeline
    print("\n--- Method 2: Running Long-Context Pipeline ---")
    lc_pipeline = LongContextPipeline(max_doc_tokens=6000, llm=llm)
    lc_results, lc_perf = lc_pipeline.run_benchmark(dataset_items, save_path=LONG_CONTEXT_ANSWERS_PATH)

    # Step 5: Save Performance Metrics JSON
    performance_metrics = {
        "rag": rag_perf,
        "long_context": lc_perf
    }
    with open(PERFORMANCE_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(performance_metrics, f, indent=2)
    print(f"Saved performance metrics to {PERFORMANCE_METRICS_PATH}")

    # Step 6: Evaluate Model Predictions & Compute Scores
    print("\n--- Step 6: Computing Evaluation Metrics ---")
    evaluator = BenchmarkEvaluator()
    eval_scores, rag_bert_list, lc_bert_list = evaluator.evaluate_all(rag_results, lc_results)

    # Step 7: Position Sensitivity Analysis ('Lost in the Middle')
    print("\n--- Step 7: Position Sensitivity Analysis ---")
    position_analysis = evaluator.analyze_position_sensitivity(dataset_items, rag_bert_list, lc_bert_list)

    # Step 8: Chunk Size Ablation Study
    print("\n--- Step 8: Chunk-Size Ablation Study ---")
    ablation_results = run_chunk_size_ablation(dataset_items, chunk_sizes=[256, 512, 1024], llm=llm)

    # Step 9: Final Benchmark Summary
    print("\n" + "=" * 70)
    print("BENCHMARK EXECUTION SUMMARY")
    print("=" * 70)
    print("Evaluation Scores:")
    print(json.dumps(eval_scores, indent=2))
    print("\nPosition Sensitivity Analysis:")
    print(json.dumps(position_analysis, indent=2))
    print("\nPerformance Metrics:")
    print(json.dumps(performance_metrics, indent=2))
    print("\nChunk Size Ablation Results:")
    print(json.dumps(ablation_results, indent=2))
    print("=" * 70)
    print("ALL CONTRACT DELIVERABLES GENERATED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
