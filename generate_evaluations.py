import os
import json
from src.data_loader import load_narrativeqa_sample
from src.evaluator import BenchmarkEvaluator
from src.config import (
    RAG_ANSWERS_PATH,
    LONG_CONTEXT_ANSWERS_PATH,
    EVALUATION_SCORES_PATH,
    POSITION_SENSITIVITY_PATH,
    POSITION_SENSITIVITY_CHART_PATH,
    PERFORMANCE_METRICS_PATH
)

def run_evaluation():
    print("Loading predictions from results directory...")
    with open(RAG_ANSWERS_PATH, "r", encoding="utf-8") as f:
        rag_results = json.load(f)

    with open(LONG_CONTEXT_ANSWERS_PATH, "r", encoding="utf-8") as f:
        lc_results = json.load(f)

    print(f"Loaded {len(rag_results)} RAG items and {len(lc_results)} Long-Context items.")

    dataset_items = load_narrativeqa_sample(sample_size=200)

    evaluator = BenchmarkEvaluator()

    # Step 1: Compute Evaluation Scores
    eval_scores, rag_bert_list, lc_bert_list = evaluator.evaluate_all(rag_results, lc_results)

    # Step 2: Compute Position Sensitivity Analysis & Chart
    position_analysis = evaluator.analyze_position_sensitivity(dataset_items, rag_bert_list, lc_bert_list)

    print("\n--- Evaluation Results ---")
    print("Evaluation Scores:", json.dumps(eval_scores, indent=2))
    print("\nPosition Sensitivity Analysis:", json.dumps(position_analysis, indent=2))

    # Verify chart file
    if os.path.exists(POSITION_SENSITIVITY_CHART_PATH):
        print(f"\nPosition sensitivity chart created successfully at: {POSITION_SENSITIVITY_CHART_PATH}")
    else:
        print(f"\nERROR: Position sensitivity chart missing at {POSITION_SENSITIVITY_CHART_PATH}")

if __name__ == "__main__":
    run_evaluation()
