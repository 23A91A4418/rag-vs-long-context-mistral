import os
import json
import time
import torch
from typing import List, Dict, Any
from src.config import MAX_DOC_TOKENS, LONG_CONTEXT_ANSWERS_PATH
from src.models import LLMInterface

class LongContextPipeline:
    def __init__(self, max_doc_tokens: int = MAX_DOC_TOKENS, llm: LLMInterface = None):
        self.max_doc_tokens = max_doc_tokens
        self.llm = llm if llm is not None else LLMInterface()

    def truncate_document(self, doc_text: str) -> str:
        """
        Truncates document text to first max_doc_tokens tokens using model tokenizer.
        """
        if not doc_text:
            return ""

        tokens = self.llm.tokenizer.encode(doc_text, truncation=True, max_length=self.max_doc_tokens)
        truncated_text = self.llm.tokenizer.decode(tokens, skip_special_tokens=True)
        return truncated_text

    def process_item(self, item: Dict[str, Any]) -> tuple:
        """
        Processes a single item for Long-Context pipeline.
        Returns result item dict and performance stats.
        """
        doc_text = item.get("document_text", "")
        question = item.get("question", "")
        ref_answer = item.get("reference_answer", "")

        # Truncate document context
        truncated_doc = self.truncate_document(doc_text)

        # Format prompt
        prompt = LLMInterface.format_long_context_prompt(truncated_doc, question)

        # Generate answer
        gen_answer, num_tokens, gen_latency = self.llm.generate(prompt)

        result_item = {
            "question": question,
            "generated_answer": gen_answer,
            "reference_answer": ref_answer
        }

        stats = {
            "latency_s": gen_latency,
            "num_tokens": num_tokens,
            "doc_length": len(truncated_doc)
        }

        return result_item, stats

    def run_benchmark(self, dataset_items: List[Dict[str, Any]], save_path: str = LONG_CONTEXT_ANSWERS_PATH) -> tuple:
        """
        Runs Long-Context pipeline over dataset_items and saves results.
        Returns list of result objects and aggregated performance stats.
        """
        print(f"Starting Long-Context pipeline benchmark for {len(dataset_items)} items...")
        results = []
        latencies = []
        token_counts = []

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for i, item in enumerate(dataset_items):
            res, stats = self.process_item(item)
            results.append(res)
            latencies.append(stats["latency_s"])
            token_counts.append(stats["num_tokens"])

            if (i + 1) % 20 == 0 or (i + 1) == len(dataset_items):
                print(f"Long-Context Progress: {i + 1}/{len(dataset_items)} completed...")

        peak_gpu_mem_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0

        # Calculate performance metrics
        latencies.sort()
        median_latency = latencies[len(latencies) // 2] if latencies else 0.0
        total_gen_time = sum(latencies)
        total_tokens = sum(token_counts)
        tokens_per_sec = (total_tokens / total_gen_time) if total_gen_time > 0 else 0.0

        perf_stats = {
            "median_query_latency_s": round(median_latency, 4),
            "generation_tokens_per_s": round(tokens_per_sec, 4),
            "peak_gpu_memory_mb": round(peak_gpu_mem_mb, 2)
        }

        # Save output JSON file
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"Saved Long-Context results ({len(results)} items) to {save_path}")
        return results, perf_stats
