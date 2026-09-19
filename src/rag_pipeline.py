import os
import json
import time
import torch
from typing import List, Dict, Any
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from src.config import EMBEDDING_MODEL_ID, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_TOP_K, RAG_ANSWERS_PATH, DEVICE
from src.models import LLMInterface

class RAGPipeline:
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP, top_k: int = DEFAULT_TOP_K, llm: LLMInterface = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        print(f"Initializing embedding model ({EMBEDDING_MODEL_ID})...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_ID,
            model_kwargs={"device": DEVICE}
        )
        self.llm = llm if llm is not None else LLMInterface()

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single Q&A item:
        1. Chunks document text
        2. Builds FAISS index
        3. Retrieves top_k chunks
        4. Formats prompt and generates answer
        Returns result dict and latency/token stats.
        """
        doc_text = item.get("document_text", "")
        question = item.get("question", "")
        ref_answer = item.get("reference_answer", "")

        if not doc_text:
            doc_text = "No document text available."

        # Step 1: Chunk document
        chunks = self.text_splitter.split_text(doc_text)
        if not chunks:
            chunks = [doc_text]

        # Step 2: Build FAISS vector store
        vectorstore = FAISS.from_texts(chunks, self.embeddings)

        # Step 3: Retrieve top-K chunks
        search_start = time.perf_counter()
        retrieved_docs = vectorstore.similarity_search(question, k=self.top_k)
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]
        retrieval_latency = time.perf_counter() - search_start

        # Step 4: Format prompt & Generate Answer
        context_str = "\n\n".join(retrieved_contexts)
        prompt = LLMInterface.format_rag_prompt(context_str, question)

        gen_answer, num_tokens, gen_latency = self.llm.generate(prompt)

        total_latency = retrieval_latency + gen_latency

        result_item = {
            "question": question,
            "generated_answer": gen_answer,
            "reference_answer": ref_answer,
            "retrieved_contexts": retrieved_contexts
        }

        stats = {
            "latency_s": total_latency,
            "generation_latency_s": gen_latency,
            "num_tokens": num_tokens,
            "doc_length": len(doc_text)
        }

        return result_item, stats

    def run_benchmark(self, dataset_items: List[Dict[str, Any]], save_path: str = RAG_ANSWERS_PATH) -> tuple:
        """
        Runs RAG pipeline over all items in dataset_items and saves results.
        Returns list of result objects and aggregated performance stats.
        """
        print(f"Starting RAG pipeline benchmark for {len(dataset_items)} items...")
        results = []
        latencies = []
        token_counts = []
        gen_latencies = []

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for i, item in enumerate(dataset_items):
            res, stats = self.process_item(item)
            results.append(res)
            latencies.append(stats["latency_s"])
            token_counts.append(stats["num_tokens"])
            gen_latencies.append(stats["generation_latency_s"])

            if (i + 1) % 20 == 0 or (i + 1) == len(dataset_items):
                print(f"RAG Progress: {i + 1}/{len(dataset_items)} completed...")

        peak_gpu_mem_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0

        # Calculate performance metrics
        latencies.sort()
        median_latency = latencies[len(latencies) // 2] if latencies else 0.0
        total_gen_time = sum(gen_latencies)
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

        print(f"Saved RAG results ({len(results)} items) to {save_path}")
        return results, perf_stats
