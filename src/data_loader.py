import os
import json
from datasets import load_dataset
from src.config import DATASET_NAME, SAMPLE_SIZE, RESULTS_DIR

def load_narrativeqa_sample(sample_size: int = SAMPLE_SIZE, split: str = "test"):
    """
    Loads narrativeqa dataset test split and extracts sample_size items.
    Each item contains:
    - question: string
    - reference_answer: string
    - document_text: string
    - doc_id: string or int
    """
    print(f"Loading dataset {DATASET_NAME} (split={split})...")
    
    # Try loading from local cached json file if available
    cache_file = os.path.join(RESULTS_DIR, f"narrativeqa_sample_{sample_size}.json")
    if os.path.exists(cache_file):
        print(f"Loading cached sample data from {cache_file}...")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    dataset = load_dataset(DATASET_NAME, split=split)
    
    processed_items = []
    for i in range(min(sample_size, len(dataset))):
        item = dataset[i]
        
        # Extract question text
        q_text = item.get("question", {}).get("text", "")
        
        # Extract reference answer (use first answer as ground truth reference)
        answers = item.get("answers", [])
        ref_answer = ""
        if isinstance(answers, list) and len(answers) > 0:
            if isinstance(answers[0], dict):
                ref_answer = answers[0].get("text", "")
            elif isinstance(answers[0], str):
                ref_answer = answers[0]
                
        # Extract document text
        doc_obj = item.get("document", {})
        doc_text = doc_obj.get("text", "") if isinstance(doc_obj, dict) else str(doc_obj)
        
        # Fallback if document text is empty but summary is available
        if not doc_text and isinstance(doc_obj, dict):
            summary_obj = doc_obj.get("summary", {})
            if isinstance(summary_obj, dict):
                doc_text = summary_obj.get("text", "")
                
        processed_items.append({
            "id": i,
            "question": q_text,
            "reference_answer": ref_answer,
            "document_text": doc_text
        })
        
    print(f"Loaded {len(processed_items)} items successfully.")
    
    # Save cache file
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(processed_items, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not cache dataset to file: {e}")
        
    return processed_items

if __name__ == "__main__":
    items = load_narrativeqa_sample(10)
    print(f"Sample item 0 question: {items[0]['question']}")
    print(f"Sample item 0 reference answer: {items[0]['reference_answer']}")
    print(f"Sample item 0 doc length: {len(items[0]['document_text'])} chars")
