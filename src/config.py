import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset configuration
DATASET_NAME = "deepmind/narrativeqa"
SAMPLE_SIZE = 200

# Model configuration
LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# RAG configuration
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 5

# Long Context configuration
MAX_DOC_TOKENS = 6000

# Device settings
HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if HAS_CUDA else "cpu"

# File output paths
RAG_ANSWERS_PATH = os.path.join(RESULTS_DIR, "rag_answers.json")
LONG_CONTEXT_ANSWERS_PATH = os.path.join(RESULTS_DIR, "long_context_answers.json")
EVALUATION_SCORES_PATH = os.path.join(RESULTS_DIR, "evaluation_scores.json")
POSITION_SENSITIVITY_PATH = os.path.join(RESULTS_DIR, "position_sensitivity_analysis.json")
POSITION_SENSITIVITY_CHART_PATH = os.path.join(RESULTS_DIR, "position_sensitivity_chart.png")
PERFORMANCE_METRICS_PATH = os.path.join(RESULTS_DIR, "performance_metrics.json")
