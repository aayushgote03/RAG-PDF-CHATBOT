import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'company_policy.pdf')
DB_FAISS_PATH = os.path.join(BASE_DIR, 'vector_store', 'db_faiss')

# Models
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL_NAME = "llama3"

# Tuning parameters
CONFIDENCE_THRESHOLD = 1.5  # Distance score threshold
RETRIEVER_K = 5            # Number of chunks to retrieve