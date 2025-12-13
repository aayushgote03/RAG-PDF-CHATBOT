import os
import re
import tempfile
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import DATA_PATH, DB_FAISS_PATH, EMBEDDING_MODEL_NAME

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace/newlines
    text = re.sub(r'\.\.\.', '', text)  # Remove '...' artifacts
    return text.strip()

def create_vector_db():
    # Check if vector database already exists
    if os.path.exists(DB_FAISS_PATH) and os.path.exists(os.path.join(DB_FAISS_PATH, "index.faiss")):
        print(f"✅ Vector Database already exists at '{DB_FAISS_PATH}'")
        print("ℹ️  Skipping ingestion. Delete the folder to recreate.")
        return
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: File not found at {DATA_PATH}")
        return

    # 1. Load Data
    print(f"Loading {DATA_PATH}...")
    loader = PyPDFLoader(DATA_PATH)
    raw_documents = loader.load()

    # 2. Clean Data
    for doc in raw_documents:
        doc.page_content = clean_text(doc.page_content)
    print(f"Loaded {len(raw_documents)} pages and cleaned text.")

    # 3. Load Embeddings
    print("Loading Embedding Model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

    # 4. Semantic Chunking
    print("Splitting text based on Semantic Meaning...")
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"
    )
    texts = text_splitter.split_documents(raw_documents)
    print(f"✅ Created {len(texts)} Meaningful Chunks.")

    # 5. Create & Save Vector DB
    print("Creating Vector Database...")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"✅ Vector Database saved to '{DB_FAISS_PATH}'")


def create_vector_db_from_upload(uploaded_files):
    """
    Handles file upload from Streamlit.
    uploaded_files: List of Streamlit file objects
    """
    # 1. Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    
    all_documents = []
    
    # 2. Process each uploaded file
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            # Save the uploaded file to a temporary location
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Load PDF
            loader = PyPDFLoader(temp_path)
            all_documents.extend(loader.load())

    # 3. Semantic Chunking
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    texts = text_splitter.split_documents(all_documents)
    
    # 4. Create/Overwrite Vector DB
    # Note: This overwrites the existing DB for the new session
    if os.path.exists(DB_FAISS_PATH):
        shutil.rmtree(DB_FAISS_PATH) # Clean old DB
        
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    
    return True

if __name__ == "__main__":
    create_vector_db()