from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import (
    DB_FAISS_PATH, 
    EMBEDDING_MODEL_NAME, 
    LLM_MODEL_NAME, 
    RETRIEVER_K
)

def load_resources():
    print("Loading Embedding Model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    
    print("Loading Vector Database...")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db, embeddings

# --- CRITICAL FIX IS HERE: Add 'callback_handler' parameter ---
def get_qa_chain(callback_handler=None):
    """
    callback_handler: A specific handler to stream tokens to the UI (Terminal or Streamlit)
    """
    # 1. Load Resources
    db, _ = load_resources()
    retriever = db.as_retriever(search_kwargs={"k": RETRIEVER_K})

    # 2. Setup Callbacks
    # If a handler is passed (like from Streamlit), use it. Otherwise empty list.
    callbacks = [callback_handler] if callback_handler else []

    # 3. Load Local LLM
    print(f"Loading Local {LLM_MODEL_NAME} via Ollama...")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",  # Groq's version of Llama 3
        temperature=0,
        api_key="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", # We will set this securely later
        callbacks=[callback_handler] if callback_handler else [],
        streaming=True
    )

    # 4. Define Prompt
    detailed_template = """You are a helpful AI assistant. Read the context below carefully.
The context might contain headers or metadata. Ignore them and focus on the main content.
Provide a comprehensive and detailed answer to the question.
If the context doesn't contain the answer, say "I don't know".

Context:
{context}

Question: 
{question}

Detailed Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(detailed_template)

    # 5. Build Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    return qa_chain, db