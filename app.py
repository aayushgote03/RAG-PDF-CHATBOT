import streamlit as st
from src.ingest import create_vector_db_from_upload
from langchain_community.callbacks import StreamlitCallbackHandler
from src.rag_engine import get_qa_chain
from src.orchestrator import SmartOrchestrator
from src.ingest import create_vector_db
import os

with st.sidebar:
    st.header("📂 Document Center")
    st.write("Upload your own PDF to chat with it.")
    
    uploaded_files = st.file_uploader(
        "Upload PDF", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("🚀 Process Documents"):
        with st.spinner("Analyzing documents... this may take a moment."):
            success = create_vector_db_from_upload(uploaded_files)
            if success:
                st.success("Documents Processed! Clearing chat memory...")
                st.session_state.messages = [] # Reset chat
                st.cache_resource.clear() # Force reload of the chain
                st.rerun()

# --- PAGE CONFIG ---
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 PDF AI Assistant")

# --- SIDEBAR: SETTINGS & INGEST ---
with st.sidebar:
    st.header("Settings")
    if st.button("🔄 Refresh/Ingest Data"):
        with st.spinner("Ingesting PDF..."):
            create_vector_db()
            st.success("Ingestion Complete! Reload the app.")

# --- CACHING THE MODEL (IMPORTANT) ---
# We use @st.cache_resource so the model loads ONLY ONCE, not on every click.
@st.cache_resource
def setup_chain():
    # We pass None for now, callbacks are added dynamically during execution
    chain, db = get_qa_chain(callback_handler=None)
    return chain, db

try:
    qa_chain, db = setup_chain()
    # Initialize logic helper
    orchestrator = SmartOrchestrator(qa_chain, db)
except Exception as e:
    st.error(f"Error loading system: {e}. Did you run 'Refresh Data'?")
    st.stop()

# --- CHAT HISTORY STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I have read the Company Policy. Ask me anything."}
    ]

# --- DISPLAY CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- USER INPUT HANDLER ---
if prompt := st.chat_input("What is the policy on sustainability?"):
    
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate AI Response
    with st.chat_message("assistant"):
        # Create a container for the streaming text
        st_callback = StreamlitCallbackHandler(st.container())
        
        # We need to inject the streamlit callback into the LLM at runtime
        # LangChain allows passing 'callbacks' in invoke/run
        
        # Check Confidence First (Quick Check)
        score, best_doc = orchestrator.get_similarity_score(prompt)
        
        if score > 1.5: # Threshold from config
            response_text = "I'm sorry, I couldn't find relevant information in the document to answer that confidently."
            st.markdown(response_text)
        else:
            # Run the Chain with the Streamlit Callback
            response = qa_chain.invoke(
                {"query": prompt}, 
                config={"callbacks": [st_callback]}
            )
            response_text = response['result']
            
            # Show Source Metadata in an expander
            with st.expander("View Source & Confidence"):
                st.write(f"**Confidence Score:** {score:.2f}")
                st.write(f"**Source Preview:** {response['source_documents'][0].page_content[:200]}...")

    # 3. Save AI message to history
    st.session_state.messages.append({"role": "assistant", "content": response_text})