from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

# 2. LOAD LOCAL MODEL (Ollama)
# Make sure you have installed Ollama and ran 'ollama pull llama3' in your terminal
print("Loading Local Llama 3 via Ollama...")
llm = ChatOllama(model="llama3", temperature=0) # temp=0 for factual answers

# 3. SETUP RETRIEVER
# Load the Vector DB from Cell 2
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 2}) # Read top 5 chunks

# 4. DEFINE PROMPT
template = """
Context:
{context}

Question: 
{question}

Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# 5. BUILD CHAINS
# Chain A: Simple QA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# Chain B: Chat Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain_with_memory = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    chain_type="stuff",
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
    verbose=False
)

print("✅ Local Llama 3 (Ollama) Loaded & Chains Ready.")