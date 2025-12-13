import argparse
from src.ingest import create_vector_db
from src.rag_engine import get_qa_chain
from src.orchestrator import SmartOrchestrator

def start_chat():
    print("⏳ Initializing System...")
    try:
        qa_chain, db = get_qa_chain()
        orchestrator = SmartOrchestrator(qa_chain, db)
        print("✅ System Ready. Type 'exit' to quit.\n")
    except Exception as e:
        print(f"❌ Error loading system: {e}")
        print("Tip: Did you run 'python main.py --ingest' first?")
        return

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        response = orchestrator.smart_query(user_input)
        
        print(f"AI: {response['answer']}")
        print(f"   [Meta]: Source='{response['source']}', Confidence='{response['confidence']}'")
        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GenAI RAG System")
    parser.add_argument("--ingest", action="store_true", help="Process PDF and create Vector DB")
    args = parser.parse_args()

    if args.ingest:
        create_vector_db()
    else:
        start_chat()