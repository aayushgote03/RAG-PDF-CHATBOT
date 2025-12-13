from src.config import CONFIDENCE_THRESHOLD

class SmartOrchestrator:
    def __init__(self, qa_chain, db):
        self.qa_chain = qa_chain
        self.db = db

    def get_similarity_score(self, query):
        """
        Checks how close the best document is to the query using Euclidean Distance.
        """
        docs_and_scores = self.db.similarity_search_with_score(query, k=2)
        if not docs_and_scores:
            return 100.0, None # High distance (bad match)
        
        score = docs_and_scores[0][1] # The FAISS distance score
        content = docs_and_scores[0][0].page_content
        return score, content

    def smart_query(self, user_query):
        """
        1. Checks Intent (Chat vs Q&A)
        2. Checks Confidence (Is the document relevant?)
        3. Returns the answer
        """
        # 1. INTENT DETECTION
        chat_keywords = ['hi', 'hello', 'hey', 'thanks', 'thank you', 'bye']
        if user_query.lower().strip() in chat_keywords:
            return {
                "answer": "Hello! I am your AI Assistant. Ask me anything about your documents.",
                "source": "Chat Logic (No Retrieval)",
                "confidence": "N/A"
            }

        # 2. CONFIDENCE CHECK
        score, best_doc = self.get_similarity_score(user_query)
        # Debug print can be removed in production
        print(f"[DEBUG] Raw Distance Score: {score:.4f}") 

        if score > CONFIDENCE_THRESHOLD:
            return {
                "answer": "I'm sorry, but I couldn't find enough information in the document to answer that confidently.",
                "source": "Confidence Guardrail (Refusal)",
                "confidence": f"Low (Score: {score:.2f})"
            }

        # 3. RAG EXECUTION
        result = self.qa_chain.invoke({"query": user_query})
        
        return {
            "answer": result['result'],
            "source": result['source_documents'][0].page_content[:100] + "...",
            "confidence": f"High (Score: {score:.2f})"
        }