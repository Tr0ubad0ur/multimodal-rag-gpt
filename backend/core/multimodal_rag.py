from qdrant_client import QdrantClient
from backend.core.embeddings import text_embedding
from backend.core.llm import get_llm_response

COLLECTION_NAME = "documents"

class LocalRAG:
    def __init__(self):
        self.client = QdrantClient(host="localhost", port=6333)

    def retrieve(self, query: str, top_k: int = 5):
        query_vector = text_embedding(query)
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )
        docs = []
        for r in results:
            docs.append({
                "text": r.payload.get("text", ""),
                "source": r.payload.get("source", "")
            })
        return docs

    def generate_answer(self, query: str, top_k: int = 5, image=None):
        docs = self.retrieve(query, top_k)
        answer_text = get_llm_response(query, context=docs, image=image)
        return {"answer": answer_text, "retrieved_docs": docs}
