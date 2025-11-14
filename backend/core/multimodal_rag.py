import os
from typing import List
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from backend.core.llm import get_llm_response

# --- SETTINGS ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")
COLLECTION_NAME = "documents"

# --- EMBEDDINGS ---
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# --- QDRANT ---
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# --- encode text for the DB ---
def embed_text(text: str) -> List[float]:
    return embedding_model.encode(text).tolist()


# --- search in vector DB ---
def search_similar(query: str, limit: int = 3) -> List[ScoredPoint]:
    vector = embed_text(query)
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=limit
    )
    return hits


# --- build RAG prompt ---
def build_prompt(question: str, docs: List[str]) -> str:
    context_text = "\n".join([f"- {d}" for d in docs])
    prompt = f"""
You are a helpful assistant.

User question:
{question}

Relevant context:
{context_text}

Based on the context above, provide the best possible answer.
If the context is not enough — say so explicitly.
"""
    return prompt


# --- full multimodal RAG pipeline ---
def multimodal_rag(question: str, image=None):
    # 1 — search text context
    hits = search_similar(question)
    docs = [hit.payload["text"] for hit in hits]

    # 2 — build final RAG prompt
    final_prompt = build_prompt(question, docs)

    # 3 — call LLM (with or without image)
    response = get_llm_response(final_prompt, image=image)

    return {
        "question": question,
        "context_used": docs,
        "answer": response
    }
