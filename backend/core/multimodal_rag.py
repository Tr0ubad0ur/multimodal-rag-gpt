from qdrant_client import QdrantClient
from backend.core.embeddings import text_embedding
from backend.core.llm import get_llm_response

COLLECTION_NAME = "documents"

class LocalRAG:
    """Local Retrieval-Augmented Generation (RAG) pipeline using Qdrant and a local LLM.

    Attributes:
        client (QdrantClient): Client to connect to the local Qdrant vector database.
    """
    def __init__(self):
        """Initialize the LocalRAG pipeline by connecting to the Qdrant instance."""
        self.client = QdrantClient(host="localhost", port=6333)

    def retrieve(self, query: str, top_k: int = 5):
        """Retrieve top-K most similar documents from Qdrant for a given query.

        Args:
            query (str): User text query.
            top_k (int, optional): Number of top documents to retrieve. Defaults to 5.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing the retrieved documents:
                - 'text' (str): The text content of the document chunk.
                - 'source' (str): Source file path or metadata for the chunk.
        """
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
        """Generate an answer using the local LLM based on the query and optionally an image.

        Args:
            query (str): User question or prompt.
            top_k (int, optional): Number of top documents to retrieve for context. Defaults to 5.
            image (Optional[str], optional): Optional image path or URL to include in the prompt. Defaults to None.

        Returns:
            Dict[str, object]: A dictionary with:
                - 'answer' (str): The generated text answer from the LLM.
                - 'retrieved_docs' (List[Dict[str, str]]): The list of retrieved documents used as context.
        """
        docs = self.retrieve(query, top_k)
        answer_text = get_llm_response(query, context=docs, image=image)
        return {"answer": answer_text, "retrieved_docs": docs}
