from typing import Any, Dict, List

from backend.core.embeddings import text_embedding
from backend.core.llm import get_llm_response
from backend.utils.config_handler import Config
from backend.utils.qdrant_handler import QdrantHandler


class LocalRAG:
    """Local Retrieval-Augmented Generation (RAG) pipeline using Qdrant and a local LLM.

    Attributes:
        client (QdrantClient): Client to connect to the local Qdrant vector database.
    """

    def __init__(self) -> None:
        """Initialize the LocalRAG pipeline by connecting to the Qdrant instance."""
        self.client = QdrantHandler(
            url='localhost:6333',
            collection_name=Config.qdrant_text_collection,
            vector_size=Config.text_vector_size,
        )

    def retrieve_data(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, str]]:
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
        results = self.client.search(query_vector=query_vector)

        docs = []
        for r in results:
            docs.append(
                {
                    'text': r['payload'].get('text', ''),
                    'source': r['payload'].get('source', ''),
                }
            )
        return docs

    def generate_answer(
        self, query: str, top_k: int = 5, image=None
    ) -> Dict[str, Any]:
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
        docs = self.retrieve_data(query, top_k)
        answer_text = get_llm_response(query, context=docs, image=image)
        return {'answer': answer_text, 'retrieved_docs': docs}
