import time
from typing import Any, Dict, List

from backend.core.embeddings import text_embedding
from backend.monitoring.metrics import observe_rag_query
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
            url=Config.qdrant_url,
            collection_name=Config.qdrant_text_collection,
            vector_size=Config.text_vector_size,
        )

    def retrieve_data(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
        folder_scopes: list[str] | None = None,
        file_ids: list[str] | None = None,
    ) -> List[Dict[str, str]]:
        """Retrieve top-K most similar documents from Qdrant for a given query.

        Args:
            query (str): User text query.
            top_k (int, optional): Number of top documents to retrieve. Defaults to 5.
            user_id (str | None, optional): Filter results by user id. Defaults to None.
            folder_scopes (list[str] | None, optional): Optional folder scope filter.
            file_ids (list[str] | None, optional): Optional file filter.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing the retrieved documents:
                - 'text' (str): The text content of the document chunk.
                - 'source' (str): Source file path or metadata for the chunk.
        """
        query_vector = text_embedding(query)
        results = self.client.search(
            query_vector=query_vector,
            top_k=top_k,
            user_id=user_id,
            folder_scopes=folder_scopes,
            file_ids=file_ids,
        )

        docs = []
        for r in results:
            payload = r.get('payload', {})
            docs.append(
                {
                    'text': payload.get('text', ''),
                    'source': payload.get('source', ''),
                }
            )

        return docs

    def generate_answer(
        self,
        query: str,
        top_k: int = 5,
        image=None,
        user_id: str | None = None,
        model: str | None = None,
        folder_scopes: list[str] | None = None,
        file_ids: list[str] | None = None,
        extra_docs: list[dict[str, str]] | None = None,
    ) -> Dict[str, Any]:
        """Generate an answer using the local LLM based on the query and optionally an image.

        Args:
            query (str): User question or prompt.
            top_k (int, optional): Number of top documents to retrieve for context. Defaults to 5.
            image (Optional[str], optional): Optional image path or URL to include in the prompt. Defaults to None.
            user_id (str | None, optional): Filter context by user id. Defaults to None.
            model (str | None, optional): LLM model name hint for backend routing.
            folder_scopes (list[str] | None, optional): Optional folder filter for retrieval.
            file_ids (list[str] | None, optional): Optional file filter for retrieval.
            extra_docs (list[dict[str, str]] | None, optional): Extra context docs from attachments.

        Returns:
            Dict[str, object]: A dictionary with:
                - 'answer' (str): The generated text answer from the LLM.
                - 'retrieved_docs' (List[Dict[str, str]]): The list of retrieved documents used as context.
        """
        query_type = 'multimodal' if image else 'text'
        started = time.perf_counter()
        status = 'ok'
        docs: List[Dict[str, str]] = []

        try:
            from backend.core.llm import get_llm_response

            docs = self.retrieve_data(
                query,
                top_k,
                user_id=user_id,
                folder_scopes=folder_scopes,
                file_ids=file_ids,
            )
            final_docs = docs + (extra_docs or [])
            answer_text = get_llm_response(
                query, context=final_docs, image=image, model=model
            )
            return {'answer': answer_text, 'retrieved_docs': final_docs}
        except Exception:
            status = 'error'
            raise
        finally:
            observe_rag_query(
                query_type=query_type,
                status=status,
                duration_seconds=time.perf_counter() - started,
                retrieved_docs_count=len(docs),
            )
