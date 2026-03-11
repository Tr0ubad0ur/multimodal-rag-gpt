import time
from typing import Any, Dict, List

from backend.core.embeddings import (
    image_embedding_from_path,
    multimodal_text_embedding,
    text_embedding,
)
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
        self.image_client = QdrantHandler(
            url=Config.qdrant_url,
            collection_name=Config.qdrant_image_collection,
            vector_size=Config.image_vector_size,
        )
        self.video_client = QdrantHandler(
            url=Config.qdrant_url,
            collection_name=Config.qdrant_video_collection,
            vector_size=Config.video_vector_size,
        )

    def retrieve_data(
        self,
        query: str,
        top_k: int = 5,
        image_query_path: str | None = None,
        user_id: str | None = None,
        folder_scopes: list[str] | None = None,
        file_ids: list[str] | None = None,
        exclude_file_ids: list[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K most similar documents from Qdrant for a given query.

        Args:
            query (str): User text query.
            top_k (int, optional): Number of top documents to retrieve. Defaults to 5.
            image_query_path (str | None, optional): Optional local path to an image used as the retrieval query. Defaults to None.
            user_id (str | None, optional): Filter results by user id. Defaults to None.
            folder_scopes (list[str] | None, optional): Optional folder scope filter.
            file_ids (list[str] | None, optional): Optional file filter.
            exclude_file_ids (list[str] | None, optional): Optional file ids to remove from retrieval results, such as the query attachment itself.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing the retrieved documents:
                - 'text' (str): The text content of the document chunk.
                - 'source' (str): Source file path or metadata for the chunk.
        """
        excluded_ids = {str(file_id) for file_id in (exclude_file_ids or [])}
        search_limit = top_k + len(excluded_ids)

        if image_query_path:
            image_query_vector = image_embedding_from_path(image_query_path)
            text_results = []
        else:
            text_results = self._filter_excluded_results(
                self.client.search(
                    query_vector=text_embedding(query),
                    top_k=search_limit,
                    user_id=user_id,
                    folder_scopes=folder_scopes,
                    file_ids=file_ids,
                ),
                excluded_ids=excluded_ids,
            )
            image_query_vector = multimodal_text_embedding(query)
        image_results = self._filter_excluded_results(
            self.image_client.search(
                query_vector=image_query_vector,
                top_k=search_limit,
                user_id=user_id,
                folder_scopes=folder_scopes,
                file_ids=file_ids,
            ),
            excluded_ids=excluded_ids,
        )
        if image_query_path:
            video_results = []
        else:
            video_results = self._filter_excluded_results(
                self.video_client.search(
                    query_vector=multimodal_text_embedding(query),
                    top_k=search_limit,
                    user_id=user_id,
                    folder_scopes=folder_scopes,
                    file_ids=file_ids,
                ),
                excluded_ids=excluded_ids,
            )

        results = self._merge_results(
            text_results=text_results,
            image_results=image_results,
            video_results=video_results,
            top_k=top_k,
        )

        docs: list[dict[str, Any]] = []
        for r in results:
            payload = r.get('payload', {})
            docs.append(
                {
                    'text': payload.get('text', ''),
                    'source': payload.get('source', ''),
                    'file_id': payload.get('file_id'),
                    'modality': payload.get('modality', 'text'),
                    'score': r.get('score'),
                    'preview_ref': self._build_preview_ref(payload),
                }
            )

        return docs

    def _filter_excluded_results(
        self,
        results: list[dict[str, Any]],
        *,
        excluded_ids: set[str],
    ) -> list[dict[str, Any]]:
        """Drop hits that refer to explicitly excluded file ids."""
        if not excluded_ids:
            return results
        filtered: list[dict[str, Any]] = []
        for item in results:
            payload = item.get('payload', {}) or {}
            file_id = str(payload.get('file_id') or '')
            if file_id and file_id in excluded_ids:
                continue
            filtered.append(item)
        return filtered

    def _merge_results(
        self,
        *,
        text_results: list[dict[str, Any]],
        image_results: list[dict[str, Any]],
        video_results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Merge modality-specific candidates with reciprocal rank fusion."""
        by_id: dict[str, dict[str, Any]] = {}
        for results in (text_results, image_results, video_results):
            for rank, item in enumerate(results, start=1):
                payload = item.get('payload', {}) or {}
                file_id = str(payload.get('file_id') or item.get('id') or '')
                if not file_id:
                    continue
                entry = by_id.setdefault(
                    file_id,
                    {
                        'id': item.get('id'),
                        'payload': payload,
                        'score': 0.0,
                        'raw_score': item.get('score'),
                    },
                )
                entry['score'] += 1.0 / (60.0 + rank)
                raw_score = item.get('score')
                if raw_score is not None:
                    prev_raw = entry.get('raw_score')
                    if prev_raw is None or raw_score > prev_raw:
                        entry['raw_score'] = raw_score
                if payload.get('modality') == 'text':
                    entry['payload'] = payload

        merged = sorted(
            by_id.values(),
            key=lambda item: float(item.get('score') or 0.0),
            reverse=True,
        )
        return merged[:top_k]

    def _build_preview_ref(self, payload: dict[str, Any]) -> str | None:
        """Return a stable preview reference for a retrieved source."""
        file_id = payload.get('file_id')
        if file_id:
            return f'/files/{file_id}/download'
        return payload.get('source_path') or payload.get('source')

    def _build_used_sources(
        self, docs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Normalize retrieved docs into frontend-facing source descriptors."""
        seen: set[tuple[str | None, str | None, str | None]] = set()
        used_sources: list[dict[str, Any]] = []
        for doc in docs:
            key = (
                doc.get('file_id'),
                doc.get('modality'),
                doc.get('preview_ref'),
            )
            if key in seen:
                continue
            seen.add(key)
            used_sources.append(
                {
                    'file_id': doc.get('file_id'),
                    'modality': doc.get('modality', 'text'),
                    'score': doc.get('score'),
                    'preview_ref': doc.get('preview_ref'),
                    'source': doc.get('source'),
                }
            )
        return used_sources

    def generate_answer(
        self,
        query: str,
        top_k: int = 5,
        image=None,
        image_query_path: str | None = None,
        user_id: str | None = None,
        model: str | None = None,
        folder_scopes: list[str] | None = None,
        file_ids: list[str] | None = None,
        exclude_file_ids: list[str] | None = None,
        extra_docs: list[dict[str, str]] | None = None,
    ) -> Dict[str, Any]:
        """Generate an answer using the local LLM based on the query and optionally an image.

        Args:
            query (str): User question or prompt.
            top_k (int, optional): Number of top documents to retrieve for context. Defaults to 5.
            image (Optional[str], optional): Optional image path or URL to include in the prompt. Defaults to None.
            image_query_path (str | None, optional): Optional local path to an attachment image used for image retrieval. Defaults to None.
            user_id (str | None, optional): Filter context by user id. Defaults to None.
            model (str | None, optional): LLM model name hint for backend routing.
            folder_scopes (list[str] | None, optional): Optional folder filter for retrieval.
            file_ids (list[str] | None, optional): Optional file filter for retrieval.
            exclude_file_ids (list[str] | None, optional): Optional file ids to exclude from retrieval results. Defaults to None.
            extra_docs (list[dict[str, str]] | None, optional): Extra context docs from attachments.

        Returns:
            Dict[str, object]: A dictionary with:
                - 'answer' (str): The generated text answer from the LLM.
                - 'retrieved_docs' (List[Dict[str, str]]): The list of retrieved documents used as context.
        """
        effective_image = image or image_query_path
        query_type = 'multimodal' if effective_image else 'text'
        started = time.perf_counter()
        status = 'ok'
        docs: List[Dict[str, Any]] = []

        try:
            from backend.core.llm import get_llm_response

            docs = self.retrieve_data(
                query,
                top_k,
                image_query_path=image_query_path,
                user_id=user_id,
                folder_scopes=folder_scopes,
                file_ids=file_ids,
                exclude_file_ids=exclude_file_ids,
            )
            final_docs = docs + (extra_docs or [])
            answer_text = get_llm_response(
                query,
                context=final_docs,
                image=effective_image,
                model=model,
            )
            return {
                'answer': answer_text,
                'retrieved_docs': final_docs,
                'used_sources': self._build_used_sources(final_docs),
            }
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
