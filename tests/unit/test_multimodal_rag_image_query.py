from backend.core import multimodal_rag


class _FakeHandler:
    results_by_collection: dict[str, list[dict]] = {}
    calls_by_collection: dict[str, list[dict]] = {}

    def __init__(
        self, *, url: str, collection_name: str, vector_size: int
    ) -> None:
        self.collection_name = collection_name
        _FakeHandler.calls_by_collection.setdefault(collection_name, [])

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        user_id: str | None = None,
        folder_scopes: list[str] | None = None,
        file_ids: list[str] | None = None,
    ) -> list[dict]:
        _FakeHandler.calls_by_collection[self.collection_name].append(
            {
                'query_vector': query_vector,
                'top_k': top_k,
                'user_id': user_id,
                'folder_scopes': folder_scopes,
                'file_ids': file_ids,
            }
        )
        return list(
            _FakeHandler.results_by_collection.get(self.collection_name, [])
        )


def test_retrieve_data_uses_attachment_image_vector(monkeypatch) -> None:
    """Search image collection by uploaded image embedding, not by query text."""
    monkeypatch.setattr(multimodal_rag, 'QdrantHandler', _FakeHandler)
    monkeypatch.setattr(
        multimodal_rag.Config, 'qdrant_text_collection', 'text_collection'
    )
    monkeypatch.setattr(
        multimodal_rag.Config, 'qdrant_image_collection', 'image_collection'
    )
    monkeypatch.setattr(
        multimodal_rag.Config, 'qdrant_video_collection', 'video_collection'
    )
    monkeypatch.setattr(multimodal_rag.Config, 'text_vector_size', 2)
    monkeypatch.setattr(multimodal_rag.Config, 'image_vector_size', 2)
    monkeypatch.setattr(multimodal_rag.Config, 'video_vector_size', 2)
    monkeypatch.setattr(
        multimodal_rag, 'text_embedding', lambda query: [1.0, 1.0]
    )
    monkeypatch.setattr(
        multimodal_rag,
        'multimodal_text_embedding',
        lambda query: [2.0, 2.0],
    )
    monkeypatch.setattr(
        multimodal_rag,
        'image_embedding_from_path',
        lambda path: [9.0, 9.0],
    )

    _FakeHandler.calls_by_collection = {}
    _FakeHandler.results_by_collection = {
        'text_collection': [],
        'video_collection': [],
        'image_collection': [
            {
                'id': 'query-point',
                'score': 0.99,
                'payload': {
                    'file_id': 'query-file',
                    'modality': 'image',
                    'source': 'query-cat.jpg',
                },
            },
            {
                'id': 'other-point',
                'score': 0.95,
                'payload': {
                    'file_id': 'other-file',
                    'modality': 'image',
                    'source': 'other-cat.jpg',
                },
            },
        ],
    }

    rag = multimodal_rag.LocalRAG()

    docs = rag.retrieve_data(
        'Найди мне похожие фото',
        top_k=5,
        image_query_path='/tmp/query-cat.jpg',
        exclude_file_ids=['query-file'],
    )

    assert _FakeHandler.calls_by_collection['image_collection'][0][
        'query_vector'
    ] == [9.0, 9.0]
    assert _FakeHandler.calls_by_collection['text_collection'] == []
    assert _FakeHandler.calls_by_collection['video_collection'] == []
    assert [doc['file_id'] for doc in docs] == ['other-file']
