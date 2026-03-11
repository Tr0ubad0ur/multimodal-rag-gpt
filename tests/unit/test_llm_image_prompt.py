from backend.core import llm, multimodal_rag


class _FakeBackend:
    supports_image = True

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, prompt: str, context=None, image=None) -> str:
        self.calls.append(
            {'prompt': prompt, 'context': context, 'image': image}
        )
        return 'ok'


def test_get_llm_response_includes_image_retrieval_instructions(
    monkeypatch,
) -> None:
    """Prompt should explicitly describe visual retrieval when image hits exist."""
    backend = _FakeBackend()
    monkeypatch.setattr(llm, '_resolve_llm_backend', lambda model: backend)

    result = llm.get_llm_response(
        'Найди похожие фото',
        context=[
            {
                'source': 'cats/a.jpg',
                'text': 'Image attachment: cat',
                'modality': 'image',
                'score': 0.91,
                'preview_ref': '/files/a/download',
                'file_id': 'a',
            }
        ],
        image='/tmp/query-cat.jpg',
    )

    assert result == 'ok'
    assert backend.calls[0]['image'] == '/tmp/query-cat.jpg'
    assert 'визуальному сходству' in backend.calls[0]['prompt']
    assert 'preview_ref=/files/a/download' in backend.calls[0]['prompt']


def test_generate_answer_passes_attachment_image_to_llm(monkeypatch) -> None:
    """Attachment image path should be forwarded to the final LLM call."""
    captured: dict[str, object] = {}

    def _fake_get_llm_response(prompt, context=None, image=None, model=None):
        captured['context'] = context
        captured['image'] = image
        return 'answer'

    monkeypatch.setattr(multimodal_rag.LocalRAG, '__init__', lambda self: None)
    monkeypatch.setattr(
        multimodal_rag.LocalRAG,
        'retrieve_data',
        lambda self, *args, **kwargs: [
            {
                'text': 'Image attachment: cat',
                'source': 'cats/a.jpg',
                'file_id': 'a',
                'modality': 'image',
                'score': 0.91,
                'preview_ref': '/files/a/download',
            }
        ],
    )
    monkeypatch.setattr(llm, 'get_llm_response', _fake_get_llm_response)

    rag = multimodal_rag.LocalRAG()
    result = rag.generate_answer(
        'Найди похожие фото',
        top_k=1,
        image_query_path='/tmp/query-cat.jpg',
    )

    assert result['answer'] == 'answer'
    assert result['retrieved_docs'][0]['file_id'] == 'a'
    assert captured['context'][0]['file_id'] == 'a'
    assert captured['image'] == '/tmp/query-cat.jpg'
