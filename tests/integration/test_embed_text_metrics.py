import re

from fastapi.testclient import TestClient

from backend.main import app


class _FakeProvider:
    def encode_text(self, text: str) -> list[float]:
        return [0.1, 0.2, float(len(text))]

    def encode_image(self, image) -> list[float]:
        return [0.0]

    def encode_video(
        self, video_path: str, sample_fps: float = 1.0
    ) -> list[float]:
        return [0.0]


def _extract_counter(metrics_text: str, provider: str) -> float:
    pattern = (
        r'embedding_requests_total\{'
        r'.*modality="text".*provider="'
        + re.escape(provider)
        + r'".*status="ok".*'
        r'\}\s+([0-9.]+)'
    )
    match = re.search(pattern, metrics_text)
    if not match:
        return 0.0
    return float(match.group(1))


def test_embed_text_increments_prometheus_counter(monkeypatch) -> None:
    """Ensure /embed/text increments embedding_requests_total for text modality."""
    from backend.core import embeddings

    monkeypatch.setattr(embeddings, 'get_provider', lambda *_: _FakeProvider())
    client = TestClient(app)

    provider = 'test-provider'
    metrics_before = client.get('/metrics').text
    before_value = _extract_counter(metrics_before, provider)

    response = client.post(
        '/embed/text',
        json={'text': 'hello metrics', 'provider': provider},
    )
    assert response.status_code == 200
    assert response.json()['modality'] == 'text'
    assert response.json()['dimension'] == 3

    metrics_after = client.get('/metrics').text
    after_value = _extract_counter(metrics_after, provider)
    assert after_value == before_value + 1
