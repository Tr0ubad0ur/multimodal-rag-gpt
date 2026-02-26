from backend.core import embeddings


class _FakeProvider:
    def encode_text(self, text: str) -> list[float]:
        return [1.0, float(len(text))]

    def encode_image(self, image) -> list[float]:
        return [2.0]

    def encode_video(
        self, video_path: str, sample_fps: float = 1.0
    ) -> list[float]:
        return [3.0, sample_fps]


def test_text_embedding_uses_default_provider_from_config(monkeypatch) -> None:
    """Use configured default provider when provider_name is not passed."""
    captured = {'provider_name': None}

    def _fake_get_provider(name: str | None = None):
        captured['provider_name'] = name
        return _FakeProvider()

    monkeypatch.setattr(
        embeddings.Config, 'default_embedding_provider', 'test-default'
    )
    monkeypatch.setattr(embeddings, 'get_provider', _fake_get_provider)

    vector = embeddings.text_embedding('abc')
    assert captured['provider_name'] == 'test-default'
    assert vector == [1.0, 3.0]


def test_video_embedding_uses_config_sample_fps(monkeypatch) -> None:
    """Use configured default sample FPS for video embeddings."""
    captured = {'sample_fps': None}

    class _FakeVideoProvider(_FakeProvider):
        def encode_video(
            self, video_path: str, sample_fps: float = 1.0
        ) -> list[float]:
            captured['sample_fps'] = sample_fps
            return [3.0, sample_fps]

    monkeypatch.setattr(embeddings.Config, 'embedding_video_sample_fps', 2.5)
    monkeypatch.setattr(
        embeddings, 'get_provider', lambda *_: _FakeVideoProvider()
    )

    vector = embeddings.video_embedding_from_path('/tmp/file.mp4')
    assert captured['sample_fps'] == 2.5
    assert vector == [3.0, 2.5]
