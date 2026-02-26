from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.api.endpoints import (
    ImageEmbeddingRequest,
    QueryRequest,
    TextEmbeddingRequest,
    VideoEmbeddingRequest,
)


def test_query_request_rejects_empty_query() -> None:
    """Reject query payloads that become empty after trimming."""
    with pytest.raises(ValidationError):
        QueryRequest(query='   ')


def test_text_embedding_request_rejects_empty_text() -> None:
    """Reject text embedding payloads that become empty after trimming."""
    with pytest.raises(ValidationError):
        TextEmbeddingRequest(text='   ')


def test_image_embedding_request_requires_existing_path(
    tmp_path: Path,
) -> None:
    """Require image_path to point to an existing local file."""
    missing = tmp_path / 'missing.jpg'
    with pytest.raises(ValidationError):
        ImageEmbeddingRequest(image_path=str(missing))


def test_video_embedding_request_accepts_existing_path(tmp_path: Path) -> None:
    """Accept valid video_path when file exists locally."""
    video_file = tmp_path / 'sample.mp4'
    video_file.write_bytes(b'00')
    request = VideoEmbeddingRequest(video_path=str(video_file), sample_fps=1.0)
    assert request.video_path == str(video_file)
