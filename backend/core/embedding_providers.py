from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from backend.utils.config_handler import Config


class EmbeddingProvider(ABC):
    """Base interface for embedding providers."""

    @abstractmethod
    def encode_text(self, text: str) -> list[float]:
        """Encode text into an embedding vector."""

    @abstractmethod
    def encode_multimodal_text(self, text: str) -> list[float]:
        """Encode text into the shared image/video embedding space."""

    @abstractmethod
    def encode_image(self, image: Image.Image) -> list[float]:
        """Encode image into an embedding vector."""

    @abstractmethod
    def encode_video(
        self, video_path: str, sample_fps: float = 1.0
    ) -> list[float]:
        """Encode video into an embedding vector."""


class SentenceTransformerProvider(EmbeddingProvider):
    """SentenceTransformers-based provider for text, image and video."""

    def __init__(
        self,
        text_model_name: str = 'all-MiniLM-L6-v2',
        image_model_name: str = 'clip-ViT-B-32',
    ) -> None:
        """Initialize text and image encoders used for all modalities.

        Args:
            text_model_name (str): SentenceTransformers model name for text.
            image_model_name (str): SentenceTransformers CLIP-like model for images.
        """
        self._text_model = SentenceTransformer(text_model_name)
        self._image_model = SentenceTransformer(image_model_name)

    def encode_text(self, text: str) -> list[float]:
        """Encode a text string into a dense vector."""
        return self._text_model.encode(text).tolist()

    def encode_multimodal_text(self, text: str) -> list[float]:
        """Encode text in CLIP-compatible space for image/video retrieval."""
        return self._image_model.encode(text).tolist()

    def encode_image(self, image: Image.Image) -> list[float]:
        """Encode a PIL image into a dense vector."""
        rgb_image = image.convert('RGB')
        return self._image_model.encode(
            rgb_image,
            convert_to_numpy=True,
        ).tolist()

    def encode_video(
        self, video_path: str, sample_fps: float = 1.0
    ) -> list[float]:
        """Encode a video by sampling frames and mean-pooling frame vectors.

        Args:
            video_path (str): Path to a local video file.
            sample_fps (float): Target frame sampling rate.

        Returns:
            list[float]: Aggregated video embedding.
        """
        if sample_fps <= 0:
            raise ValueError('sample_fps must be > 0')

        # Delayed import avoids loading video stack for text-only workloads.
        from torchvision.io import read_video

        frames, _, info = read_video(video_path, pts_unit='sec')
        if frames.numel() == 0:
            raise ValueError(f'No frames extracted from video: {video_path}')

        source_fps = float(info.get('video_fps') or 1.0)
        stride = max(int(round(source_fps / sample_fps)), 1)
        sampled_frames = frames[::stride]
        if sampled_frames.numel() == 0:
            sampled_frames = frames[:1]

        pil_frames = [
            Image.fromarray(frame.numpy()).convert('RGB')
            for frame in sampled_frames
        ]
        frame_vectors = self._image_model.encode(
            pil_frames,
            convert_to_numpy=True,
        )
        video_vector = np.mean(frame_vectors, axis=0)
        return video_vector.tolist()


_PROVIDER_INSTANCES: Dict[str, EmbeddingProvider] = {}


def _build_provider(name: str) -> EmbeddingProvider:
    provider_config = Config.embedding_providers.get(name)
    if not provider_config:
        available = ', '.join(sorted(Config.embedding_providers))
        raise ValueError(
            f'Unknown embedding provider "{name}". Available: {available}'
        )

    provider_type = provider_config.get('type')
    if provider_type == 'sentence-transformers':
        return SentenceTransformerProvider(
            text_model_name=provider_config['text_model_name'],
            image_model_name=provider_config['image_model_name'],
        )
    raise ValueError(
        f'Unsupported provider type "{provider_type}" for "{name}"'
    )


def get_provider(name: str | None = None) -> EmbeddingProvider:
    """Resolve embedding provider by name."""
    resolved_name = name or Config.default_embedding_provider
    if resolved_name not in _PROVIDER_INSTANCES:
        _PROVIDER_INSTANCES[resolved_name] = _build_provider(resolved_name)
    return _PROVIDER_INSTANCES[resolved_name]
