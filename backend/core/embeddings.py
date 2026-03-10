import time
from typing import List

from PIL import Image

from backend.core.embedding_providers import get_provider
from backend.monitoring.metrics import observe_embedding_request
from backend.utils.config_handler import Config


def text_embedding(
    text: str,
    provider_name: str | None = None,
) -> list[float]:
    """Generate an embedding vector for a given text string.

    Args:
        text (str): The input text to encode.
        provider_name (str | None): Optional provider name override.

    Returns:
        List[float]: A list of floats representing the text embedding.

    Notes:
        - This embedding can be stored in a vector database like Qdrant.
        - Ensure the model used for text embeddings is compatible with your retrieval pipeline.
    """
    started = time.perf_counter()
    resolved_provider = provider_name or Config.default_embedding_provider
    status = 'ok'
    try:
        provider = get_provider(resolved_provider)
        vector = provider.encode_text(text)
        return vector
    except Exception:
        status = 'error'
        raise
    finally:
        observe_embedding_request(
            modality='text',
            provider=resolved_provider,
            status=status,
            duration_seconds=time.perf_counter() - started,
        )


def multimodal_text_embedding(
    text: str,
    provider_name: str | None = None,
) -> list[float]:
    """Generate a query embedding in the shared image/video space."""
    started = time.perf_counter()
    resolved_provider = provider_name or Config.default_embedding_provider
    status = 'ok'
    try:
        provider = get_provider(resolved_provider)
        vector = provider.encode_multimodal_text(text)
        return vector
    except Exception:
        status = 'error'
        raise
    finally:
        observe_embedding_request(
            modality='multimodal_text',
            provider=resolved_provider,
            status=status,
            duration_seconds=time.perf_counter() - started,
        )


def image_embedding_from_path(
    image_path: str,
    provider_name: str | None = None,
) -> List[float]:
    """Generate an embedding vector for an image from a file path.

    Args:
        image_path (str): Path to the image file to encode.
        provider_name (str | None): Optional provider name override.

    Returns:
        List[float]: A list of floats representing the image embedding.

    Notes:
        - The image is converted to RGB before encoding.
        - Uses a CLIP-based model ("clip-ViT-B-32") for generating visual embeddings.
        - Embeddings can be stored in Qdrant or compared with other image embeddings.
    """
    started = time.perf_counter()
    resolved_provider = provider_name or Config.default_embedding_provider
    status = 'ok'
    try:
        provider = get_provider(resolved_provider)
        with Image.open(image_path) as img:
            vector = provider.encode_image(img)
        return vector
    except Exception:
        status = 'error'
        raise
    finally:
        observe_embedding_request(
            modality='image',
            provider=resolved_provider,
            status=status,
            duration_seconds=time.perf_counter() - started,
        )


def video_embedding_from_path(
    video_path: str,
    sample_fps: float | None = None,
    provider_name: str | None = None,
) -> List[float]:
    """Generate an embedding vector for a video from a file path.

    Args:
        video_path (str): Path to the video file to encode.
        sample_fps (float | None): Sampling FPS. Falls back to config when None.
        provider_name (str | None): Optional provider name override.
    """
    started = time.perf_counter()
    resolved_provider = provider_name or Config.default_embedding_provider
    resolved_sample_fps = sample_fps or Config.embedding_video_sample_fps
    status = 'ok'
    try:
        provider = get_provider(resolved_provider)
        vector = provider.encode_video(
            video_path,
            sample_fps=resolved_sample_fps,
        )
        return vector
    except Exception:
        status = 'error'
        raise
    finally:
        observe_embedding_request(
            modality='video',
            provider=resolved_provider,
            status=status,
            duration_seconds=time.perf_counter() - started,
        )
