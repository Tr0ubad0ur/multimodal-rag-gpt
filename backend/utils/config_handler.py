import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'backend_config.yaml'

load_dotenv()

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    _config = yaml.safe_load(f)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {'1', 'true', 'yes', 'on'}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


class Config:
    """Application settings loaded from backend_config.yaml."""

    qdrant_text_collection: str = _config['qdrant']['text_collection']
    qdrant_image_collection: str = _config['qdrant']['image_collection']
    qdrant_video_collection: str = _config['qdrant']['video_collection']
    qdrant_url: str = os.getenv('QDRANT_URL', 'http://localhost:6333')
    text_vector_size: int = _config['qdrant']['text_vector_size']
    image_vector_size: int = _config['qdrant']['image_vector_size']
    video_vector_size: int = _config['qdrant']['video_vector_size']
    score_threshold: float = _config['qdrant']['score_threshold']

    data_folder: str = _config['data']['data_folder']
    chunk_size: int = _config['data']['chunk_size']
    chunk_overlap: int = _config['data']['chunk_overlap']
    text_extensions: list = _config['data']['supported_text_extensions']
    pdf_extensions: list = _config['data']['supported_pdf_extensions']
    image_extensions: list = _config['data']['supported_image_extensions']
    video_extensions: list = _config['data']['supported_video_extensions']

    llm_model_name: str = os.getenv(
        'LLM_MODEL_NAME', _config['llm']['model_name']
    )
    llm_max_new_tokens: int = _env_int(
        'LLM_MAX_NEW_TOKENS', _config['llm']['max_new_tokens']
    )
    llm_available_models: list[str] = [
        model.strip()
        for model in os.getenv('LLM_AVAILABLE_MODELS', llm_model_name).split(
            ','
        )
        if model.strip()
    ]
    rag_max_context_docs: int = _env_int('RAG_MAX_CONTEXT_DOCS', 4)
    rag_max_context_chars: int = _env_int('RAG_MAX_CONTEXT_CHARS', 2400)
    llm_fast_mode: bool = _env_bool('LLM_FAST_MODE', False)

    default_embedding_provider: str = _config['embeddings']['default_provider']
    embedding_video_sample_fps: float = _env_float(
        'EMBEDDING_VIDEO_SAMPLE_FPS',
        _config['embeddings']['video_sample_fps'],
    )
    embedding_providers: dict = _config['embeddings']['providers']
    ingest_enable_ocr: bool = _env_bool('INGEST_ENABLE_OCR', True)
    ingest_ocr_timeout_seconds: int = _env_int('INGEST_OCR_TIMEOUT_SECONDS', 6)

    log_dir: str = _config['logging']['log_dir']
    log_file: str = _config['logging']['log_file']
    max_bytes: int = _config['logging']['max_bytes']
    backup_count: int = _config['logging']['backup_count']

    supabase_url: str | None = os.getenv('SUPABASE_URL')
    supabase_anon_key: str | None = os.getenv('SUPABASE_ANON_KEY')
    supabase_service_role_key: str | None = os.getenv(
        'SUPABASE_SERVICE_ROLE_KEY'
    )


settings = Config()
