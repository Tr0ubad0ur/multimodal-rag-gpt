import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'backend_config.yaml'

load_dotenv()

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    _config = yaml.safe_load(f)


class Config:
    """Application settings loaded from backend_config.yaml."""

    qdrant_text_collection: str = _config['qdrant']['text_collection']
    qdrant_image_collection: str = _config['qdrant']['image_collection']
    text_vector_size: int = _config['qdrant']['text_vector_size']
    image_vector_size: int = _config['qdrant']['image_vector_size']
    score_threshold: float = _config['qdrant']['score_threshold']

    data_folder: str = _config['data']['data_folder']
    chunk_size: int = _config['data']['chunk_size']
    chunk_overlap: int = _config['data']['chunk_overlap']
    text_extensions: list = _config['data']['supported_text_extensions']
    pdf_extensions: list = _config['data']['supported_pdf_extensions']
    image_extensions: list = _config['data']['supported_image_extensions']

    llm_model_name: str = _config['llm']['model_name']
    llm_max_new_tokens: int = _config['llm']['max_new_tokens']

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
