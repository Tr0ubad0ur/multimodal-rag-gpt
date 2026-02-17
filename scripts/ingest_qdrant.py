import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.utils.config_handler import Config  # noqa: E402
from backend.utils.qdrant_handler import QdrantHandler  # noqa: E402


def main() -> None:
    """Function to load data to qdrant container."""
    url = os.getenv('QDRANT_URL', 'localhost:6333')
    data_folder = os.getenv('DATA_FOLDER', Config.data_folder)
    embed_type = os.getenv('EMBED_TYPE', 'text')

    if embed_type == 'image':
        collection_name = Config.qdrant_image_collection
        vector_size = Config.image_vector_size
    else:
        collection_name = Config.qdrant_text_collection
        vector_size = Config.text_vector_size

    qdrant = QdrantHandler(
        url=url,
        collection_name=collection_name,
        vector_size=vector_size,
    )
    qdrant.enrich_with_data(folder=data_folder, embed_type=embed_type)


if __name__ == '__main__':
    main()
