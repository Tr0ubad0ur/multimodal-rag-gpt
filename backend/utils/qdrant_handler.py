import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from backend.core.embeddings import image_embedding_from_path, text_embedding
from backend.utils.config_handler import Config
from backend.utils.load_data import DataLoader

logger = logging.getLogger(__name__)


class QdrantHandler:
    """Универсальный обработчик Qdrant коллекций для текста или изображений."""

    def __init__(
        self, url: str, collection_name: str, vector_size: int
    ) -> None:
        """Initialize QdrantHandler instance.

        Args:
        url (str): URL сервера Qdrant.
        collection_name (str): Имя коллекции.
        vector_size (int): Размерность векторов.
        """
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.vector_size = vector_size

    def create_collection(self) -> None:
        """Создает коллекцию, если она не существует."""
        try:
            self.client.get_collection(self.collection_name)
            logger.info(f'Коллекция {self.collection_name} уже существует')
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size, distance=Distance.COSINE
                ),
            )
            logger.info(f'Коллекция {self.collection_name} успешно создана')

    def enrich_with_data(
        self, folder: str = Config.data_folder, embed_type: str = 'text'
    ) -> None:
        """Load all data from a folder into the current Qdrant collection.

        Args:
            folder (str): Path to the folder with documents/images.
            embed_type (str): 'text' or 'image'.
        """
        self.create_collection()

        folder_path = Path(folder)
        if not folder_path.exists():
            logger.warning(
                f'Folder {folder} does not exist. Skipping data enrichment.'
            )
            return

        logger.info(
            f'Start uploading {embed_type} data from {folder} to collection {self.collection_name}...'
        )
        self.load_folder_to_qdrant(
            folder=folder_path,
            collection_name=self.collection_name,
            embed_type=embed_type,
        )
        logger.info(
            f'✅ Data enrichment completed for collection {self.collection_name}.'
        )

    def add_points(self, points: List[Dict[str, Any]]) -> None:
        """Добавляет точки в коллекцию.

        Args:
            points (List[Dict[str, Any]]): Список точек вида {"id": str, "vector": List[float], "payload": {...}}
        """
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(**p) for p in points],
        )

    def search(
        self, query_vector: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search nearest points by query vector.

        Note:
            This method only performs retrieval and does not trigger indexing.
            Data ingestion should be done explicitly via ``enrich_with_data``.
        """
        self.create_collection()

        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            score_threshold=Config.score_threshold,
        ).points

        results = []
        for hit in hits:
            results.append(
                {'id': hit.id, 'score': hit.score, 'payload': hit.payload}
            )

        return results

    def load_folder_to_qdrant(
        self,
        folder: Path,
        collection_name: str,
        embed_type: str = 'text',  # "text" or "image"
    ) -> None:
        """Load all documents or images from folder into Qdrant."""
        all_files = list(folder.glob('*'))

        for file_path in all_files:
            if (
                embed_type == 'text'
                and file_path.suffix.lower()
                in Config.text_extensions + Config.pdf_extensions
            ):
                loader = DataLoader()
                chunks = loader.process_file(file_path)
                for chunk_idx, chunk in enumerate(chunks):
                    vector = text_embedding(chunk)
                    self.client.upsert(
                        collection_name=collection_name,
                        points=[
                            {
                                'id': str(
                                    uuid.uuid5(
                                        uuid.NAMESPACE_URL,
                                        f'{file_path}:{chunk_idx}',
                                    )
                                ),
                                'vector': vector,
                                'payload': {
                                    'text': chunk,
                                    'source': str(file_path),
                                },
                            }
                        ],
                    )
                if chunks:
                    logger.info(
                        f'✅ Uploaded {len(chunks)} text chunks from {file_path.name}'
                    )
                else:
                    logger.info(f'⚠ Skipped empty text file {file_path.name}')

            elif (
                embed_type == 'image'
                and file_path.suffix.lower() in Config.image_extensions
            ):
                vector = image_embedding_from_path(str(file_path))
                self.client.upsert(
                    collection_name=collection_name,
                    points=[
                        {
                            'id': str(
                                uuid.uuid5(
                                    uuid.NAMESPACE_URL,
                                    f'{file_path}:image',
                                )
                            ),
                            'vector': vector,
                            'payload': {'source': str(file_path)},
                        }
                    ],
                )
                logger.info(f'✅ Uploaded image {file_path.name}')
