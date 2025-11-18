import logging
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

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
        self.ensure_collection()

    def ensure_collection(self) -> None:
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
        """Поиск ближайших точек по вектору.

        Args:
            query_vector (List[float]): Вектор запроса.
            top_k (int): Количество результатов.

        Returns:
            List[Dict]: Результаты поиска с id, score, payload.
        """
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )
        return [
            {'id': h.id, 'score': h.score, 'payload': h.payload} for h in hits
        ]
