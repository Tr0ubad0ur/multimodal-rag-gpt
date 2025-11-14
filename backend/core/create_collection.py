# Создание коллекции для Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import logging

logger = logging.getLogger(__name__)

COLLECTION_NAME = "documents"
VECTOR_SIZE = 384
DISTANCE_METRIC = Distance.COSINE

def create_collection():
    """Create a Qdrant collection if it does not already exist.

    This function connects to a local Qdrant instance, checks if a collection
    with the specified name exists, and creates it with the defined vector size
    and distance metric if it does not.

    Uses:
        COLLECTION_NAME (str): Name of the collection to create.
        VECTOR_SIZE (int): Dimensionality of vectors to store.
        DISTANCE_METRIC (str): Distance metric for similarity search ('Cosine', 'Euclid', etc.).

    Returns:
        None
    """
    client = QdrantClient(host="localhost", port=6333)

    # Проверка, существует ли коллекция
    existing = client.get_collections().collections
    if COLLECTION_NAME in [c.name for c in existing]:
        logger.info(f'Коллекция "{COLLECTION_NAME}" уже существует.')
        return

    # Создание коллекции
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=DISTANCE_METRIC)
    )

    logger.info(f'Коллекция "{COLLECTION_NAME}" успешно создана!')

if __name__ == "__main__":
    create_collection()
