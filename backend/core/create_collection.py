from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Настройки
COLLECTION_NAME = "documents"       # Имя коллекции
VECTOR_SIZE = 384                   # Размер векторов (для all-MiniLM-L6-v2)
DISTANCE_METRIC = Distance.COSINE   # Косинусная мера расстояния

def create_collection():
    client = QdrantClient(host="localhost", port=6333)

    # Проверка, существует ли коллекция
    existing = client.get_collections().collections
    if COLLECTION_NAME in [c.name for c in existing]:
        print(f"Коллекция '{COLLECTION_NAME}' уже существует.")
        return

    # Создание коллекции
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=DISTANCE_METRIC)
    )

    print(f"Коллекция '{COLLECTION_NAME}' успешно создана!")

if __name__ == "__main__":
    create_collection()
