from qdrant_client import QdrantClient
from qdrant_client.http import models
from backend.utils.config import settings
from typing import List, Dict, Any

def get_qdrant() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)

def ensure_collection_text():
    q = get_qdrant()
    try:
        q.get_collection(settings.qdrant_text_collection)
    except Exception:
        q.create_collection(
            collection_name=settings.qdrant_text_collection,
            vectors_config=models.VectorParams(size=settings.text_vector_size, distance=models.Distance.COSINE),
        )

def ensure_collection_image():
    q = get_qdrant()
    try:
        q.get_collection(settings.qdrant_image_collection)
    except Exception:
        q.create_collection(
            collection_name=settings.qdrant_image_collection,
            vectors_config=models.VectorParams(size=settings.image_vector_size, distance=models.Distance.COSINE),
        )

def add_text_points(points: List[Dict[str, Any]]):
    """
    points: list of {"id": str, "vector": List[float], "payload": {...}}
    """
    q = get_qdrant()
    ensure_collection_text()
    q.upsert(collection_name=settings.qdrant_text_collection, points=[models.PointStruct(**p) for p in points])

def add_image_point(point: Dict[str, Any]):
    q = get_qdrant()
    ensure_collection_image()
    q.upsert(collection_name=settings.qdrant_image_collection, points=[models.PointStruct(**point)])

def search_text(query_vector: List[float], limit: int = 5):
    q = get_qdrant()
    ensure_collection_text()
    hits = q.search(collection_name=settings.qdrant_text_collection, query_vector=query_vector, limit=limit)
    # Приводим к общему виду
    return [{"id": h.id, "score": h.score, "payload": h.payload} for h in hits]

def search_images(query_vector: List[float], limit: int = 5):
    q = get_qdrant()
    ensure_collection_image()
    hits = q.search(collection_name=settings.qdrant_image_collection, query_vector=query_vector, limit=limit)
    return [{"id": h.id, "score": h.score, "payload": h.payload} for h in hits]
