from qdrant_client import QdrantClient
from qdrant_client.http import models
from backend.utils.config import settings
from typing import List, Dict, Any

def get_qdrant() -> QdrantClient:
    """Return a Qdrant client instance connected to the configured URL.

    Returns:
        QdrantClient: Connected Qdrant client.
    """
    return QdrantClient(url=settings.qdrant_url)

def ensure_collection_text():
    """Ensure that the text collection exists in Qdrant.

    If the collection does not exist, it is created with the configured
    vector size and cosine distance.
    """
    q = get_qdrant()
    try:
        q.get_collection(settings.qdrant_text_collection)
    except Exception:
        q.create_collection(
            collection_name=settings.qdrant_text_collection,
            vectors_config=models.VectorParams(size=settings.text_vector_size, distance=models.Distance.COSINE),
        )

def ensure_collection_image():
    """Ensure that the image collection exists in Qdrant.

    If the collection does not exist, it is created with the configured
    vector size and cosine distance.
    """
    q = get_qdrant()
    try:
        q.get_collection(settings.qdrant_image_collection)
    except Exception:
        q.create_collection(
            collection_name=settings.qdrant_image_collection,
            vectors_config=models.VectorParams(size=settings.image_vector_size, distance=models.Distance.COSINE),
        )

def add_text_points(points: List[Dict[str, Any]]):
    """Add multiple text points to the text collection in Qdrant.

    Args:
        points (List[Dict[str, Any]]): List of points, each containing:
            - "id" (str): Unique identifier for the point.
            - "vector" (List[float]): Embedding vector.
            - "payload" (dict): Additional metadata.
    """
    q = get_qdrant()
    ensure_collection_text()
    q.upsert(collection_name=settings.qdrant_text_collection, points=[models.PointStruct(**p) for p in points])

def add_image_point(point: Dict[str, Any]):
    """Add a single image point to the image collection in Qdrant.

    Args:
        point (Dict[str, Any]): A point dictionary containing:
            - "id" (str): Unique identifier.
            - "vector" (List[float]): Embedding vector for the image.
            - "payload" (dict): Optional metadata.
    """
    q = get_qdrant()
    ensure_collection_image()
    q.upsert(collection_name=settings.qdrant_image_collection, points=[models.PointStruct(**point)])

def search_text(query_vector: List[float], limit: int = 5):
    """Search the text collection in Qdrant using a query vector.

    Args:
        query_vector (List[float]): Embedding vector of the query.
        limit (int, optional): Number of top results to return. Defaults to 5.

    Returns:
        List[Dict[str, Any]]: List of search hits, each containing:
            - "id" (str): Point ID.
            - "score" (float): Similarity score.
            - "payload" (dict): Stored metadata.
    """
    q = get_qdrant()
    ensure_collection_text()
    hits = q.search(collection_name=settings.qdrant_text_collection, query_vector=query_vector, limit=limit)
    # Приводим к общему виду
    return [{"id": h.id, "score": h.score, "payload": h.payload} for h in hits]

def search_images(query_vector: List[float], limit: int = 5):
    """Search the image collection in Qdrant using a query vector.

    Args:
        query_vector (List[float]): Embedding vector of the query image.
        limit (int, optional): Number of top results to return. Defaults to 5.

    Returns:
        List[Dict[str, Any]]: List of search hits, each containing:
            - "id" (str): Point ID.
            - "score" (float): Similarity score.
            - "payload" (dict): Stored metadata.
    """
    q = get_qdrant()
    ensure_collection_image()
    hits = q.search(collection_name=settings.qdrant_image_collection, query_vector=query_vector, limit=limit)
    return [{"id": h.id, "score": h.score, "payload": h.payload} for h in hits]
