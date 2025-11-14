from sentence_transformers import SentenceTransformer

# Загружаем локальную модель для embeddings
# Можно заменить на любую модель из Hugging Face
model = SentenceTransformer("all-MiniLM-L6-v2")

def text_embedding(text: str) -> list[float]:
    """
    Генерирует embedding для текста локально.
    """
    vector = model.encode(text).tolist()  # преобразуем в список float для Qdrant
    return vector
