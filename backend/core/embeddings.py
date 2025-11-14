# Делаем ембеддинги
from sentence_transformers import SentenceTransformer
from PIL import Image

# Загружаем локальную модель для embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

def text_embedding(text: str) -> list[float]:
    """
    Генерирует embedding для текста.
    """
    vector = model.encode(text).tolist()  # преобразуем в список float для Qdrant
    return vector

#TODO: check this
_model = SentenceTransformer("clip-ViT-B-32")
def image_embedding_from_path(image_path: str):
    """
    Генерирует embedding для изображений.
    """
    img = Image.open(image_path).convert("RGB")
    vector = _model.encode(img, convert_to_numpy=True).tolist()
    return vector
