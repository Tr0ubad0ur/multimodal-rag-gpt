# Делаем ембеддинги
from sentence_transformers import SentenceTransformer
from PIL import Image

# Загружаем локальную модель для embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

def text_embedding(text: str) -> list[float]:
    """Generate an embedding vector for a given text string.

    Args:
        text (str): The input text to encode.

    Returns:
        List[float]: A list of floats representing the text embedding.

    Notes:
        - This embedding can be stored in a vector database like Qdrant.
        - Ensure the model used for text embeddings is compatible with your retrieval pipeline.
    """
    vector = model.encode(text).tolist()  # преобразуем в список float для Qdrant
    return vector

#TODO: check this
_model = SentenceTransformer("clip-ViT-B-32")
def image_embedding_from_path(image_path: str):
    """Generate an embedding vector for an image from a file path.

    Args:
        image_path (str): Path to the image file to encode.

    Returns:
        List[float]: A list of floats representing the image embedding.

    Notes:
        - The image is converted to RGB before encoding.
        - Uses a CLIP-based model ("clip-ViT-B-32") for generating visual embeddings.
        - Embeddings can be stored in Qdrant or compared with other image embeddings.
    """
    img = Image.open(image_path).convert("RGB")
    vector = _model.encode(img, convert_to_numpy=True).tolist()
    return vector
