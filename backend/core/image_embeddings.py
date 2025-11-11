from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np

_model = SentenceTransformer("clip-ViT-B-32")  # Может быть локальный alias
def image_embedding_from_path(image_path: str):
    img = Image.open(image_path).convert("RGB")
    emb = _model.encode(img, convert_to_numpy=True)
    return emb.tolist()
