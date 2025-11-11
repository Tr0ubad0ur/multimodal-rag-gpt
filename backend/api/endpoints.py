from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
import os
import uuid

from backend.utils.loaders import load_documents
from backend.core.embeddings import text_embedding
from backend.core.image_embeddings import image_embedding_from_path
from backend.core import vectordb
from backend.core.multimodal_rag import ask_text_query, ask_image_query

router = APIRouter()

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


# Модели данных
class Query(BaseModel):
    question: str


# Вспомогательные функции
def create_text_points(docs):
    """Формирует список точек для вставки в Qdrant из документов LangChain"""
    points = []
    for i, d in enumerate(docs):
        vector = text_embedding(d.page_content)
        point = {
            "id": str(uuid.uuid4()),
            "vector": vector,
            "payload": {
                "type": "text",
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page", 0),
                "chunk_id": i,
                "text": d.page_content,
            },
        }
        points.append(point)
    return points

# Эндпоинты
@router.get("/ping")
async def ping():
    """Проверка состояния backend"""
    return {"status": "ok", "message": "Multimodal RAG backend is running 🚀"}


# Индексация текстов (PDF / TXT)
@router.post("/load_texts")
async def load_texts():
    """
    Загружает все PDF и TXT из папки data/, разбивает на фрагменты,
    создает эмбеддинги и добавляет в векторную базу (Qdrant)
    """
    docs = load_documents(DATA_DIR)
    points = create_text_points(docs)
    vectordb.add_text_points(points)
    return {"message": f"Загружено {len(points)} текстовых фрагментов."}


# Добавление изображений
@router.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """
    Загружает изображение и добавляет его эмбеддинг в базу
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, file.filename)

    with open(path, "wb") as f:
        f.write(await file.read())

    vector = image_embedding_from_path(path)
    point = {
        "id": str(uuid.uuid4()),
        "vector": vector,
        "payload": {
            "type": "image",
            "path": path,
            "caption": "",
        },
    }
    vectordb.add_image_point(point)
    return {"message": f"Изображение {file.filename} добавлено в базу."}


# Мультимодальный поиск
@router.post("/ask_mixed")
async def ask_mixed(question: str = Form(None), image: UploadFile = File(None)):
    """
    Универсальный эндпоинт:
    - если передан только question — текстовый RAG;
    - если передано изображение — мультимодальный поиск;
    - если и то, и другое — GPT комбинирует контекст.
    """
    if image is not None:
        os.makedirs("temp", exist_ok=True)
        temp_path = os.path.join("temp", image.filename)

        with open(temp_path, "wb") as f:
            f.write(await image.read())

        try:
            result = ask_image_query(temp_path, question)
        finally:
            os.remove(temp_path)

        return result

    elif question:
        return ask_text_query(question)

    else:
        return {"error": "Нужно передать либо question, либо image."}
