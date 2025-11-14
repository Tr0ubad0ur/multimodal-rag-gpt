from pathlib import Path
from qdrant_client import QdrantClient
from backend.core.embeddings import text_embedding
from backend.utils.loaders import load_documents

# ----------------------------
# Настройки
# ----------------------------
COLLECTION_NAME = "documents"
DATA_FOLDER = Path("data")  # папка с PDF/TXT
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Подключение к Qdrant
client = QdrantClient(host="localhost", port=6333)

# ----------------------------
# Функции
# ----------------------------
def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    """
    Разбивает текст на чанки с указанным размером и перекрытием.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start += size - overlap
    return chunks

def process_file(file_path: Path) -> list[str]:
    """
    Загружает файл и возвращает список текстовых чанков.
    """
    text = load_documents(file_path)
    if not text:
        return []
    return chunk_text(text)

def load_documents_to_qdrant():
    """
    Загружает все документы из папки DATA_FOLDER в коллекцию Qdrant.
    """
    all_files = list(DATA_FOLDER.glob("**/*"))
    point_id = 0

    for file_path in all_files:
        chunks = process_file(file_path)
        for chunk in chunks:
            vector = text_embedding(chunk)
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[{
                    "id": point_id,
                    "vector": vector,
                    "payload": {"text": chunk, "source": str(file_path)}
                }]
            )
            point_id += 1
        if chunks:
            print(f"✅ Загружено {len(chunks)} чанков из {file_path.name}")
        else:
            print(f"⚠ Пропущен {file_path.name} (нет текста)")

# ----------------------------
# Запуск скрипта
# ----------------------------
if __name__ == "__main__":
    load_documents_to_qdrant()
    print("🎉 Все документы загружены в Qdrant")
