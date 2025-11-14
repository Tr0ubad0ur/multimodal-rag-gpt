from pathlib import Path
from qdrant_client import QdrantClient
from backend.core.embeddings import text_embedding
from backend.utils.loaders import load_documents
import logging

logger = logging.getLogger(__name__)

COLLECTION_NAME = "documents"
DATA_FOLDER = Path("data/test_data")  # папка с PDF/TXT
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Подключение к Qdrant
client = QdrantClient(host="localhost", port=6333)

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    """Split a text string into overlapping chunks.

    Args:
        text (str): The input text to split.
        size (int, optional): Maximum size of each chunk. Defaults to CHUNK_SIZE.
        overlap (int, optional): Number of overlapping characters between chunks. Defaults to CHUNK_OVERLAP.

    Returns:
        list[str]: A list of text chunks.

    Example:
        >>> chunk_text("abcdefgh", size=3, overlap=1)
        ['abc', 'cde', 'efg', 'gh']
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
    """Load a document and split it into text chunks.

    Args:
        file_path (Path): Path to the document file (txt, pdf, etc.).

    Returns:
        list[str]: List of text chunks extracted from the document.

    Notes:
        - Uses `load_documents` to read text from the file.
        - Returns an empty list if the document is empty or unsupported.
    """
    text = load_documents(file_path)
    if not text:
        return []
    return chunk_text(text)

def load_documents_to_qdrant():
    """Load all documents from DATA_FOLDER into a Qdrant collection as vectors.

    Reads all files recursively, splits them into chunks, converts each chunk
    to an embedding vector, and upserts it into the Qdrant collection.

    Uses:
        DATA_FOLDER (Path): Folder containing documents.
        COLLECTION_NAME (str): Qdrant collection name.
        client (QdrantClient): Qdrant client instance.
        text_embedding (Callable): Function to convert text to vector.

    Returns:
        None

    Notes:
        - Each chunk is stored with an incremental integer ID and payload containing
          the text and source file path.
        - Logs the number of chunks processed for each file.
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
            logger.info(f'✅ Загружено {len(chunks)} чанков из {file_path.name}')
        else:
            logger.info(f'⚠ Пропущен {file_path.name}')

if __name__ == "__main__":
    load_documents_to_qdrant()
    logger.info('Все документы загружены в Qdrant!')
