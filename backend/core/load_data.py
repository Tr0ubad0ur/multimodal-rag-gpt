import logging
from pathlib import Path
from typing import List

from qdrant_client import QdrantClient
from utils.config_handler import Config

from backend.core.embeddings import image_embedding_from_path, text_embedding

# Поддерживаемые форматы
TEXT_EXTENSIONS = Config.text_extensions
PDF_EXTENSIONS = Config.pdf_extensions
IMAGE_EXTENSIONS = Config.image_extensions

logger = logging.getLogger(__name__)


class DataLoader:
    """Unified loader for text documents and images with optional Qdrant upsert."""

    def __init__(
        self, qdrant_host: str = 'localhost', qdrant_port: int = 6333
    ) -> None:
        """Initialize the data loader with Qdrant connection parameters.

        Args:
            qdrant_host (str): Qdrant server host.
            qdrant_port (int): Qdrant server port.
        """
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)

    @staticmethod
    def chunk_text(
        text: str,
        size: int = Config.chunk_size,
        overlap: int = Config.chunk_overlap,
    ) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start += size - overlap
        return chunks

    @staticmethod
    def load_file(file_path: Path) -> str:
        """Load text from a single file (PDF, TXT, MD)."""
        text_parts = []
        if file_path.suffix.lower() in PDF_EXTENSIONS:
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            text_parts.extend([doc.page_content for doc in docs])
        elif file_path.suffix.lower() in TEXT_EXTENSIONS:
            from langchain_community.document_loaders import TextLoader

            loader = TextLoader(str(file_path))
            docs = loader.load()
            text_parts.extend([doc.page_content for doc in docs])
        else:
            logger.info(f'Skipped unsupported file {file_path.name}')
        return '\n'.join(text_parts)

    @staticmethod
    def list_images(folder: Path) -> List[Path]:
        """List all image paths in a folder."""
        return [
            p for p in folder.glob('*') if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

    def process_file(self, file_path: Path) -> List[str]:
        """Process a single text document into chunks."""
        text = self.load_file(file_path)
        if not text:
            return []
        return self.chunk_text(text)

    def load_folder_to_qdrant(
        self,
        folder: Path,
        collection_name: str,
        embed_type: str = 'text',  # "text" or "image"
    ) -> None:
        """Load all documents or images from folder into Qdrant."""
        all_files = list(folder.glob('*'))
        point_id = 0

        for file_path in all_files:
            if (
                embed_type == 'text'
                and file_path.suffix.lower()
                in TEXT_EXTENSIONS + PDF_EXTENSIONS
            ):
                chunks = self.process_file(file_path)
                for chunk in chunks:
                    vector = text_embedding(chunk)
                    self.client.upsert(
                        collection_name=collection_name,
                        points=[
                            {
                                'id': point_id,
                                'vector': vector,
                                'payload': {
                                    'text': chunk,
                                    'source': str(file_path),
                                },
                            }
                        ],
                    )
                    point_id += 1
                if chunks:
                    logger.info(
                        f'✅ Uploaded {len(chunks)} text chunks from {file_path.name}'
                    )
                else:
                    logger.info(f'⚠ Skipped empty text file {file_path.name}')

            elif (
                embed_type == 'image'
                and file_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                vector = image_embedding_from_path(str(file_path))
                self.client.upsert(
                    collection_name=collection_name,
                    points=[
                        {
                            'id': point_id,
                            'vector': vector,
                            'payload': {'source': str(file_path)},
                        }
                    ],
                )
                logger.info(f'✅ Uploaded image {file_path.name}')
                point_id += 1
