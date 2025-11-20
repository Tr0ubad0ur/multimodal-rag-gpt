import logging
from pathlib import Path
from typing import List

from backend.utils.config_handler import Config

logger = logging.getLogger(__name__)


class DataLoader:
    """Unified loader for text documents and images with optional Qdrant upsert."""

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
        if file_path.suffix.lower() in Config.pdf_extensions:
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            text_parts.extend([doc.page_content for doc in docs])
        elif file_path.suffix.lower() in Config.text_extensions:
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
            p
            for p in folder.glob('*')
            if p.suffix.lower() in Config.image_extensions
        ]

    def process_file(self, file_path: Path) -> List[str]:
        """Process a single text document into chunks."""
        text = self.load_file(file_path)
        if not text:
            return []
        return self.chunk_text(text)
