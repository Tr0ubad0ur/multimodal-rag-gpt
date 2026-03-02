from __future__ import annotations

import re
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import pdfplumber
from PIL import Image
from qdrant_client.models import Distance, PointStruct, VectorParams

from backend.core.embeddings import text_embedding
from backend.utils.config_handler import Config
from backend.utils.load_data import DataLoader
from backend.utils.qdrant_handler import QdrantHandler

TEXT_MIME_TYPES = {
    'text/plain',
    'text/markdown',
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
}
IMAGE_MIME_TYPES = {'image/jpeg', 'image/png', 'image/webp'}
VIDEO_MIME_TYPES = {
    'video/mp4',
    'video/quicktime',
    'video/x-msvideo',
    'video/x-matroska',
}
ROOT_SCOPE = 'root'


class IngestService:
    """Unified ingest pipeline into a single text-vector collection."""

    def __init__(self) -> None:
        """Initialize Qdrant text handler used by retrieval pipeline."""
        self.text_client = QdrantHandler(
            url=Config.qdrant_url,
            collection_name=Config.qdrant_text_collection,
            vector_size=Config.text_vector_size,
        )
        self.loader = DataLoader()

    def extract_attachment_context(
        self, file_path: str, mime: str
    ) -> list[str]:
        """Extract contextual text from different attachment modalities."""
        path = Path(file_path)
        if mime in TEXT_MIME_TYPES:
            return self._extract_text_chunks(path, mime)
        if mime in IMAGE_MIME_TYPES:
            return self._extract_image_chunks(path)
        if mime in VIDEO_MIME_TYPES:
            return self._extract_video_chunks(path)
        return []

    def ingest_file(
        self,
        *,
        file_id: str,
        file_path: str,
        filename: str,
        mime: str,
        user_id: str,
        folder_id: str | None = None,
        folder_name: str | None = None,
    ) -> None:
        """Extract chunks, embed them and upsert to Qdrant with metadata."""
        self.text_client.create_collection()
        path = Path(file_path)

        if mime in TEXT_MIME_TYPES:
            chunks = self._extract_text_chunks(path, mime)
        elif mime in IMAGE_MIME_TYPES:
            chunks = self._extract_image_chunks(path)
        elif mime in VIDEO_MIME_TYPES:
            chunks = self._extract_video_chunks(path)
        else:
            chunks = []

        self._upsert_text_chunks(
            chunks=chunks,
            file_id=file_id,
            source=str(path),
            filename=filename,
            mime=mime,
            user_id=user_id,
            folder_id=folder_id,
            folder_name=folder_name,
        )

    def _extract_text_chunks(self, path: Path, mime: str) -> list[str]:
        if mime == 'application/pdf':
            with pdfplumber.open(path) as pdf:
                text = '\n'.join(
                    (page.extract_text() or '').strip() for page in pdf.pages
                ).strip()
            return self.loader.chunk_text(text) if text else []

        if (
            mime
            == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        ):
            text = self._read_docx(path)
            return self.loader.chunk_text(text) if text else []

        text = path.read_text(encoding='utf-8', errors='ignore').strip()
        return self.loader.chunk_text(text) if text else []

    def _extract_image_chunks(self, path: Path) -> list[str]:
        try:
            import pytesseract

            with Image.open(path) as image:
                extracted = pytesseract.image_to_string(image).strip()
            if extracted:
                return self.loader.chunk_text(extracted)
        except Exception:
            pass
        return [f'Image attachment content from file: {path.name}']

    def _extract_video_chunks(self, path: Path) -> list[str]:
        return [f'Video attachment content from file: {path.name}']

    def _read_docx(self, path: Path) -> str:
        try:
            with ZipFile(path) as zipped:
                xml_content = zipped.read('word/document.xml')
            root = ET.fromstring(xml_content)
            texts = [
                node.text.strip()
                for node in root.iter()
                if node.tag.endswith('}t') and node.text
            ]
            return '\n'.join(texts)
        except Exception:
            return ''

    def _upsert_text_chunks(
        self,
        *,
        chunks: list[str],
        file_id: str,
        source: str,
        filename: str,
        mime: str,
        user_id: str,
        folder_id: str | None,
        folder_name: str | None,
    ) -> None:
        if not chunks:
            return

        folder_scope = folder_id or ROOT_SCOPE
        points: list[PointStruct] = []
        for idx, chunk in enumerate(chunks):
            text = chunk.strip()
            if not text:
                continue
            vector = text_embedding(text)
            if len(vector) != Config.text_vector_size:
                raise ValueError(
                    f'Embedding dimension mismatch: expected {Config.text_vector_size}, got {len(vector)}'
                )
            points.append(
                PointStruct(
                    id=str(
                        uuid.uuid5(uuid.NAMESPACE_URL, f'{file_id}:text:{idx}')
                    ),
                    vector=vector,
                    payload={
                        'text': text,
                        'source': source,
                        'filename': filename,
                        'mime': mime,
                        'user_id': user_id,
                        'folder_id': folder_id,
                        'folder_scope': folder_scope,
                        'file_id': file_id,
                        'modality': 'text',
                    },
                )
            )

        if points:
            self.text_client.client.upsert(
                collection_name=self.text_client.collection_name,
                points=points,
            )
            scoped_collection = self._scoped_collection_name(
                user_id=user_id,
                folder_id=folder_id,
                folder_name=folder_name,
            )
            self._ensure_collection(scoped_collection)
            self.text_client.client.upsert(
                collection_name=scoped_collection,
                points=points,
            )

    def _ensure_collection(self, collection_name: str) -> None:
        try:
            self.text_client.client.get_collection(
                collection_name=collection_name
            )
        except Exception:
            self.text_client.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=Config.text_vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def _scoped_collection_name(
        self,
        *,
        user_id: str,
        folder_id: str | None,
        folder_name: str | None,
    ) -> str:
        raw_scope = folder_name if folder_name else ROOT_SCOPE
        normalized_scope = re.sub(r'[^a-zA-Z0-9_]+', '_', raw_scope).strip('_')
        if not normalized_scope:
            normalized_scope = ROOT_SCOPE
        short_user = user_id.replace('-', '')[:8]
        short_folder = (folder_id or ROOT_SCOPE).replace('-', '')[:8]
        return f'kb_{short_user}_{normalized_scope}_{short_folder}'
