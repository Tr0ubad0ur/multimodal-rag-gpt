from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import logging

logger = logging.getLogger(__name__)

# Поддерживаемые форматы
TEXT_EXTENSIONS = [".txt", ".md"]
PDF_EXTENSIONS = [".pdf"]

def load_documents(path: Path) -> str:
    """Load all PDF and text/Markdown files from a folder and return combined text.

    Args:
        path (Path): Path to the folder containing documents.

    Returns:
        str: A single string with the concatenated content of all loaded documents.

    Notes:
        - Supported formats: PDF, TXT, MD.
        - Unsupported files are skipped with an info log.
        - Each document's content is extracted page by page (for PDFs) or line by line (for text files).

    Example:
        >>> load_documents(Path("./data"))
        "Text from document1\nText from document2\n..."
    """
    docs_text = []

    for file in path.glob("*"):
        if file.suffix.lower() in PDF_EXTENSIONS:
            loader = PyPDFLoader(str(file))
            docs = loader.load()
            docs_text.extend([doc.page_content for doc in docs])
        elif file.suffix.lower() in TEXT_EXTENSIONS:
            loader = TextLoader(str(file))
            docs = loader.load()
            docs_text.extend([doc.page_content for doc in docs])
        else:
            logger.info(f'Пропущен файл {file.name} (не поддерживаемый формат)')

    # Объединяем все тексты в одну строку
    return "\n".join(docs_text)
