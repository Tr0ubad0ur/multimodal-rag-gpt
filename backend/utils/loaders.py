from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Поддерживаемые форматы
TEXT_EXTENSIONS = [".txt", ".md"]
PDF_EXTENSIONS = [".pdf"]

def load_documents(path: Path) -> str:
    """
    Загружает все PDF и TXT/MD файлы из папки path и возвращает объединённый текст.
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
            print(f"Пропущен файл {file.name} (не поддерживаемый формат)")

    # Объединяем все тексты в одну строку
    return "\n".join(docs_text)
