from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

def load_documents(path: str):
    docs = []
    for file in Path(path).glob("*"):
        if file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file))
            docs.extend(loader.load())
        elif file.suffix.lower() in [".txt", ".md"]:
            loader = TextLoader(str(file))
            docs.extend(loader.load())
    return split_documents(docs)

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

def list_images(path: str):
    """Собираем пути всех изображений"""
    return [str(p) for p in Path(path).glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
