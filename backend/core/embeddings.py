from langchain.embeddings import OpenAIEmbeddings
from backend.utils.config import settings

def text_embedding(text: str):
    emb = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=settings.openai_api_key)
    v = emb.embed_query(text)
    return v
