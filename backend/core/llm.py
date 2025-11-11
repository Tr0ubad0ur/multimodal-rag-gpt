from langchain_openai import ChatOpenAI
from backend.utils.config import settings

def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        openai_api_key=settings.openai_api_key
    )
