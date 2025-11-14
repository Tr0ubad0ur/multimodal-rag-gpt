from fastapi import APIRouter
from pydantic import BaseModel
from backend.core.multimodal_rag import LocalRAG
from typing import Optional

router = APIRouter()
rag = LocalRAG()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    image: Optional[str] = None  # путь или URL

@router.post("/ask_text")
def ask_text(request: QueryRequest):
    result = rag.generate_answer(request.query, top_k=request.top_k)
    return result

@router.post("/ask_mixed")
def ask_mixed(request: QueryRequest):
    result = rag.generate_answer(request.query, top_k=request.top_k, image=request.image)
    return result
