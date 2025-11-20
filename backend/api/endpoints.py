from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from backend.core.multimodal_rag import LocalRAG

router = APIRouter()
rag = LocalRAG()


class QueryRequest(BaseModel):
    """Schema for user query requests.

    Attributes:
        query (str): The text query from the user.
        top_k (int, optional): Number of top documents to retrieve from RAG. Defaults to 5.
        image (Optional[str], optional): Optional path or URL to an image to include in the query. Defaults to None.
    """

    query: str
    top_k: int = 5
    image: Optional[str] = None


@router.post('/ask')
def ask_mixed(request: QueryRequest) -> dict:
    """Handle a multimodal query request (text + optional image) and generate an answer using RAG.

    Args:
        request (QueryRequest): The query request containing the text, optional image, and retrieval parameters.

    Returns:
        dict: The generated answer from the RAG pipeline, potentially considering the image.
    """
    result = rag.generate_answer(
        request.query, top_k=request.top_k, image=request.image
    )
    return result


# TODO write this requests
# @router.post('/test_llm')
# @router.post('/test_qdrant')
