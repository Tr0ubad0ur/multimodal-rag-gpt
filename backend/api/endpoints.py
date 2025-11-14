from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
from io import BytesIO
from backend.core.multimodal_rag import multimodal_rag

router = APIRouter()


@router.post("/ask_text")
async def ask_text(question: str):
    """
    pure text RAG
    """
    result = multimodal_rag(question)
    return result


@router.post("/ask_mixed")
async def ask_mixed(
    question: str = Form(...),
    image: UploadFile = File(None)
):
    """
    multimodal RAG: question + optional image
    """
    img = None
    if image:
        content = await image.read()
        img = Image.open(BytesIO(content)).convert("RGB")

    result = multimodal_rag(question, image=img)
    return result
