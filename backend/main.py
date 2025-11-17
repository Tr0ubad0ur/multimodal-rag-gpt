from fastapi import FastAPI
from backend.api.endpoints import router
#TODO transfer this to another place (maybe)
import logging
from utils.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
#

app = FastAPI(
    title="Multimodal RAG Backend",
    version="0.1"
)

app.include_router(router)

@app.get("/")
def root():
    """Health check endpoint.

    Returns:
        dict: A simple JSON message confirming that the backend is running.

    Example response:
        {
            "message": "Multimodal RAG backend is running!!!"
        }
    """
    return {"message": "Multimodal RAG backend is running!!!"}
