import logging
from typing import Dict

from fastapi import FastAPI

from backend.api.endpoints import router
from backend.utils.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


app = FastAPI(title='Multimodal RAG Backend', version='0.1')

app.include_router(router)


@app.get('/')
def root() -> Dict[str, str]:
    """Health check endpoint.

    Returns:
        dict: A simple JSON message confirming that the backend is running.

    Example response:
        {
            "message": "Multimodal RAG backend is running!!!"
        }
    """
    return {'message': 'Multimodal RAG backend is running!!!'}
