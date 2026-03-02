import logging
import os
import time
from typing import Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from backend.api.endpoints import router
from backend.monitoring.metrics import observe_http_request
from backend.utils.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


app = FastAPI(title='Multimodal RAG Backend', version='0.1')

default_origins = [
    'http://localhost:3000',
    'http://127.0.0.1:3000',
    'http://localhost:5173',
    'http://127.0.0.1:5173',
    'http://localhost:8080',
    'http://127.0.0.1:8080',
]
raw_origins = os.getenv('CORS_ALLOW_ORIGINS', '')
allowed_origins = (
    [origin.strip() for origin in raw_origins.split(',') if origin.strip()]
    if raw_origins
    else default_origins
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(router)


@app.middleware('http')
async def prometheus_http_metrics(request, call_next):
    """Collect basic HTTP metrics for Prometheus."""
    started = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - started
    observe_http_request(
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_seconds=duration,
    )
    return response


@app.get('/metrics')
def metrics() -> Response:
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(ValueError)
def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Convert ValueError into explicit 400 response."""
    return JSONResponse(
        status_code=400,
        content={
            'detail': str(exc),
            'path': request.url.path,
            'error_type': 'ValueError',
        },
    )


@app.exception_handler(Exception)
def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all handler for unexpected server errors."""
    logger.exception(
        'Unhandled exception at %s', request.url.path, exc_info=exc
    )
    return JSONResponse(
        status_code=500,
        content={
            'detail': 'Internal server error',
            'path': request.url.path,
            'error_type': exc.__class__.__name__,
        },
    )


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
