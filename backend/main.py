import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from backend.api.endpoints import router
from backend.monitoring.metrics import observe_http_request
from backend.services.ingest_poller import IngestPoller
from backend.utils.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start and stop background services tied to app lifecycle."""
    poller_enabled = os.getenv(
        'INGEST_POLLER_ENABLED', 'true'
    ).strip().lower() not in {'0', 'false', 'no', 'off'}
    if not poller_enabled:
        yield
        return

    interval = int(os.getenv('INGEST_POLLER_INTERVAL_SECONDS', '5'))
    batch_size = int(os.getenv('INGEST_POLLER_BATCH_SIZE', '10'))
    stale_seconds = int(os.getenv('INGEST_POLLER_STALE_SECONDS', '300'))
    max_concurrency = int(os.getenv('INGEST_WORKER_MAX_CONCURRENCY', '4'))
    poller = IngestPoller(
        interval_seconds=max(1, interval),
        batch_size=max(1, batch_size),
        stale_seconds=max(30, stale_seconds),
        max_concurrency=max(1, max_concurrency),
    )
    app.state.ingest_poller = poller
    app.state.ingest_poller_task = asyncio.create_task(poller.run())
    try:
        yield
    finally:
        task = getattr(app.state, 'ingest_poller_task', None)
        poller = getattr(app.state, 'ingest_poller', None)
        if poller is not None:
            poller.stop()
        if task is not None:
            try:
                await task
            except Exception:
                logger.exception('Failed to stop ingest poller cleanly')


app = FastAPI(title='Multimodal RAG Backend', version='0.1', lifespan=lifespan)

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
