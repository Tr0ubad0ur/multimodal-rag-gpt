import asyncio
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from backend.api.endpoints import router
from backend.monitoring.metrics import observe_http_request
from backend.services.health_checks import check_dependencies
from backend.services.ingest_poller import IngestPoller
from backend.utils.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _normalize_error_code(detail: object, *, fallback: str) -> str:
    """Convert error detail into a stable snake_case error code."""
    if isinstance(detail, dict):
        raw = detail.get('error_code') or detail.get('code')
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        detail = detail.get('detail') or detail.get('message') or fallback
    text = str(detail or fallback).strip().lower()
    normalized = re.sub(r'[^a-z0-9]+', '_', text).strip('_')
    return normalized or fallback


def _format_validation_errors(
    errors: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return compact validation error details for frontend consumption."""
    formatted: list[dict[str, object]] = []
    for error in errors:
        formatted.append(
            {
                'field': '.'.join(str(part) for part in error.get('loc', [])),
                'message': str(error.get('msg', 'Invalid value')),
                'code': str(error.get('type', 'validation_error')),
            }
        )
    return formatted


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start and stop background services tied to app lifecycle."""
    poller_enabled = os.getenv(
        'INGEST_POLLER_ENABLED', 'false'
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
            'error_code': 'bad_request',
            'path': request.url.path,
            'error_type': 'ValueError',
        },
    )


@app.exception_handler(RequestValidationError)
def request_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a stable validation error contract for invalid request payloads."""
    return JSONResponse(
        status_code=422,
        content={
            'detail': 'Request validation failed',
            'error_code': 'validation_error',
            'path': request.url.path,
            'status_code': 422,
            'errors': _format_validation_errors(exc.errors()),
        },
    )


@app.exception_handler(HTTPException)
def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Return a stable JSON error contract for expected API errors."""
    detail = exc.detail
    payload: dict[str, object]
    if isinstance(detail, dict):
        payload = dict(detail)
        payload.setdefault(
            'detail',
            payload.get('message') or 'Request failed',
        )
    else:
        payload = {'detail': str(detail)}
    payload.setdefault(
        'error_code',
        _normalize_error_code(payload.get('detail'), fallback='http_error'),
    )
    payload['path'] = request.url.path
    payload['status_code'] = exc.status_code
    return JSONResponse(status_code=exc.status_code, content=payload)


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
            'error_code': 'internal_server_error',
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


@app.get('/health/live')
def health_live() -> Dict[str, str]:
    """Liveness probe."""
    return {'status': 'ok'}


@app.get('/health/ready', response_model=None)
def health_ready() -> JSONResponse:
    """Readiness probe with dependency checks."""
    deps = check_dependencies(mode='web')
    is_ready = all(bool(item.get('ok')) for item in deps.values())
    payload: Dict[str, object] = {
        'status': 'ready' if is_ready else 'not_ready',
        'dependencies': deps,
    }
    return JSONResponse(status_code=200 if is_ready else 503, content=payload)
