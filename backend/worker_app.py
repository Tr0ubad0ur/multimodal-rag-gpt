from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from backend.services.health_checks import check_dependencies
from backend.services.ingest_poller import IngestPoller
from backend.utils.log_config import setup_logging

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run ingest poller in worker API process lifecycle."""
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
            await task


app = FastAPI(title='Multimodal RAG Worker', version='0.1', lifespan=lifespan)


@app.get('/metrics')
def metrics() -> Response:
    """Prometheus scrape endpoint for worker process metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get('/health/live')
def health_live() -> Dict[str, str]:
    """Worker liveness probe."""
    return {'status': 'ok'}


@app.get('/health/ready', response_model=None)
def health_ready() -> JSONResponse:
    """Worker readiness probe with dependency checks."""
    deps = check_dependencies(mode='worker')
    is_ready = all(bool(item.get('ok')) for item in deps.values())
    payload: Dict[str, object] = {
        'status': 'ready' if is_ready else 'not_ready',
        'dependencies': deps,
    }
    return JSONResponse(status_code=200 if is_ready else 503, content=payload)
