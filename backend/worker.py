from __future__ import annotations

import asyncio
import logging
import os

from backend.services.ingest_poller import IngestPoller
from backend.utils.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


async def _run_worker() -> None:
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

    try:
        await poller.run()
    except asyncio.CancelledError:
        poller.stop()
        raise


def main() -> None:
    """Run ingest worker process."""
    logger.info('Starting dedicated ingest worker process')
    try:
        asyncio.run(_run_worker())
    except KeyboardInterrupt:
        logger.info('Ingest worker stopped by keyboard interrupt')


if __name__ == '__main__':
    main()
