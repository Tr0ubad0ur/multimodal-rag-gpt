from __future__ import annotations

import asyncio
import logging
import os
import uuid

from backend.services.guest_cleanup import GuestCleanupService
from backend.services.ingest_jobs import IngestJobsService
from backend.services.ingest_worker import IngestWorker

logger = logging.getLogger(__name__)


class IngestPoller:
    """Periodic poller that processes queued ingest jobs."""

    def __init__(
        self,
        *,
        interval_seconds: int = 5,
        batch_size: int = 10,
        stale_seconds: int = 300,
        max_concurrency: int = 4,
    ) -> None:
        """Configure polling cadence, claim batch size and concurrency."""
        self.interval_seconds = interval_seconds
        self.batch_size = batch_size
        self.stale_seconds = stale_seconds
        self.max_concurrency = max(1, max_concurrency)
        self.lock_seconds = max(30, stale_seconds)
        self.worker_id = f'ingest-poller-{uuid.uuid4()}'
        self.jobs = IngestJobsService()
        self.worker = IngestWorker(max_attempts=3)
        self.guest_cleanup = GuestCleanupService()
        self.guest_ttl_hours = max(
            1, int(os.getenv('GUEST_SESSION_TTL_HOURS', '24'))
        )
        self.guest_cleanup_interval_seconds = max(
            60, int(os.getenv('GUEST_CLEANUP_INTERVAL_SECONDS', '3600'))
        )
        self._last_guest_cleanup_monotonic = 0.0
        self._stop_event = asyncio.Event()

    async def run(self) -> None:
        """Start polling loop until stop() is called."""
        logger.info(
            'IngestPoller started (interval=%ss, batch=%s, stale=%ss)',
            self.interval_seconds,
            self.batch_size,
            self.stale_seconds,
        )
        while not self._stop_event.is_set():
            try:
                revived = self.jobs.requeue_stale_processing(
                    stale_seconds=self.stale_seconds
                )
                if revived:
                    logger.info('Re-queued %s stale ingest jobs', revived)

                claimed = self.jobs.claim_jobs(
                    worker_id=self.worker_id,
                    batch_size=self.batch_size,
                    lock_seconds=self.lock_seconds,
                )
                await self._process_claimed_jobs(claimed)
                await self._run_guest_cleanup_if_due()
                self.jobs.refresh_depth_metrics()
            except Exception:
                logger.exception('IngestPoller loop failed')

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.interval_seconds,
                )
            except TimeoutError:
                continue
        logger.info('IngestPoller stopped')

    def stop(self) -> None:
        """Request graceful stop of polling loop."""
        self._stop_event.set()

    async def _process_claimed_jobs(self, claimed: list[dict]) -> None:
        if not claimed:
            return
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _run_one(job: dict) -> None:
            job_id = str(job.get('id') or '')
            if not job_id:
                return
            job_user_id = job.get('user_id')
            async with semaphore:
                try:
                    await asyncio.to_thread(
                        self.worker.process_job,
                        job_id=job_id,
                        user_id=str(job_user_id) if job_user_id else None,
                    )
                except Exception:
                    logger.exception('Failed processing ingest job %s', job_id)

        await asyncio.gather(*(_run_one(job) for job in claimed))

    async def _run_guest_cleanup_if_due(self) -> None:
        now = asyncio.get_running_loop().time()
        if (
            now - self._last_guest_cleanup_monotonic
            < self.guest_cleanup_interval_seconds
        ):
            return
        self._last_guest_cleanup_monotonic = now
        try:
            report = await asyncio.to_thread(
                self.guest_cleanup.cleanup_expired,
                ttl_hours=self.guest_ttl_hours,
            )
            deleted_total = sum(int(value) for value in report.values())
            if deleted_total:
                logger.info('Guest TTL cleanup removed artifacts: %s', report)
        except Exception:
            logger.exception('Guest TTL cleanup failed')
