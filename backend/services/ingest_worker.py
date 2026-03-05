from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import HTTPException

from backend.services.ingest import IngestService
from backend.services.ingest_jobs import IngestJobsService
from backend.services.kb import KBService


class IngestWorker:
    """Background worker that processes queued ingest jobs with retries."""

    def __init__(self, *, max_attempts: int = 3) -> None:
        """Initialize worker clients and retry policy."""
        self.max_attempts = max_attempts
        self.base_backoff_seconds = 10
        self.max_backoff_seconds = 300
        self.jobs = IngestJobsService()
        self.kb = KBService()
        self.ingest = IngestService()

    def process_job(self, *, job_id: str, user_id: str | None = None) -> None:
        """Process one ingest job attempt and schedule retry on failure."""
        job = self.jobs.get_job(job_id=job_id, user_id=user_id)
        if not job:
            return
        if job.get('status') == 'completed':
            return

        if job.get('owner_type') != 'user':
            self.jobs.mark_failed(
                job_id=job_id,
                error='Only user-owned ingest jobs are supported by worker',
            )
            return

        owner_user_id = job.get('user_id')
        file_id = str(job.get('file_id') or '')
        if not owner_user_id or not file_id:
            self.jobs.move_to_dlq(
                job=job, reason='Ingest job payload is incomplete'
            )
            self.jobs.mark_failed(
                job_id=job_id,
                error='Ingest job payload is incomplete',
            )
            return

        attempt = int(job.get('attempt') or 0)
        if job.get('status') != 'processing':
            attempt += 1
            self.jobs.mark_processing(job_id=job_id, attempt=attempt)

        try:
            file_row = self.kb.get_file(file_id=file_id, user_id=owner_user_id)
            folder_id = file_row.get('folder_id')
            folder_name: str | None = None
            if folder_id:
                folder = self.kb.get_folder(
                    folder_id=folder_id,
                    user_id=owner_user_id,
                )
                folder_name = folder.get('name')

            # Make retry idempotent for Qdrant.
            self.kb.delete_vectors_for_file(file_id)
            self.ingest.ingest_file(
                file_id=file_row['id'],
                file_path=file_row['storage_path'],
                filename=file_row['filename'],
                mime=file_row['mime'],
                user_id=owner_user_id,
                folder_id=folder_id,
                folder_name=folder_name,
                source_path=job.get('source_path') or file_row.get('filename'),
            )
            self.jobs.mark_completed(job_id=job_id)
            return
        except HTTPException as exc:
            last_error = f'HTTP {exc.status_code}: {exc.detail}'
        except Exception as exc:
            last_error = str(exc)

        if attempt >= self.max_attempts:
            self.jobs.move_to_dlq(
                job=job,
                reason=f'Max attempts reached ({attempt})',
            )
            self.jobs.mark_failed(job_id=job_id, error=last_error)
            return

        backoff_seconds = min(
            self.base_backoff_seconds * (2 ** max(attempt - 1, 0)),
            self.max_backoff_seconds,
        )
        next_retry_at = datetime.now(timezone.utc) + timedelta(
            seconds=backoff_seconds
        )
        self.jobs.schedule_retry(
            job_id=job_id,
            error=last_error,
            next_retry_at=next_retry_at,
        )
