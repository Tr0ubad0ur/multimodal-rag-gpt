from __future__ import annotations

from datetime import datetime, timedelta, timezone

from backend.services.ingest import IngestService
from backend.services.storage import delete_stored_file
from backend.utils.supabase_client import get_supabase_client


class GuestCleanupService:
    """Remove expired guest uploads, vectors, jobs and DLQ snapshots."""

    def __init__(self) -> None:
        """Initialize DB and vector cleanup clients."""
        self.supabase = get_supabase_client(role='service')
        self.ingest = IngestService()

    def cleanup_expired(
        self,
        *,
        ttl_hours: int = 24,
        limit: int = 500,
    ) -> dict[str, int]:
        """Delete expired guest artifacts older than the configured TTL."""
        cutoff = datetime.now(timezone.utc) - timedelta(
            hours=max(1, ttl_hours)
        )
        cutoff_iso = cutoff.isoformat()

        resp = (
            self.supabase.table('ingest_jobs')
            .select('id,file_id,metadata')
            .eq('owner_type', 'guest')
            .lt('created_at', cutoff_iso)
            .limit(max(1, min(limit, 5000)))
            .execute()
        )
        jobs = getattr(resp, 'data', None) or []
        if not jobs:
            return {
                'jobs_deleted': 0,
                'vectors_deleted': 0,
                'files_deleted': 0,
                'dlq_deleted': 0,
            }

        job_ids = [str(job.get('id')) for job in jobs if job.get('id')]
        file_ids = {
            str(job.get('file_id')) for job in jobs if job.get('file_id')
        }
        storage_paths = {
            str((job.get('metadata') or {}).get('storage_path'))
            for job in jobs
            if (job.get('metadata') or {}).get('storage_path')
        }

        vectors_deleted = 0
        for file_id in file_ids:
            self.ingest.delete_vectors_for_file(file_id=file_id)
            vectors_deleted += 1

        files_deleted = 0
        for storage_path in storage_paths:
            delete_stored_file(storage_path)
            files_deleted += 1

        dlq_resp = (
            self.supabase.table('ingest_jobs_dlq')
            .select('id')
            .eq('owner_type', 'guest')
            .lt('created_at', cutoff_iso)
            .limit(max(1, min(limit, 5000)))
            .execute()
        )
        dlq_rows = getattr(dlq_resp, 'data', None) or []
        dlq_ids = [
            int(row.get('id')) for row in dlq_rows if row.get('id') is not None
        ]
        if dlq_ids:
            self.supabase.table('ingest_jobs_dlq').delete().in_(
                'id', dlq_ids
            ).execute()

        if job_ids:
            self.supabase.table('ingest_jobs').delete().in_(
                'id', job_ids
            ).execute()

        return {
            'jobs_deleted': len(job_ids),
            'vectors_deleted': vectors_deleted,
            'files_deleted': files_deleted,
            'dlq_deleted': len(dlq_ids),
        }
