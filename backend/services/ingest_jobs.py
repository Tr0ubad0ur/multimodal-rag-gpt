from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from backend.monitoring.metrics import (
    observe_ingest_job_event,
    observe_ingest_retry_delay,
    set_ingest_dlq_depth,
    set_ingest_queue_depth,
)
from backend.utils.supabase_client import get_supabase_client


class IngestJobsService:
    """Track ingest lifecycle in Supabase, fallback to no-op when unavailable."""

    def __init__(self) -> None:
        """Initialize Supabase client for ingest job operations."""
        try:
            self.supabase = get_supabase_client(role='service')
        except Exception:
            self.supabase = None

    def create_job(
        self,
        *,
        owner_type: str,
        owner_id: str,
        file_id: str,
        filename: str,
        mime: str,
        source_path: str | None = None,
        folder_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Create ingest job row and return job id."""
        if self.supabase is None:
            return None
        try:
            resp = (
                self.supabase.table('ingest_jobs')
                .insert(
                    {
                        'owner_type': owner_type,
                        'owner_id': owner_id,
                        'file_id': file_id,
                        'filename': filename,
                        'mime': mime,
                        'source_path': source_path,
                        'folder_id': folder_id,
                        'user_id': user_id,
                        'metadata': metadata or {},
                    }
                )
                .execute()
            )
            data = getattr(resp, 'data', None) or []
            if not data:
                observe_ingest_job_event('create', 'error')
                return None
            observe_ingest_job_event('create', 'ok')
            return data[0].get('id')
        except Exception:
            observe_ingest_job_event('create', 'error')
            return None

    def refresh_depth_metrics(self) -> None:
        """Refresh queue depth gauges from Supabase counts."""
        statuses = ['queued', 'processing', 'completed', 'failed']
        for status in statuses:
            set_ingest_queue_depth(status, self._count_jobs(status=status))
        set_ingest_dlq_depth(self._count_dlq())

    def list_jobs(
        self,
        *,
        user_id: str,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List user's ingest jobs with optional status filter."""
        if self.supabase is None:
            return []
        try:
            query = (
                self.supabase.table('ingest_jobs')
                .select('*')
                .eq('user_id', user_id)
                .order('created_at', desc=True)
                .limit(limit)
            )
            if status:
                query = query.eq('status', status)
            resp = query.execute()
            return getattr(resp, 'data', None) or []
        except Exception:
            return []

    def list_jobs_for_file(
        self,
        *,
        user_id: str,
        file_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List ingest jobs for one file and user."""
        if self.supabase is None:
            return []
        try:
            resp = (
                self.supabase.table('ingest_jobs')
                .select('*')
                .eq('user_id', user_id)
                .eq('file_id', file_id)
                .order('created_at', desc=True)
                .limit(limit)
                .execute()
            )
            return getattr(resp, 'data', None) or []
        except Exception:
            return []

    def list_jobs_admin(
        self,
        *,
        status: str,
        limit: int = 50,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List jobs for admin batch actions."""
        if self.supabase is None:
            return []
        try:
            query = (
                self.supabase.table('ingest_jobs')
                .select('*')
                .eq('status', status)
                .order('created_at')
                .limit(limit)
            )
            if user_id:
                query = query.eq('user_id', user_id)
            resp = query.execute()
            return getattr(resp, 'data', None) or []
        except Exception:
            return []

    def list_queued_jobs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """List queued user-owned jobs in creation order."""
        if self.supabase is None:
            return []
        try:
            resp = (
                self.supabase.table('ingest_jobs')
                .select('*')
                .eq('status', 'queued')
                .eq('owner_type', 'user')
                .order('created_at')
                .limit(limit)
                .execute()
            )
            return getattr(resp, 'data', None) or []
        except Exception:
            return []

    def claim_jobs(
        self,
        *,
        worker_id: str,
        batch_size: int = 10,
        lock_seconds: int = 300,
    ) -> list[dict[str, Any]]:
        """Atomically claim queued jobs for processing."""
        if self.supabase is None:
            return []
        try:
            resp = self.supabase.rpc(
                'claim_ingest_jobs',
                {
                    'p_worker_id': worker_id,
                    'p_batch_size': max(1, batch_size),
                    'p_lock_seconds': max(30, lock_seconds),
                },
            ).execute()
            return getattr(resp, 'data', None) or []
        except Exception:
            # Graceful fallback for environments without migration applied yet.
            return self.list_queued_jobs(limit=batch_size)

    def requeue_stale_processing(self, *, stale_seconds: int = 300) -> int:
        """Move stale processing jobs back to queued state."""
        if self.supabase is None:
            return 0
        cutoff = datetime.now(timezone.utc).timestamp() - stale_seconds
        cutoff_iso = datetime.fromtimestamp(
            cutoff, tz=timezone.utc
        ).isoformat()
        try:
            resp = (
                self.supabase.table('ingest_jobs')
                .update(
                    {
                        'status': 'queued',
                        'claimed_by': None,
                        'claim_expires_at': None,
                    }
                )
                .eq('status', 'processing')
                .eq('owner_type', 'user')
                .lt('claim_expires_at', cutoff_iso)
                .execute()
            )
            return len(getattr(resp, 'data', None) or [])
        except Exception:
            return 0

    def purge_jobs(
        self,
        *,
        statuses: list[str],
        older_than_hours: int,
        limit: int = 500,
    ) -> int:
        """Delete old jobs by status in batches."""
        if self.supabase is None:
            return 0
        cutoff = datetime.now(timezone.utc).timestamp() - (
            max(1, older_than_hours) * 3600
        )
        cutoff_iso = datetime.fromtimestamp(
            cutoff, tz=timezone.utc
        ).isoformat()
        try:
            ids_resp = (
                self.supabase.table('ingest_jobs')
                .select('id')
                .in_('status', statuses)
                .lt('finished_at', cutoff_iso)
                .limit(max(1, min(limit, 2000)))
                .execute()
            )
            ids = [
                row.get('id')
                for row in (getattr(ids_resp, 'data', None) or [])
            ]
            ids = [job_id for job_id in ids if job_id]
            if not ids:
                return 0
            (
                self.supabase.table('ingest_jobs')
                .delete()
                .in_('id', ids)
                .execute()
            )
            observe_ingest_job_event('purge_jobs', 'ok')
            return len(ids)
        except Exception:
            observe_ingest_job_event('purge_jobs', 'error')
            return 0

    def purge_dlq(
        self,
        *,
        older_than_hours: int,
        limit: int = 500,
    ) -> int:
        """Delete old DLQ records in batches."""
        if self.supabase is None:
            return 0
        cutoff = datetime.now(timezone.utc).timestamp() - (
            max(1, older_than_hours) * 3600
        )
        cutoff_iso = datetime.fromtimestamp(
            cutoff, tz=timezone.utc
        ).isoformat()
        try:
            ids_resp = (
                self.supabase.table('ingest_jobs_dlq')
                .select('id')
                .lt('created_at', cutoff_iso)
                .limit(max(1, min(limit, 2000)))
                .execute()
            )
            ids = [
                row.get('id')
                for row in (getattr(ids_resp, 'data', None) or [])
            ]
            ids = [dlq_id for dlq_id in ids if dlq_id is not None]
            if not ids:
                return 0
            (
                self.supabase.table('ingest_jobs_dlq')
                .delete()
                .in_('id', ids)
                .execute()
            )
            observe_ingest_job_event('purge_dlq', 'ok')
            return len(ids)
        except Exception:
            observe_ingest_job_event('purge_dlq', 'error')
            return 0

    def get_job(
        self,
        *,
        job_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Get one ingest job by id with optional user ownership check."""
        if self.supabase is None:
            return None
        try:
            query = (
                self.supabase.table('ingest_jobs').select('*').eq('id', job_id)
            )
            if user_id:
                query = query.eq('user_id', user_id)
            resp = query.limit(1).execute()
            data = getattr(resp, 'data', None) or []
            if not data:
                return None
            return data[0]
        except Exception:
            return None

    def mark_processing(
        self, *, job_id: str | None, attempt: int | None = None
    ) -> None:
        """Mark job as processing and refresh lock metadata."""
        if self.supabase is None or not job_id:
            return
        payload: dict[str, Any] = {
            'status': 'processing',
            'started_at': datetime.now(timezone.utc).isoformat(),
            'claim_expires_at': datetime.fromtimestamp(
                datetime.now(timezone.utc).timestamp() + 300,
                tz=timezone.utc,
            ).isoformat(),
            'error': None,
            'next_retry_at': None,
            'dead_lettered_at': None,
        }
        if attempt is not None:
            payload['attempt'] = attempt
        try:
            (
                self.supabase.table('ingest_jobs')
                .update(payload)
                .eq('id', job_id)
                .execute()
            )
            observe_ingest_job_event('processing', 'ok')
        except Exception:
            observe_ingest_job_event('processing', 'error')
            return

    def mark_queued(self, *, job_id: str | None) -> None:
        """Mark job as queued and clear runtime execution fields."""
        if self.supabase is None or not job_id:
            return
        try:
            (
                self.supabase.table('ingest_jobs')
                .update(
                    {
                        'status': 'queued',
                        'started_at': None,
                        'finished_at': None,
                        'claimed_by': None,
                        'claim_expires_at': None,
                        'next_retry_at': None,
                    }
                )
                .eq('id', job_id)
                .execute()
            )
            observe_ingest_job_event('queued', 'ok')
        except Exception:
            observe_ingest_job_event('queued', 'error')
            return

    def mark_completed(self, *, job_id: str | None) -> None:
        """Mark job as completed and clear retry/lock fields."""
        if self.supabase is None or not job_id:
            return
        try:
            (
                self.supabase.table('ingest_jobs')
                .update(
                    {
                        'status': 'completed',
                        'finished_at': datetime.now(timezone.utc).isoformat(),
                        'claimed_by': None,
                        'claim_expires_at': None,
                        'next_retry_at': None,
                        'dead_lettered_at': None,
                        'error': None,
                    }
                )
                .eq('id', job_id)
                .execute()
            )
            observe_ingest_job_event('completed', 'ok')
        except Exception:
            observe_ingest_job_event('completed', 'error')
            return

    def schedule_retry(
        self,
        *,
        job_id: str | None,
        error: str,
        next_retry_at: datetime,
    ) -> None:
        """Schedule failed job for retry at specific timestamp."""
        if self.supabase is None or not job_id:
            return
        try:
            (
                self.supabase.table('ingest_jobs')
                .update(
                    {
                        'status': 'queued',
                        'finished_at': datetime.now(timezone.utc).isoformat(),
                        'claimed_by': None,
                        'claim_expires_at': None,
                        'next_retry_at': next_retry_at.astimezone(
                            timezone.utc
                        ).isoformat(),
                        'error': error[:4000],
                    }
                )
                .eq('id', job_id)
                .execute()
            )
            observe_ingest_job_event('retry_scheduled', 'ok')
            delay = (
                next_retry_at.astimezone(timezone.utc)
                - datetime.now(timezone.utc)
            ).total_seconds()
            observe_ingest_retry_delay(delay)
        except Exception:
            observe_ingest_job_event('retry_scheduled', 'error')
            return

    def mark_failed(self, *, job_id: str | None, error: str) -> None:
        """Mark job as failed and set dead-letter timestamp metadata."""
        if self.supabase is None or not job_id:
            return
        try:
            (
                self.supabase.table('ingest_jobs')
                .update(
                    {
                        'status': 'failed',
                        'finished_at': datetime.now(timezone.utc).isoformat(),
                        'claimed_by': None,
                        'claim_expires_at': None,
                        'next_retry_at': None,
                        'dead_lettered_at': datetime.now(
                            timezone.utc
                        ).isoformat(),
                        'error': error[:4000],
                    }
                )
                .eq('id', job_id)
                .execute()
            )
            observe_ingest_job_event('failed', 'ok')
        except Exception:
            observe_ingest_job_event('failed', 'error')
            return

    def move_to_dlq(
        self,
        *,
        job: dict[str, Any] | None,
        reason: str,
    ) -> None:
        """Persist failed job snapshot into DLQ table."""
        if self.supabase is None or not job:
            return
        try:
            (
                self.supabase.table('ingest_jobs_dlq')
                .insert(
                    {
                        'job_id': job.get('id'),
                        'user_id': job.get('user_id'),
                        'owner_type': job.get('owner_type'),
                        'owner_id': job.get('owner_id'),
                        'reason': reason[:1000],
                        'payload': job,
                    }
                )
                .execute()
            )
            observe_ingest_job_event('dlq_insert', 'ok')
        except Exception:
            observe_ingest_job_event('dlq_insert', 'error')
            return

    def list_dlq(
        self,
        *,
        user_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List DLQ items for one user."""
        if self.supabase is None:
            return []
        try:
            resp = (
                self.supabase.table('ingest_jobs_dlq')
                .select('*')
                .eq('user_id', user_id)
                .order('created_at', desc=True)
                .limit(limit)
                .execute()
            )
            return getattr(resp, 'data', None) or []
        except Exception:
            return []

    def get_dlq_item(
        self,
        *,
        dlq_id: int,
        user_id: str,
    ) -> dict[str, Any] | None:
        """Get one DLQ item by id for one user."""
        if self.supabase is None:
            return None
        try:
            resp = (
                self.supabase.table('ingest_jobs_dlq')
                .select('*')
                .eq('id', dlq_id)
                .eq('user_id', user_id)
                .limit(1)
                .execute()
            )
            data = getattr(resp, 'data', None) or []
            if not data:
                return None
            return data[0]
        except Exception:
            return None

    def requeue_from_dlq(
        self,
        *,
        dlq_id: int,
        user_id: str,
    ) -> dict[str, Any] | None:
        """Move job from DLQ back to queued state and remove DLQ row."""
        if self.supabase is None:
            return None
        dlq_item = self.get_dlq_item(dlq_id=dlq_id, user_id=user_id)
        if not dlq_item:
            return None

        job_id = dlq_item.get('job_id')
        if not job_id:
            return None
        job = self.get_job(job_id=str(job_id), user_id=user_id)
        if not job:
            return None

        try:
            (
                self.supabase.table('ingest_jobs')
                .update(
                    {
                        'status': 'queued',
                        'attempt': 0,
                        'started_at': None,
                        'finished_at': None,
                        'claimed_by': None,
                        'claim_expires_at': None,
                        'next_retry_at': None,
                        'dead_lettered_at': None,
                        'error': None,
                    }
                )
                .eq('id', str(job_id))
                .eq('user_id', user_id)
                .execute()
            )
            (
                self.supabase.table('ingest_jobs_dlq')
                .delete()
                .eq('id', dlq_id)
                .eq('user_id', user_id)
                .execute()
            )
            observe_ingest_job_event('dlq_requeue', 'ok')
        except Exception:
            observe_ingest_job_event('dlq_requeue', 'error')
            return None
        return self.get_job(job_id=str(job_id), user_id=user_id)

    def _count_jobs(self, *, status: str) -> int:
        if self.supabase is None:
            return 0
        try:
            resp = (
                self.supabase.table('ingest_jobs')
                .select('id', count='exact', head=True)
                .eq('status', status)
                .execute()
            )
            return int(getattr(resp, 'count', 0) or 0)
        except Exception:
            return 0

    def _count_dlq(self) -> int:
        if self.supabase is None:
            return 0
        try:
            resp = (
                self.supabase.table('ingest_jobs_dlq')
                .select('id', count='exact', head=True)
                .execute()
            )
            return int(getattr(resp, 'count', 0) or 0)
        except Exception:
            return 0
