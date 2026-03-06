from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue

from backend.services.ingest_jobs import IngestJobsService
from backend.services.kb import KBService
from backend.services.storage import delete_stored_file
from backend.utils.supabase_client import get_supabase_client


class DataConsistencyService:
    """Consistency operations: reindex scheduling and orphan cleanup."""

    def __init__(self) -> None:
        """Initialize DB/vector clients and job service."""
        self.supabase = get_supabase_client(role='service')
        self.kb = KBService()
        self.jobs = IngestJobsService()

    def schedule_reindex(
        self,
        *,
        user_id: str | None = None,
        file_ids: list[str] | None = None,
        limit: int = 100,
        only_missing_vectors: bool = False,
    ) -> dict[str, Any]:
        """Create ingest jobs for files that should be reindexed."""
        rows = self._list_kb_files(
            user_id=user_id,
            file_ids=file_ids,
            limit=max(1, min(limit, 2_000)),
        )
        scheduled: list[dict[str, str]] = []
        skipped_existing_vectors = 0
        for row in rows:
            file_id = str(row.get('id') or '')
            owner_user_id = str(row.get('user_id') or '')
            if not file_id or not owner_user_id:
                continue

            if only_missing_vectors and self.kb.has_vectors_for_file(
                file_id=file_id
            ):
                skipped_existing_vectors += 1
                continue

            folder_id = row.get('folder_id')
            folder_path = self.kb.get_folder_path(
                user_id=owner_user_id,
                folder_id=folder_id,
            )
            job_id = self.jobs.create_job(
                owner_type='user',
                owner_id=owner_user_id,
                user_id=owner_user_id,
                file_id=file_id,
                filename=str(row.get('filename') or ''),
                mime=str(row.get('mime') or ''),
                source_path=str(row.get('filename') or ''),
                folder_id=folder_id,
                metadata={
                    'origin': 'manual_reindex',
                    'folder_path': folder_path,
                },
            )
            if not job_id:
                continue
            scheduled.append({'job_id': job_id, 'user_id': owner_user_id})

        return {
            'selected': len(rows),
            'scheduled': len(scheduled),
            'skipped_existing_vectors': skipped_existing_vectors,
            'jobs': scheduled,
        }

    def cleanup_missing_storage_records(
        self,
        *,
        dry_run: bool = True,
        user_id: str | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        """Delete DB/Qdrant rows for kb_files pointing to missing local files."""
        rows = self._list_kb_files(user_id=user_id, file_ids=None, limit=limit)
        missing: list[dict[str, str]] = []
        removed = 0
        for row in rows:
            file_id = str(row.get('id') or '')
            owner_user_id = str(row.get('user_id') or '')
            storage_path = str(row.get('storage_path') or '')
            if not file_id or not owner_user_id or not storage_path:
                continue
            if Path(storage_path).exists():
                continue
            missing.append(
                {
                    'file_id': file_id,
                    'user_id': owner_user_id,
                    'storage_path': storage_path,
                }
            )
            if dry_run:
                continue
            self.kb.delete_vectors_for_file(file_id)
            (
                self.supabase.table('kb_files')
                .delete()
                .eq('id', file_id)
                .eq('user_id', owner_user_id)
                .execute()
            )
            removed += 1

        return {
            'checked': len(rows),
            'missing_records': len(missing),
            'removed': removed,
            'items': missing,
            'dry_run': dry_run,
        }

    def cleanup_orphan_uploads(
        self,
        *,
        dry_run: bool = True,
        min_age_seconds: int = 1_800,
        limit: int = 500,
    ) -> dict[str, Any]:
        """Delete local uploaded files not referenced by kb_files."""
        uploads_dir = Path(os.getenv('UPLOADS_DIR', 'data/uploads'))
        if not uploads_dir.exists() or not uploads_dir.is_dir():
            return {
                'checked': 0,
                'orphan_uploads': 0,
                'deleted': 0,
                'items': [],
                'dry_run': dry_run,
            }

        referenced = self._list_storage_paths(limit=20_000)
        now = time.time()
        candidates: list[str] = []
        for path in sorted(uploads_dir.glob('*')):
            if not path.is_file():
                continue
            abs_path = str(path)
            if abs_path in referenced:
                continue
            age_seconds = now - path.stat().st_mtime
            if age_seconds < max(0, min_age_seconds):
                continue
            candidates.append(abs_path)
            if len(candidates) >= max(1, limit):
                break

        deleted = 0
        if not dry_run:
            for storage_path in candidates:
                delete_stored_file(storage_path)
                deleted += 1

        return {
            'checked': len(list(uploads_dir.glob('*'))),
            'orphan_uploads': len(candidates),
            'deleted': deleted,
            'items': candidates,
            'dry_run': dry_run,
        }

    def cleanup_orphan_vectors(
        self,
        *,
        dry_run: bool = True,
        limit_orphan_file_ids: int = 2_000,
    ) -> dict[str, Any]:
        """Delete vectors whose `file_id` no longer exists in kb_files."""
        existing_file_ids = self._list_file_ids(limit=50_000)
        orphan_file_ids_by_collection: dict[str, set[str]] = defaultdict(set)
        scanned_points = 0

        for collection in self.kb._all_collection_names():
            offset = None
            while True:
                try:
                    points, next_offset = self.kb.qdrant.scroll(
                        collection_name=collection,
                        scroll_filter=None,
                        with_payload=True,
                        with_vectors=False,
                        limit=256,
                        offset=offset,
                    )
                except Exception:
                    break

                if not points:
                    break

                for point in points:
                    scanned_points += 1
                    payload = point.payload or {}
                    file_id = payload.get('file_id')
                    owner_type = payload.get('owner_type')
                    if owner_type and owner_type != 'user':
                        continue
                    if not file_id:
                        continue
                    if file_id not in existing_file_ids:
                        orphan_file_ids_by_collection[collection].add(
                            str(file_id)
                        )
                    if sum(
                        len(items)
                        for items in orphan_file_ids_by_collection.values()
                    ) >= max(1, limit_orphan_file_ids):
                        break
                else:
                    if next_offset is None:
                        break
                    offset = next_offset
                    continue
                break

        deleted_file_ids = 0
        if not dry_run:
            for (
                collection,
                orphan_file_ids,
            ) in orphan_file_ids_by_collection.items():
                for orphan_file_id in orphan_file_ids:
                    selector = Filter(
                        must=[
                            FieldCondition(
                                key='file_id',
                                match=MatchValue(value=orphan_file_id),
                            )
                        ]
                    )
                    try:
                        self.kb.qdrant.delete(
                            collection_name=collection,
                            points_selector=selector,
                        )
                        deleted_file_ids += 1
                    except Exception:
                        continue

        serializable = {
            key: sorted(value)
            for key, value in orphan_file_ids_by_collection.items()
        }
        return {
            'scanned_points': scanned_points,
            'orphan_file_ids_total': sum(
                len(v) for v in serializable.values()
            ),
            'deleted_file_ids': deleted_file_ids,
            'collections': serializable,
            'dry_run': dry_run,
        }

    def _list_kb_files(
        self,
        *,
        user_id: str | None,
        file_ids: list[str] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        query = (
            self.supabase.table('kb_files')
            .select('*')
            .order('created_at', desc=True)
            .limit(max(1, limit))
        )
        if user_id:
            query = query.eq('user_id', user_id)
        if file_ids:
            query = query.in_('id', file_ids)
        resp = query.execute()
        return getattr(resp, 'data', None) or []

    def _list_storage_paths(self, *, limit: int) -> set[str]:
        query = (
            self.supabase.table('kb_files')
            .select('storage_path')
            .limit(max(1, limit))
        )
        resp = query.execute()
        rows = getattr(resp, 'data', None) or []
        return {
            str(row.get('storage_path'))
            for row in rows
            if row.get('storage_path')
        }

    def _list_file_ids(self, *, limit: int) -> set[str]:
        query = (
            self.supabase.table('kb_files').select('id').limit(max(1, limit))
        )
        resp = query.execute()
        rows = getattr(resp, 'data', None) or []
        return {str(row.get('id')) for row in rows if row.get('id')}
