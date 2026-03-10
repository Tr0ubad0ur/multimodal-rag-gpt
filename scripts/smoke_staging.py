#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.admin_rate_limiter import AdminRateLimiter  # noqa: E402
from backend.services.ingest_jobs import IngestJobsService  # noqa: E402
from backend.utils.supabase_client import get_supabase_client  # noqa: E402


def _http_ok(url: str) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            body = response.read().decode('utf-8', errors='ignore')
            return 200 <= getattr(response, 'status', 200) < 300, body[:400]
    except urllib.error.HTTPError as exc:
        return False, f'HTTP {exc.code}'
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    """Run a narrow production-gate smoke test against services and DB job flow."""
    parser = argparse.ArgumentParser(description='Staging smoke-check')
    parser.add_argument(
        '--web-url', default='http://localhost:18000/health/ready'
    )
    parser.add_argument(
        '--worker-url', default='http://localhost:18010/health/ready'
    )
    args = parser.parse_args()

    report: dict[str, object] = {'checks': []}

    web_ok, web_detail = _http_ok(args.web_url)
    report['checks'].append(
        {'name': 'web_readiness', 'ok': web_ok, 'detail': web_detail}
    )
    _assert(web_ok, f'Web readiness failed: {web_detail}')

    worker_ok, worker_detail = _http_ok(args.worker_url)
    report['checks'].append(
        {'name': 'worker_readiness', 'ok': worker_ok, 'detail': worker_detail}
    )
    _assert(worker_ok, f'Worker readiness failed: {worker_detail}')

    jobs = IngestJobsService()
    _assert(jobs.supabase is not None, 'Supabase service client unavailable')
    supabase = get_supabase_client(role='service')

    smoke_email = f'smoke-{uuid.uuid4()}@example.com'
    smoke_password = str(uuid.uuid4())
    created_user = supabase.auth.admin.create_user(
        {
            'email': smoke_email,
            'password': smoke_password,
            'email_confirm': True,
        }
    )
    smoke_user = getattr(created_user, 'user', None)
    smoke_user_id = getattr(smoke_user, 'id', None)
    _assert(bool(smoke_user_id), 'Failed to create smoke auth user')
    smoke_file_id = str(uuid.uuid4())
    smoke_worker_id = f'smoke-{uuid.uuid4()}'

    job_id = jobs.create_job(
        owner_type='user',
        owner_id=smoke_user_id,
        user_id=smoke_user_id,
        file_id=smoke_file_id,
        filename='smoke.txt',
        mime='text/plain',
        source_path='smoke.txt',
        folder_id=None,
        metadata={'origin': 'staging_smoke'},
    )
    _assert(bool(job_id), 'Failed to create ingest job')
    report['checks'].append(
        {'name': 'ingest_job_create', 'ok': True, 'detail': job_id}
    )

    claimed = jobs.claim_jobs(
        worker_id=smoke_worker_id, batch_size=1, lock_seconds=60
    )
    claimed_job = next(
        (item for item in claimed if str(item.get('id')) == job_id), None
    )
    _assert(claimed_job is not None, 'Created job was not claimable')
    report['checks'].append({'name': 'claim', 'ok': True, 'detail': job_id})

    jobs.mark_processing(job_id=job_id, attempt=1)
    jobs.schedule_retry(
        job_id=job_id,
        error='smoke retry',
        next_retry_at=datetime.now(timezone.utc) + timedelta(seconds=5),
    )
    retried = jobs.get_job(job_id=job_id, user_id=smoke_user_id)
    _assert(
        retried is not None and retried.get('status') == 'queued',
        'Retry scheduling failed',
    )
    report['checks'].append(
        {'name': 'retry', 'ok': True, 'detail': retried.get('status')}
    )

    jobs.mark_failed(job_id=job_id, error='smoke failed')
    failed_job = jobs.get_job(job_id=job_id, user_id=smoke_user_id)
    _assert(
        failed_job is not None and failed_job.get('status') == 'failed',
        'Mark failed failed',
    )
    jobs.move_to_dlq(job=failed_job, reason='smoke dlq')
    dlq_items = jobs.list_dlq(user_id=smoke_user_id, limit=10)
    dlq_item = next(
        (item for item in dlq_items if str(item.get('job_id')) == job_id), None
    )
    _assert(dlq_item is not None, 'DLQ insertion failed')
    report['checks'].append(
        {'name': 'dlq', 'ok': True, 'detail': dlq_item.get('id')}
    )

    limiter = AdminRateLimiter()
    scope = f'smoke-admin-{uuid.uuid4()}'
    first = limiter.is_allowed(scope=scope, limit=1, window_seconds=60)
    second = limiter.is_allowed(scope=scope, limit=1, window_seconds=60)
    _assert(first is True and second is False, 'Admin rate limit smoke failed')
    report['checks'].append(
        {
            'name': 'admin_rate_limit',
            'ok': True,
            'detail': '1 then 429-equivalent',
        }
    )

    if jobs.supabase is not None:
        jobs.supabase.table('ingest_jobs_dlq').delete().eq(
            'user_id', smoke_user_id
        ).execute()
        jobs.supabase.table('ingest_jobs').delete().eq(
            'user_id', smoke_user_id
        ).execute()
    try:
        supabase.auth.admin.delete_user(smoke_user_id)
    except Exception:
        pass

    sys.stdout.write(f'{json.dumps(report, ensure_ascii=False, indent=2)}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
