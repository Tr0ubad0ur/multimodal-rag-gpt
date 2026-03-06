from __future__ import annotations

import os
from typing import Literal

RuntimeMode = Literal['web', 'worker']

_BASE_REQUIRED_VARS = (
    'SUPABASE_URL',
    'SUPABASE_ANON_KEY',
    'SUPABASE_SERVICE_ROLE_KEY',
    'QDRANT_URL',
)


def _is_truthy(raw: str | None) -> bool:
    return (raw or '').strip().lower() not in {'', '0', 'false', 'no', 'off'}


def validate_runtime_env(*, mode: RuntimeMode) -> dict[str, object]:
    """Validate runtime env vars required for web/worker deployment."""
    required = set(_BASE_REQUIRED_VARS)
    if mode == 'web':
        required.update(
            {
                'ADMIN_API_KEY',
                'INGEST_POLLER_ENABLED',
                'ADMIN_RATE_LIMIT_PER_MINUTE',
            }
        )
    else:
        required.add('INGEST_WORKER_MAX_CONCURRENCY')

    missing = sorted(
        var_name
        for var_name in required
        if not (os.getenv(var_name) or '').strip()
    )
    invalid: list[str] = []

    if mode == 'web':
        rate_limit_raw = (
            os.getenv('ADMIN_RATE_LIMIT_PER_MINUTE') or ''
        ).strip()
        if rate_limit_raw:
            try:
                if int(rate_limit_raw) < 1:
                    invalid.append(
                        'ADMIN_RATE_LIMIT_PER_MINUTE must be an integer >= 1'
                    )
            except ValueError:
                invalid.append(
                    'ADMIN_RATE_LIMIT_PER_MINUTE must be an integer'
                )

        # In production/staging web must not run embedded poller.
        is_prod_like = _is_truthy(os.getenv('PRODUCTION')) or (
            (os.getenv('APP_ENV') or '').strip().lower()
            in {'production', 'staging'}
        )
        if is_prod_like and _is_truthy(os.getenv('INGEST_POLLER_ENABLED')):
            invalid.append(
                'INGEST_POLLER_ENABLED must be false for web in production/staging'
            )
    else:
        concurrency_raw = (
            os.getenv('INGEST_WORKER_MAX_CONCURRENCY') or ''
        ).strip()
        if concurrency_raw:
            try:
                if int(concurrency_raw) < 1:
                    invalid.append(
                        'INGEST_WORKER_MAX_CONCURRENCY must be an integer >= 1'
                    )
            except ValueError:
                invalid.append(
                    'INGEST_WORKER_MAX_CONCURRENCY must be an integer'
                )

    redis_url = (os.getenv('REDIS_URL') or '').strip()
    if redis_url and not (
        redis_url.startswith('redis://') or redis_url.startswith('rediss://')
    ):
        invalid.append('REDIS_URL must start with redis:// or rediss://')

    return {
        'ok': not missing and not invalid,
        'mode': mode,
        'missing': missing,
        'invalid': invalid,
    }
