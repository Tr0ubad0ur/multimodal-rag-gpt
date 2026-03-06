from backend.services.runtime_config import validate_runtime_env


def _set_base_env(monkeypatch) -> None:
    monkeypatch.setenv('SUPABASE_URL', 'https://example.supabase.co')
    monkeypatch.setenv('SUPABASE_ANON_KEY', 'anon-key')
    monkeypatch.setenv('SUPABASE_SERVICE_ROLE_KEY', 'service-key')
    monkeypatch.setenv('QDRANT_URL', 'http://localhost:6333')


def test_validate_runtime_env_web_ok(monkeypatch) -> None:
    """Validate required production env for web process."""
    _set_base_env(monkeypatch)
    monkeypatch.setenv('ADMIN_API_KEY', 'admin-secret')
    monkeypatch.setenv('INGEST_POLLER_ENABLED', 'false')
    monkeypatch.setenv('ADMIN_RATE_LIMIT_PER_MINUTE', '60')
    monkeypatch.setenv('APP_ENV', 'production')
    monkeypatch.delenv('REDIS_URL', raising=False)

    result = validate_runtime_env(mode='web')

    assert result['ok'] is True
    assert result['missing'] == []
    assert result['invalid'] == []


def test_validate_runtime_env_web_rejects_embedded_poller_in_prod(
    monkeypatch,
) -> None:
    """Reject web process when embedded poller is enabled in production mode."""
    _set_base_env(monkeypatch)
    monkeypatch.setenv('ADMIN_API_KEY', 'admin-secret')
    monkeypatch.setenv('INGEST_POLLER_ENABLED', 'true')
    monkeypatch.setenv('ADMIN_RATE_LIMIT_PER_MINUTE', '60')
    monkeypatch.setenv('APP_ENV', 'production')

    result = validate_runtime_env(mode='web')

    assert result['ok'] is False
    assert (
        'INGEST_POLLER_ENABLED must be false for web in production/staging'
        in (result['invalid'])
    )


def test_validate_runtime_env_worker_requires_positive_concurrency(
    monkeypatch,
) -> None:
    """Reject worker runtime when max concurrency is not a positive integer."""
    _set_base_env(monkeypatch)
    monkeypatch.setenv('INGEST_WORKER_MAX_CONCURRENCY', '0')

    result = validate_runtime_env(mode='worker')

    assert result['ok'] is False
    assert (
        'INGEST_WORKER_MAX_CONCURRENCY must be an integer >= 1'
        in result['invalid']
    )
