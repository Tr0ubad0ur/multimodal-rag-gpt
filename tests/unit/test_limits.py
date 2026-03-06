import pytest
from fastapi import HTTPException

from backend.api.endpoints import _enforce_quota_capacity, _env_int
from backend.services.request_rate_limiter import RequestRateLimiter


def test_env_int_returns_default_when_empty(monkeypatch) -> None:
    """Use default int value when env var is not set."""
    monkeypatch.delenv('SOME_LIMIT', raising=False)
    assert _env_int('SOME_LIMIT', 10) == 10


def test_env_int_raises_on_invalid_value(monkeypatch) -> None:
    """Raise explicit HTTP error on non-integer env value."""
    monkeypatch.setenv('BROKEN_LIMIT', 'abc')
    with pytest.raises(HTTPException) as exc:
        _env_int('BROKEN_LIMIT', 10)
    assert exc.value.status_code == 500


def test_enforce_quota_capacity_rejects_file_quota(monkeypatch) -> None:
    """Reject uploads when projected file count exceeds limit."""
    monkeypatch.setenv('MAX_FILES_PER_USER', '1')
    monkeypatch.setenv('MAX_STORAGE_BYTES_PER_USER', str(1024 * 1024))
    with pytest.raises(HTTPException) as exc:
        _enforce_quota_capacity(total_files_after=2, total_size_after=10)
    assert exc.value.status_code == 413
    assert exc.value.detail == 'User file quota exceeded'


def test_request_rate_limiter_blocks_after_limit() -> None:
    """Block second request inside same window when limit is one."""
    limiter = RequestRateLimiter()
    scope = 'unit:test'
    assert limiter.is_allowed(scope=scope, limit=1, window_seconds=60) is True
    assert limiter.is_allowed(scope=scope, limit=1, window_seconds=60) is False
