from backend.monitoring.metrics import normalize_http_path


def test_normalize_http_path_replaces_numeric_ids() -> None:
    """Replace numeric path segments with a stable {id} placeholder."""
    assert normalize_http_path('/history/123') == '/history/{id}'


def test_normalize_http_path_replaces_tokens() -> None:
    """Replace UUID-like path segments with a stable {token} placeholder."""
    path = '/history/550e8400-e29b-41d4-a716-446655440000'
    assert normalize_http_path(path) == '/history/{token}'
