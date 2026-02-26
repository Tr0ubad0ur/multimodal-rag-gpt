import re

from prometheus_client import Counter, Histogram

HTTP_REQUESTS_TOTAL = Counter(
    'http_requests_total',
    'Total number of HTTP requests.',
    ['method', 'path', 'status'],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency in seconds.',
    ['method', 'path'],
    buckets=(0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0),
)

RAG_QUERIES_TOTAL = Counter(
    'rag_queries_total',
    'Total number of RAG queries.',
    ['query_type', 'status'],
)
RAG_QUERY_DURATION_SECONDS = Histogram(
    'rag_query_duration_seconds',
    'RAG query latency in seconds.',
    ['query_type'],
    buckets=(0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0),
)
RAG_RETRIEVED_DOCS = Histogram(
    'rag_retrieved_docs',
    'Number of retrieved documents per RAG query.',
    ['query_type'],
    buckets=(0, 1, 2, 3, 5, 10, 20, 50),
)

EMBEDDING_REQUESTS_TOTAL = Counter(
    'embedding_requests_total',
    'Total number of embedding requests.',
    ['modality', 'provider', 'status'],
)
EMBEDDING_DURATION_SECONDS = Histogram(
    'embedding_duration_seconds',
    'Embedding generation latency in seconds.',
    ['modality', 'provider'],
    buckets=(0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0),
)


def normalize_http_path(path: str) -> str:
    """Reduce path cardinality for metrics labels."""
    normalized = re.sub(
        r'/[0-9a-fA-F-]{8,}',
        '/{token}',
        path,
    )
    normalized = re.sub(r'/\d+(?=/|$)', '/{id}', normalized)
    return normalized


def observe_http_request(
    method: str,
    path: str,
    status: int,
    duration_seconds: float,
) -> None:
    """Observe common HTTP metrics."""
    normalized_path = normalize_http_path(path)
    HTTP_REQUESTS_TOTAL.labels(
        method=method,
        path=normalized_path,
        status=str(status),
    ).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=method,
        path=normalized_path,
    ).observe(duration_seconds)


def observe_rag_query(
    query_type: str,
    status: str,
    duration_seconds: float,
    retrieved_docs_count: int,
) -> None:
    """Observe RAG query metrics."""
    RAG_QUERIES_TOTAL.labels(query_type=query_type, status=status).inc()
    RAG_QUERY_DURATION_SECONDS.labels(query_type=query_type).observe(
        duration_seconds
    )
    RAG_RETRIEVED_DOCS.labels(query_type=query_type).observe(
        retrieved_docs_count
    )


def observe_embedding_request(
    modality: str,
    provider: str,
    status: str,
    duration_seconds: float,
) -> None:
    """Observe embedding generation metrics."""
    EMBEDDING_REQUESTS_TOTAL.labels(
        modality=modality,
        provider=provider,
        status=status,
    ).inc()
    EMBEDDING_DURATION_SECONDS.labels(
        modality=modality,
        provider=provider,
    ).observe(duration_seconds)
