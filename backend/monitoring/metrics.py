import re

from prometheus_client import Counter, Gauge, Histogram

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

INGEST_JOB_EVENTS_TOTAL = Counter(
    'ingest_job_events_total',
    'Total number of ingest job lifecycle events.',
    ['event', 'result'],
)
INGEST_RETRY_DELAY_SECONDS = Histogram(
    'ingest_retry_delay_seconds',
    'Delay before next ingest retry in seconds.',
    buckets=(1, 5, 10, 20, 30, 60, 120, 300, 600, 1800),
)
INGEST_QUEUE_DEPTH = Gauge(
    'ingest_queue_depth',
    'Current ingest job queue depth by status.',
    ['status'],
)
INGEST_DLQ_DEPTH = Gauge(
    'ingest_dlq_depth',
    'Current ingest dead-letter queue depth.',
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


def observe_ingest_job_event(event: str, result: str = 'ok') -> None:
    """Observe ingest job event counters."""
    INGEST_JOB_EVENTS_TOTAL.labels(event=event, result=result).inc()


def observe_ingest_retry_delay(delay_seconds: float) -> None:
    """Observe scheduled retry delay."""
    INGEST_RETRY_DELAY_SECONDS.observe(max(delay_seconds, 0.0))


def set_ingest_queue_depth(status: str, depth: int) -> None:
    """Set ingest queue depth gauge for status."""
    INGEST_QUEUE_DEPTH.labels(status=status).set(max(depth, 0))


def set_ingest_dlq_depth(depth: int) -> None:
    """Set ingest DLQ depth gauge."""
    INGEST_DLQ_DEPTH.set(max(depth, 0))
