# Production Checklist

## 1. Backend release gate

- Apply all Supabase migrations
- Run staging smoke-check
- Confirm:
  - ingest job create
  - claim
  - retry
  - DLQ insert/list
  - admin rate limit

Suggested command:

```bash
uv run python scripts/smoke_staging.py \
  --web-url http://localhost:18000/health/ready \
  --worker-url http://localhost:18010/health/ready
```

## 2. Production env freeze

Required:

- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`
- `QDRANT_URL`
- `ADMIN_API_KEY`
- `INGEST_POLLER_ENABLED=false`
- `INGEST_WORKER_MAX_CONCURRENCY`
- `ADMIN_RATE_LIMIT_PER_MINUTE`

Optional:

- `REDIS_URL`

Example file:

- [`.env.production.example`](.env.production.example)

Recommended runtime:

- `APP_MODE=web` for web
- `APP_MODE=worker` for worker

## 3. Production topology

- `web` and `worker` run as separate processes/services
- embedded poller disabled in `web`
- readiness required before traffic

Readiness URLs:

- `GET /health/ready` on web
- `GET /health/ready` on worker

## 4. Monitoring minimum

Must be scraped and visible:

- `ingest_queue_depth`
- `ingest_dlq_depth`
- `ingest_job_events_total`
- `ingest_retry_delay_seconds`
- `ingest_queue_age_seconds`
- `ingest_processing_duration_seconds`
- `ingest_end_to_end_latency_seconds`

Verification:

```bash
curl -s http://localhost:18000/metrics | rg "ingest_(queue_depth|dlq_depth|job_events_total|retry_delay_seconds|queue_age_seconds|processing_duration_seconds|end_to_end_latency_seconds)"
```

Operational interpretation:

- queue depth -> backlog size
- queue age -> processing lag
- failed/dlq event counts -> fail rate and DLQ growth
- retry delay / retry events -> retry rate
- end-to-end latency -> user-visible ingest latency

## 5. Security minimum

- strong `ADMIN_API_KEY`
- admin audit logging enabled
- rotation procedure documented

Audit storage:

- `public.admin_audit_events`

## 6. Frontend ingest contract

Frontend must handle:

- `ingest_job_id`
- polling `GET /ingest/jobs/{job_id}` or `GET /files/{id}/processing`
- retry action for failed jobs
- requeue action for DLQ items
- readable error states

## 7. Quotas and limits

Must be configured and validated:

- ask rate limit
- upload rate limit
- per-user file count limit
- per-user storage limit
- mime/type guardrails
- max upload size

## 8. Data consistency minimum

Must be documented operationally:

- repeat upload idempotency via `content_hash`
- reindex procedure
- orphan upload cleanup
- orphan vector cleanup

## 9. Load test minimum

Run before go-live:

- peak uploads
- parallel workers
- worker restart under load
- partial failure recovery

## 10. Runbooks required before go-live

- queue grows
- DLQ grows
- Qdrant unavailable
- Supabase degraded
- safe purge/replay
