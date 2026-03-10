# Data Consistency Policy

## Goals

- no duplicate vectors for repeat uploads
- predictable reindex flow
- explicit orphan cleanup rules
- clear operator action when Supabase and Qdrant drift

## 1. Repeat upload idempotency

- persisted user uploads are deduplicated by `content_hash` within the same folder
- if the same file is uploaded again into the same folder:
  - the existing file record is reused
  - no new vectors are created
  - response returns `deduplicated=true`

## 2. Reindex policy

Use reindex when:

- embedding model changes
- chunking logic changes
- metadata mapping changes
- vectors are missing or corrupted

Operational rule:

- schedule reindex through admin consistency API
- worker deletes existing vectors for the file first
- worker writes a fresh set of vectors

This makes reindex idempotent.

## 3. Orphan cleanup rules

### Orphan uploads

Definition:

- local files under `UPLOADS_DIR` not referenced by `kb_files`

Action:

- run cleanup in dry-run mode first
- only delete files older than the minimum age threshold

### Orphan vectors

Definition:

- Qdrant points whose `file_id` no longer exists in `kb_files`

Action:

- identify orphan `file_id`s per collection
- delete vectors only after dry-run verification

### Missing storage records

Definition:

- DB file records pointing to local files that no longer exist

Action:

- delete vectors for the file
- delete the stale DB file record

## 4. Partial failure policy

Current model:

- Supabase is the source of truth for files and ingest jobs
- Qdrant is the derived index

If Supabase and Qdrant diverge:

- prefer repairing Qdrant from Supabase state
- use reindex before manual deletion
- use orphan cleanup only after confirming the canonical state in Supabase

## 5. Operator procedures

- before destructive cleanup: run dry-run first
- before large reindex: monitor queue depth, lag, and DLQ
- after cleanup/reindex: verify `/health/ready` and ingest metrics

## 6. Known limitation

- there is no strict transactional saga/outbox between Supabase and Qdrant yet
- consistency is maintained operationally through idempotency, reindex, and cleanup
