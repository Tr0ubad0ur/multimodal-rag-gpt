# Limits and Guardrails Policy

## Goals

- protect backend from abuse and accidental overload
- keep ingest/query latency predictable
- provide stable frontend-facing error codes

## 1. Request rate limits

### Ask

- `ASK_RATE_LIMIT_PER_MINUTE_AUTH`
- `ASK_RATE_LIMIT_PER_MINUTE_GUEST`

Error code:

- `ask_rate_limit_exceeded`

### Upload

- `UPLOAD_RATE_LIMIT_PER_MINUTE_AUTH`
- `UPLOAD_RATE_LIMIT_PER_MINUTE_GUEST`

Error code:

- `upload_rate_limit_exceeded`

## 2. Per-user capacity limits

### File count

- `MAX_FILES_PER_USER`

Error code:

- `user_file_quota_exceeded`

### Storage size

- `MAX_STORAGE_BYTES_PER_USER`

Error code:

- `user_storage_quota_exceeded`

### Folder upload batch size

- `MAX_FILES_PER_FOLDER_UPLOAD`

Current behavior:

- request is rejected if too many files are uploaded in one folder request

## 3. Upload guardrails

### Max upload size

- `MAX_UPLOAD_SIZE_BYTES`

Error code:

- `file_is_too_large`

### MIME/type restrictions

Allowed classes:

- text
- pdf/docx
- image
- video

Error code:

- `unsupported_mime_type`

### Missing filename

Error code:

- `filename_is_required`

## 4. Auth and admin access

Stable error codes:

- `missing_or_invalid_authorization_header`
- `invalid_or_expired_token`
- `invalid_email_or_password`
- `user_already_registered`
- `authentication_failed`
- `admin_api_not_configured`
- `invalid_admin_key`
- `admin_rate_limit_exceeded`

## 5. Ingest retry/requeue errors

Stable error codes:

- `ingest_job_not_found`
- `only_user_owned_jobs_can_be_retried`
- `only_failed_jobs_can_be_retried`
- `ingest_dlq_item_not_found`
- `ingest_dlq_item_not_requeueable`

## 6. Operational verification

Before production:

1. verify env values are set
2. verify upload size rejection
3. verify unsupported MIME rejection
4. verify ask/upload rate limiting
5. verify quota overrun behavior
6. verify frontend maps `error_code` instead of free-form text

Suggested command:

```bash
uv run python scripts/smoke_limits.py --base-url http://localhost:18000
```

If you temporarily lower ask rate limit in env, you can verify the 429 contract:

```bash
uv run python scripts/smoke_limits.py \
  --base-url http://localhost:18000 \
  --expect-ask-rate-limit
```
