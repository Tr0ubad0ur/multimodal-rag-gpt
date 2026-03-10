# Admin API Key Rotation

## Goal

Rotate `ADMIN_API_KEY` without leaving admin endpoints exposed or undocumented.

## Minimum policy

- use a long random secret
- store it only in deployment secret storage
- never commit it to the repo
- rotate on suspected leak or scheduled maintenance window

## Rotation procedure

1. Generate a new key

Example:

```bash
python - <<'PY'
import secrets
print(secrets.token_urlsafe(48))
PY
```

2. Update deployment secret store

- replace `ADMIN_API_KEY` for the web runtime
- keep `worker` unchanged if it does not expose admin routes

3. Restart web service

- restart only the web process/service
- verify readiness after restart

4. Validate admin access

- call one admin endpoint with the new key
- confirm the old key no longer works

## Verification checklist

- `GET /health/ready` returns ready after restart
- admin request with new `X-Admin-Key` succeeds
- admin request with old `X-Admin-Key` fails with `403`
- audit log still records admin actions

## Incident response

Rotate immediately if:

- key was shared in chat/email
- key appeared in logs
- key was committed locally
- access to the deployment secret store is suspected compromised
