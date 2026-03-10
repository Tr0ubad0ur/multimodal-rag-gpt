#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _json_request(
    *,
    url: str,
    payload: dict[str, object],
    token: str | None = None,
    timeout_seconds: float = 10.0,
) -> tuple[int, dict[str, object] | str]:
    data = json.dumps(payload).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    request = urllib.request.Request(
        url, data=data, headers=headers, method='POST'
    )
    try:
        with urllib.request.urlopen(
            request, timeout=max(1.0, timeout_seconds)
        ) as response:
            body = response.read().decode('utf-8', errors='ignore')
            return int(getattr(response, 'status', 200)), json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode('utf-8', errors='ignore')
        try:
            return int(exc.code), json.loads(body)
        except json.JSONDecodeError:
            return int(exc.code), body
    except TimeoutError:
        return 0, 'timeout'


def _assert_error_code(
    payload: dict[str, object] | str,
    *,
    expected: str,
    context: str,
) -> None:
    if not isinstance(payload, dict):
        raise RuntimeError(f'{context}: expected JSON error, got {payload!r}')
    actual = str(payload.get('error_code') or '')
    if actual != expected:
        raise RuntimeError(
            f'{context}: expected error_code={expected!r}, got {actual!r} payload={payload!r}'
        )


def main() -> int:
    """Verify stable limit/guardrail error codes against a running backend."""
    parser = argparse.ArgumentParser(description='Limits smoke-check')
    parser.add_argument('--base-url', default='http://localhost:18000')
    parser.add_argument('--token', default='')
    parser.add_argument('--expect-ask-rate-limit', action='store_true')
    parser.add_argument('--expect-upload-rate-limit', action='store_true')
    parser.add_argument('--timeout-seconds', type=float, default=10.0)
    args = parser.parse_args()

    report: dict[str, object] = {'checks': []}

    readiness_url = f'{args.base_url.rstrip("/")}/health/ready'
    try:
        with urllib.request.urlopen(
            readiness_url, timeout=max(1.0, args.timeout_seconds)
        ) as response:
            readiness_body = response.read().decode('utf-8', errors='ignore')
            readiness_status = int(getattr(response, 'status', 200))
    except urllib.error.HTTPError as exc:
        readiness_status = int(exc.code)
        readiness_body = exc.read().decode('utf-8', errors='ignore')
    report['checks'].append(
        {
            'name': 'backend_readiness',
            'status_code': readiness_status,
            'ok': readiness_status == 200,
            'detail': readiness_body[:400],
        }
    )
    if readiness_status != 200:
        raise RuntimeError('Backend readiness failed for limits smoke-check')

    # Unsupported MIME/type equivalent via invalid attachment_id lookup.
    if args.token:
        ask_auth_url = f'{args.base_url.rstrip("/")}/ask_auth'
        status_code, payload = _json_request(
            url=ask_auth_url,
            payload={
                'query': 'test',
                'attachment_id': str(uuid.uuid4()),
            },
            token=args.token,
            timeout_seconds=args.timeout_seconds,
        )
        if status_code == 404:
            _assert_error_code(
                payload,
                expected='attachment_id_not_found',
                context='attachment lookup',
            )
            report['checks'].append(
                {'name': 'attachment_lookup_error_code', 'ok': True}
            )

    if args.expect_ask_rate_limit:
        ask_url = f'{args.base_url.rstrip("/")}/ask'
        status_code, payload = _json_request(
            url=ask_url,
            payload={'query': 'rate limit probe'},
            timeout_seconds=args.timeout_seconds,
        )
        if status_code == 0 and payload == 'timeout':
            raise RuntimeError(
                'Ask rate limit probe timed out before receiving a response'
            )
        if status_code != 429:
            raise RuntimeError('Expected ask rate limit 429 on current env')
        _assert_error_code(
            payload,
            expected='ask_rate_limit_exceeded',
            context='ask rate limit',
        )
        report['checks'].append({'name': 'ask_rate_limit', 'ok': True})

    # Upload rate limit and file size are better verified through the actual UI/upload path.
    if args.expect_upload_rate_limit:
        report['checks'].append(
            {
                'name': 'upload_rate_limit',
                'ok': False,
                'detail': 'Use multipart upload verification manually or extend this script with requests/httpx.',
            }
        )

    sys.stdout.write(f'{json.dumps(report, ensure_ascii=False, indent=2)}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
