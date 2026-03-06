#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean


def _build_request(
    *,
    url: str,
    query: str,
    top_k: int,
    model: str | None,
    bearer_token: str | None,
) -> urllib.request.Request:
    payload = {'query': query, 'top_k': top_k}
    if model:
        payload['model'] = model
    data = json.dumps(payload).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    if bearer_token:
        headers['Authorization'] = f'Bearer {bearer_token}'
    return urllib.request.Request(
        url=url,
        data=data,
        headers=headers,
        method='POST',
    )


def _one_request(
    *,
    url: str,
    query: str,
    top_k: int,
    model: str | None,
    bearer_token: str | None,
    timeout_seconds: float,
) -> tuple[bool, float, int, str]:
    started = time.perf_counter()
    request = _build_request(
        url=url,
        query=query,
        top_k=top_k,
        model=model,
        bearer_token=bearer_token,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as resp:
            _ = resp.read()
            status_code = int(getattr(resp, 'status', 200))
            latency = time.perf_counter() - started
            return 200 <= status_code < 300, latency, status_code, ''
    except urllib.error.HTTPError as exc:
        latency = time.perf_counter() - started
        return False, latency, int(exc.code), str(exc)
    except Exception as exc:  # noqa: BLE001
        latency = time.perf_counter() - started
        return False, latency, 0, str(exc)


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int((len(sorted_values) - 1) * ratio)
    return sorted_values[idx]


def _out(line: str) -> None:
    sys.stdout.write(f'{line}\n')


def main() -> None:
    """Run concurrent load test for /ask or /ask_auth endpoint."""
    parser = argparse.ArgumentParser(description='Concurrent ASK load test')
    parser.add_argument('--base-url', default='http://localhost:8000')
    parser.add_argument('--endpoint', default='/ask_auth')
    parser.add_argument('--token', default='')
    parser.add_argument('--query', default='Кратко опиши содержание контекста')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--model', default='')
    parser.add_argument('--requests', type=int, default=200)
    parser.add_argument('--concurrency', type=int, default=20)
    parser.add_argument('--timeout-seconds', type=float, default=30.0)
    args = parser.parse_args()

    url = f'{args.base_url.rstrip("/")}/{args.endpoint.lstrip("/")}'
    total = max(1, args.requests)
    concurrency = max(1, args.concurrency)

    latencies: list[float] = []
    status_codes: dict[int, int] = {}
    ok_count = 0
    lock = threading.Lock()
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _one_request,
                url=url,
                query=args.query,
                top_k=max(1, args.top_k),
                model=(args.model or None),
                bearer_token=(args.token or None),
                timeout_seconds=max(1.0, args.timeout_seconds),
            )
            for _ in range(total)
        ]
        for future in as_completed(futures):
            ok, latency, status_code, _error = future.result()
            with lock:
                latencies.append(latency)
                status_codes[status_code] = (
                    status_codes.get(status_code, 0) + 1
                )
                if ok:
                    ok_count += 1

    duration = max(0.001, time.perf_counter() - started)
    _out(f'url={url}')
    _out(f'total_requests={total}')
    _out(f'concurrency={concurrency}')
    _out(f'duration_seconds={duration:.3f}')
    _out(f'success={ok_count}')
    _out(f'errors={total - ok_count}')
    _out(f'rps={total / duration:.2f}')
    _out(f'latency_avg_ms={mean(latencies) * 1000:.1f}')
    _out(f'latency_p50_ms={_percentile(latencies, 0.50) * 1000:.1f}')
    _out(f'latency_p95_ms={_percentile(latencies, 0.95) * 1000:.1f}')
    _out(f'latency_p99_ms={_percentile(latencies, 0.99) * 1000:.1f}')
    _out(f'status_codes={json.dumps(status_codes, sort_keys=True)}')


if __name__ == '__main__':
    main()
