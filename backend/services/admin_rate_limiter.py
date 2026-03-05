from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock

from backend.services.admin_rate_limit import AdminRateLimitService


class AdminRateLimiter:
    """Hierarchical admin rate limiter: DB RPC -> Redis -> in-memory."""

    def __init__(self) -> None:
        """Initialize limiter backends and fallback state."""
        self._db = AdminRateLimitService()
        self._memory_lock = Lock()
        self._memory_buckets: dict[str, deque[float]] = defaultdict(deque)
        self._redis_client = self._build_redis_client()

    def is_allowed(
        self,
        *,
        scope: str,
        limit: int,
        window_seconds: int = 60,
    ) -> bool:
        """Return whether request is allowed for the given scope."""
        if limit <= 0:
            return True

        db_allowed = self._db.is_allowed(
            scope=scope,
            limit=limit,
            window_seconds=window_seconds,
        )
        if db_allowed is not None:
            return db_allowed

        redis_allowed = self._check_redis(
            scope=scope,
            limit=limit,
            window_seconds=window_seconds,
        )
        if redis_allowed is not None:
            return redis_allowed

        return self._check_memory(
            scope=scope,
            limit=limit,
            window_seconds=window_seconds,
        )

    def _check_memory(
        self,
        *,
        scope: str,
        limit: int,
        window_seconds: int,
    ) -> bool:
        now = time.monotonic()
        with self._memory_lock:
            q = self._memory_buckets[scope]
            while q and (now - q[0]) > window_seconds:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
        return True

    def _check_redis(
        self,
        *,
        scope: str,
        limit: int,
        window_seconds: int,
    ) -> bool | None:
        client = self._redis_client
        if client is None:
            return None
        bucket = int(time.time() // window_seconds)
        key = f'admin_rate_limit:{scope}:{bucket}'
        ttl_seconds = max(window_seconds * 2, 60)
        try:
            pipe = client.pipeline()
            pipe.incr(key)
            pipe.expire(key, ttl_seconds)
            count, _ = pipe.execute()
            return int(count) <= limit
        except Exception:
            return None

    def _build_redis_client(self):
        import os

        redis_url = (os.getenv('REDIS_URL') or '').strip()
        if not redis_url:
            return None
        try:
            import redis  # type: ignore

            return redis.Redis.from_url(redis_url)
        except Exception:
            return None
