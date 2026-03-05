from __future__ import annotations

from backend.utils.supabase_client import get_supabase_client


class AdminRateLimitService:
    """Cross-instance admin rate limiter backed by Supabase RPC."""

    def __init__(self) -> None:
        """Initialize Supabase service client or disable DB limiter."""
        try:
            self.supabase = get_supabase_client(role='service')
        except Exception:
            self.supabase = None

    def is_allowed(
        self,
        *,
        scope: str,
        limit: int,
        window_seconds: int = 60,
    ) -> bool | None:
        """Return True/False when DB limiter is available, otherwise None."""
        if self.supabase is None:
            return None
        try:
            resp = self.supabase.rpc(
                'check_admin_rate_limit',
                {
                    'p_scope': scope,
                    'p_limit': max(1, int(limit)),
                    'p_window_seconds': max(1, int(window_seconds)),
                },
            ).execute()
            return bool(getattr(resp, 'data', False))
        except Exception:
            return None
