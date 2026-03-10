from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from backend.utils.supabase_client import get_supabase_client


class AdminAuditService:
    """Persist admin API actions for traceability, fallback to no-op."""

    def __init__(self) -> None:
        """Initialize Supabase service client or disable audit persistence."""
        try:
            self.supabase = get_supabase_client(role='service')
        except Exception:
            self.supabase = None

    def log_event(
        self,
        *,
        action: str,
        actor: str,
        request_path: str,
        ip_address: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Write one audit event row if audit storage is available."""
        if self.supabase is None:
            return
        try:
            (
                self.supabase.table('admin_audit_events')
                .insert(
                    {
                        'action': action,
                        'actor': actor,
                        'request_path': request_path,
                        'ip_address': ip_address,
                        'details': details or {},
                        'created_at': datetime.now(timezone.utc).isoformat(),
                    }
                )
                .execute()
            )
        except Exception:
            return
