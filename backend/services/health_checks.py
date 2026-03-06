from __future__ import annotations

from qdrant_client import QdrantClient

from backend.services.ingest_jobs import IngestJobsService
from backend.services.runtime_config import RuntimeMode, validate_runtime_env
from backend.utils.config_handler import Config
from backend.utils.supabase_client import get_supabase_client


def check_supabase() -> tuple[bool, str]:
    """Check Supabase service connectivity."""
    try:
        supabase = get_supabase_client(role='service')
        _ = supabase.table('ingest_jobs').select('id').limit(1).execute()
        return True, 'ok'
    except Exception as exc:
        return False, str(exc)


def check_qdrant() -> tuple[bool, str]:
    """Check Qdrant connectivity."""
    try:
        client = QdrantClient(url=Config.qdrant_url)
        _ = client.get_collections()
        return True, 'ok'
    except Exception as exc:
        return False, str(exc)


def check_ingest_queue() -> tuple[bool, str]:
    """Check ingest queue accessibility in Supabase."""
    try:
        service = IngestJobsService()
        _ = service.list_queued_jobs(limit=1)
        return True, 'ok'
    except Exception as exc:
        return False, str(exc)


def check_dependencies(*, mode: RuntimeMode) -> dict[str, dict[str, object]]:
    """Return dependency status map for readiness endpoints."""
    env_check = validate_runtime_env(mode=mode)
    supabase_ok, supabase_msg = check_supabase()
    qdrant_ok, qdrant_msg = check_qdrant()
    queue_ok, queue_msg = check_ingest_queue()
    return {
        'config': {'ok': bool(env_check['ok']), 'detail': env_check},
        'supabase': {'ok': supabase_ok, 'detail': supabase_msg},
        'qdrant': {'ok': qdrant_ok, 'detail': qdrant_msg},
        'ingest_queue': {'ok': queue_ok, 'detail': queue_msg},
    }
