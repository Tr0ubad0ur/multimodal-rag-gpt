import os
from functools import lru_cache

from dotenv import load_dotenv

from supabase import Client, create_client

load_dotenv()


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f'Missing environment variable: {name}')
    return value


@lru_cache(maxsize=2)
def get_supabase_client(role: str = 'service') -> Client:
    """Create a cached Supabase client for the given role."""
    url = _require('SUPABASE_URL')
    if role == 'anon':
        key = _require('SUPABASE_ANON_KEY')
    else:
        key = _require('SUPABASE_SERVICE_ROLE_KEY')
    return create_client(url, key)
