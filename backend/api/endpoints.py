from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from backend.core.multimodal_rag import LocalRAG
from backend.utils.supabase_client import get_supabase_client

router = APIRouter()
rag = LocalRAG()


class QueryRequest(BaseModel):
    """Schema for user query requests.

    Attributes:
        query (str): The text query from the user.
        top_k (int, optional): Number of top documents to retrieve from RAG. Defaults to 5.
        image (Optional[str], optional): Optional path or URL to an image to include in the query. Defaults to None.
    """

    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)
    image: Optional[str] = None

    @field_validator('query')
    @classmethod
    def validate_query(cls, value: str) -> str:
        """validate_query."""
        stripped = value.strip()
        if not stripped:
            raise ValueError('query must not be empty')
        return stripped


class AuthRequest(BaseModel):
    """Request body for sign up / sign in."""

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=6, max_length=128)


class RefreshRequest(BaseModel):
    """Request body for token refresh."""

    refresh_token: str = Field(min_length=10, max_length=2048)


class LogoutRequest(BaseModel):
    """Request body for logout."""

    scope: str = Field(default='global')


def _serialize(obj):
    """Serialize Supabase SDK responses into plain dicts."""
    if obj is None:
        return None
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    if hasattr(obj, 'dict'):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return obj


def get_access_token(authorization: str = Header(None)) -> str:
    """Extract bearer token from Authorization header."""
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Missing or invalid Authorization header',
        )
    return authorization.split(' ', 1)[1].strip()


def get_current_user(token: str = Depends(get_access_token)) -> dict:
    """Resolve current user from Supabase using access token."""
    supabase = get_supabase_client(role='anon')
    user_resp = supabase.auth.get_user(token)
    user = getattr(user_resp, 'user', None)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid or expired token',
        )
    return _serialize(user)


@router.post('/auth/signup')
def signup(request: AuthRequest) -> dict:
    """Sign up a user with email/password via Supabase."""
    supabase = get_supabase_client(role='anon')
    resp = supabase.auth.sign_up(
        {'email': request.email, 'password': request.password}
    )
    return {
        'user': _serialize(getattr(resp, 'user', None)),
        'session': _serialize(getattr(resp, 'session', None)),
    }


@router.post('/auth/signin')
def signin(request: AuthRequest) -> dict:
    """Sign in a user with email/password via Supabase."""
    supabase = get_supabase_client(role='anon')
    resp = supabase.auth.sign_in_with_password(
        {'email': request.email, 'password': request.password}
    )
    return {
        'user': _serialize(getattr(resp, 'user', None)),
        'session': _serialize(getattr(resp, 'session', None)),
    }


@router.get('/auth/me')
def me(user: Annotated[dict, Depends(get_current_user)]) -> dict:
    """Return the current authenticated user."""
    return user


@router.post('/auth/refresh')
def refresh(request: RefreshRequest) -> dict:
    """Refresh access token using a refresh token."""
    supabase = get_supabase_client(role='anon')
    resp = supabase.auth.refresh_session(request.refresh_token)
    return {
        'user': _serialize(getattr(resp, 'user', None)),
        'session': _serialize(getattr(resp, 'session', None)),
    }


@router.post('/auth/logout')
def logout(
    request: LogoutRequest,
    token: Annotated[str, Depends(get_access_token)],
) -> dict:
    """Revoke refresh tokens for the user and clear sessions."""
    supabase = get_supabase_client(role='service')
    supabase.auth.admin.sign_out(token, request.scope)
    return {'ok': True}


@router.post('/ask')
def ask_mixed(request: QueryRequest) -> dict:
    """Handle a multimodal query request (text + optional image) and generate an answer using RAG.

    Args:
        request (QueryRequest): The query request containing the text, optional image, and retrieval parameters.

    Returns:
        dict: The generated answer from the RAG pipeline, potentially considering the image.
    """
    result = rag.generate_answer(
        request.query, top_k=request.top_k, image=request.image
    )
    return result


@router.post('/ask_auth')
def ask_mixed_auth(
    request: QueryRequest,
    user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """Run RAG with user auth and persist query history."""
    result = rag.generate_answer(
        request.query,
        top_k=request.top_k,
        image=request.image,
        user_id=user['id'],
    )

    supabase = get_supabase_client(role='service')
    supabase.table('query_history').insert(
        {
            'user_id': user['id'],
            'query': request.query,
            'answer': result.get('answer', ''),
            'retrieved_docs': result.get('retrieved_docs', []),
        }
    ).execute()

    return result


@router.get('/history')
def get_history(user: Annotated[dict, Depends(get_current_user)]) -> dict:
    """Return latest query history for the current user."""
    supabase = get_supabase_client(role='service')
    resp = (
        supabase.table('query_history')
        .select('*')
        .eq('user_id', user['id'])
        .order('created_at', desc=True)
        .limit(50)
        .execute()
    )
    return {'data': getattr(resp, 'data', None)}


@router.delete('/history/{item_id}')
def delete_history(
    item_id: int, user: Annotated[dict, Depends(get_current_user)]
) -> dict:
    """Delete a single history item for the current user."""
    supabase = get_supabase_client(role='service')
    (
        supabase.table('query_history')
        .delete()
        .eq('id', item_id)
        .eq('user_id', user['id'])
        .execute()
    )
    return {'ok': True}


# TODO write this requests
# @router.post('/test_llm')
# @router.post('/test_qdrant')
