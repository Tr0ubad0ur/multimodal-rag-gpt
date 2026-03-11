import os
import uuid
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

from backend.core.embeddings import (
    image_embedding_from_path,
    text_embedding,
    video_embedding_from_path,
)
from backend.core.multimodal_rag import LocalRAG
from backend.services.admin_audit import AdminAuditService
from backend.services.admin_rate_limiter import AdminRateLimiter
from backend.services.data_consistency import DataConsistencyService
from backend.services.ingest import IngestService
from backend.services.ingest_jobs import IngestJobsService
from backend.services.ingest_worker import IngestWorker
from backend.services.kb import KBService
from backend.services.request_rate_limiter import RequestRateLimiter
from backend.services.storage import delete_stored_file, save_upload_file
from backend.utils.config_handler import Config
from backend.utils.supabase_client import get_supabase_client

router = APIRouter()
rag = LocalRAG()
REQUIRED_UPLOAD_FILE = File(...)
OPTIONAL_UPLOAD_FILE = File(default=None)
REQUIRED_UPLOAD_FILES = File(...)
OPTIONAL_RELATIVE_PATHS = Form(default=None)
_ADMIN_RATE_LIMITER = AdminRateLimiter()
_ADMIN_AUDIT = AdminAuditService()
_REQUEST_RATE_LIMITER = RequestRateLimiter()

DEFAULT_MAX_FILES_PER_USER = 2000
DEFAULT_MAX_STORAGE_BYTES_PER_USER = 10 * 1024 * 1024 * 1024
DEFAULT_MAX_FILES_PER_FOLDER_UPLOAD = 100
DEFAULT_ASK_RATE_LIMIT_PER_MINUTE_AUTH = 120
DEFAULT_ASK_RATE_LIMIT_PER_MINUTE_GUEST = 40
DEFAULT_UPLOAD_RATE_LIMIT_PER_MINUTE_AUTH = 60
DEFAULT_UPLOAD_RATE_LIMIT_PER_MINUTE_GUEST = 20
IMAGE_MIME_TYPES = {'image/jpeg', 'image/png', 'image/webp'}


def _api_error(
    *,
    status_code: int,
    detail: str,
    error_code: str,
    **extra: Any,
) -> HTTPException:
    """Build API error with stable frontend-facing error code."""
    payload: dict[str, Any] = {
        'detail': detail,
        'error_code': error_code,
    }
    payload.update(extra)
    return HTTPException(status_code=status_code, detail=payload)


def _to_auth_http_exception(exc: Exception) -> HTTPException:
    """Map Supabase auth exceptions to user-facing HTTP errors."""
    message = str(exc)
    normalized = message.lower()
    if 'invalid login credentials' in normalized:
        return _api_error(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid email or password',
            error_code='invalid_email_or_password',
        )
    if 'user already registered' in normalized:
        return _api_error(
            status_code=status.HTTP_409_CONFLICT,
            detail='User already registered',
            error_code='user_already_registered',
        )
    return _api_error(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=message or 'Authentication failed',
        error_code='authentication_failed',
    )


class QueryRequest(BaseModel):
    """Schema for user query requests."""

    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)
    image: Optional[str] = None
    model: Optional[str] = Field(default=None, max_length=256)
    attachment_id: Optional[str] = Field(default=None, max_length=64)
    guest_session_id: Optional[str] = Field(default=None, max_length=128)
    folder_ids: list[str] = Field(default_factory=list)
    file_ids: list[str] = Field(default_factory=list)

    @field_validator('query')
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Ensure query is not empty after trimming."""
        stripped = value.strip()
        if not stripped:
            raise ValueError('query must not be empty')
        return stripped


class TextEmbeddingRequest(BaseModel):
    """Schema for text embedding endpoint."""

    text: str = Field(min_length=1, max_length=4000)
    provider: str = Field(
        default=Config.default_embedding_provider, min_length=1
    )

    @field_validator('text')
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Ensure non-empty text after stripping."""
        stripped = value.strip()
        if not stripped:
            raise ValueError('text must not be empty')
        return stripped


class ImageEmbeddingRequest(BaseModel):
    """Schema for image embedding endpoint."""

    image_path: str = Field(min_length=1, max_length=2048)
    provider: str = Field(
        default=Config.default_embedding_provider, min_length=1
    )

    @field_validator('image_path')
    @classmethod
    def validate_image_path(cls, value: str) -> str:
        """Ensure image path exists and points to a file."""
        image_path = Path(value)
        if not image_path.exists() or not image_path.is_file():
            raise ValueError('image_path must point to an existing file')
        return str(image_path)


class VideoEmbeddingRequest(BaseModel):
    """Schema for video embedding endpoint."""

    video_path: str = Field(min_length=1, max_length=2048)
    sample_fps: float = Field(
        default=Config.embedding_video_sample_fps, gt=0, le=10
    )
    provider: str = Field(
        default=Config.default_embedding_provider, min_length=1
    )

    @field_validator('video_path')
    @classmethod
    def validate_video_path(cls, value: str) -> str:
        """Ensure video path exists and points to a file."""
        video_path = Path(value)
        if not video_path.exists() or not video_path.is_file():
            raise ValueError('video_path must point to an existing file')
        return str(video_path)


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


class KBFolderCreateRequest(BaseModel):
    """Create KB folder request."""

    name: str = Field(min_length=1, max_length=255)
    parent_id: str | None = Field(default=None)

    @field_validator('name')
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Ensure folder name is not empty after trimming."""
        stripped = value.strip()
        if not stripped:
            raise ValueError('name must not be empty')
        return stripped


class KBFileAttachRequest(BaseModel):
    """Attach uploaded file to folder request."""

    file_id: str = Field(min_length=1, max_length=64)
    folder_id: str | None = Field(default=None, max_length=64)


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
        raise _api_error(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Missing or invalid Authorization header',
            error_code='missing_or_invalid_authorization_header',
        )
    return authorization.split(' ', 1)[1].strip()


def get_admin_access(x_admin_key: str = Header(default='')) -> bool:
    """Protect admin-only endpoints with static admin key."""
    admin_key = (os.getenv('ADMIN_API_KEY') or '').strip()
    if not admin_key:
        raise _api_error(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Admin API is not configured',
            error_code='admin_api_not_configured',
        )
    if x_admin_key != admin_key:
        raise _api_error(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Invalid admin key',
            error_code='invalid_admin_key',
        )
    _enforce_admin_rate_limit()
    return True


def _enforce_admin_rate_limit() -> None:
    limit = int(os.getenv('ADMIN_RATE_LIMIT_PER_MINUTE', '60'))
    if not _ADMIN_RATE_LIMITER.is_allowed(
        scope='admin_global',
        limit=limit,
        window_seconds=60,
    ):
        raise _api_error(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail='Admin rate limit exceeded',
            error_code='admin_rate_limit_exceeded',
        )


def _audit_admin_action(
    *,
    request: Request,
    action: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Persist audit event for an admin API action."""
    _ADMIN_AUDIT.log_event(
        action=action,
        actor='admin_api_key',
        request_path=request.url.path,
        ip_address=request.client.host if request.client else None,
        details=details,
    )


def _env_int(name: str, default: int, *, min_value: int = 1) -> int:
    raw = (os.getenv(name) or '').strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise _api_error(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Invalid integer for {name}',
            error_code='invalid_runtime_integer',
            config_key=name,
        ) from exc
    return max(min_value, value)


def _client_identifier(request: Request) -> str:
    forwarded = (request.headers.get('x-forwarded-for') or '').strip()
    if forwarded:
        return forwarded.split(',')[0].strip()
    return getattr(request.client, 'host', 'unknown') or 'unknown'


def _enforce_request_rate_limit(
    *, scope: str, limit: int, message: str
) -> None:
    if not _REQUEST_RATE_LIMITER.is_allowed(
        scope=scope,
        limit=limit,
        window_seconds=60,
    ):
        raise _api_error(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=message,
            error_code=message.strip().lower().replace(' ', '_'),
        )


def _enforce_quota_capacity(
    *,
    total_files_after: int,
    total_size_after: int,
) -> None:
    max_files = _env_int(
        'MAX_FILES_PER_USER',
        DEFAULT_MAX_FILES_PER_USER,
    )
    max_storage = _env_int(
        'MAX_STORAGE_BYTES_PER_USER',
        DEFAULT_MAX_STORAGE_BYTES_PER_USER,
    )
    if total_files_after > max_files:
        raise _api_error(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail='User file quota exceeded',
            error_code='user_file_quota_exceeded',
        )
    if total_size_after > max_storage:
        raise _api_error(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail='User storage quota exceeded',
            error_code='user_storage_quota_exceeded',
        )


def get_current_user(token: str = Depends(get_access_token)) -> dict:
    """Resolve current user from Supabase using access token."""
    try:
        supabase = get_supabase_client(role='anon')
        user_resp = supabase.auth.get_user(token)
        user = getattr(user_resp, 'user', None)
    except Exception:
        user = None

    if user is None:
        raise _api_error(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid or expired token',
            error_code='invalid_or_expired_token',
        )
    return _serialize(user)


def _kb_service() -> KBService:
    return KBService()


def _ingest_service() -> IngestService:
    return IngestService()


def _ingest_jobs_service() -> IngestJobsService:
    return IngestJobsService()


def _ingest_worker() -> IngestWorker:
    return IngestWorker(max_attempts=3)


def _consistency_service() -> DataConsistencyService:
    return DataConsistencyService()


def _enqueue_ingest_job(
    *,
    background_tasks: BackgroundTasks,
    file_id: str,
    file_path: str,
    filename: str,
    mime: str,
    user_id: str | None = None,
    guest_session_id: str | None = None,
    source_path: str | None = None,
    folder_id: str | None = None,
    folder_path: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    owner_id = user_id or guest_session_id
    if not owner_id:
        raise ValueError('Either user_id or guest_session_id must be provided')
    owner_type = 'user' if user_id else 'guest'

    jobs = _ingest_jobs_service()
    effective_metadata = dict(metadata or {})
    if folder_path is not None:
        effective_metadata.setdefault('folder_path', folder_path)
    effective_metadata.setdefault('storage_path', file_path)
    job_id = jobs.create_job(
        owner_type=owner_type,
        owner_id=owner_id,
        user_id=user_id,
        file_id=file_id,
        filename=filename,
        mime=mime,
        source_path=source_path,
        folder_id=folder_id,
        metadata=effective_metadata,
    )
    if job_id and user_id:
        worker = _ingest_worker()
        background_tasks.add_task(
            worker.process_job,
            job_id=job_id,
            user_id=user_id,
        )
    elif user_id:
        # Safe fallback when jobs table is unavailable: ingest immediately.
        _ingest_with_job(
            file_id=file_id,
            file_path=file_path,
            filename=filename,
            mime=mime,
            user_id=user_id,
            folder_id=folder_id,
            folder_path=folder_path,
            source_path=source_path,
            metadata=metadata,
        )
    return job_id


def _ingest_with_job(
    *,
    file_id: str,
    file_path: str,
    filename: str,
    mime: str,
    user_id: str | None = None,
    guest_session_id: str | None = None,
    folder_id: str | None = None,
    folder_name: str | None = None,
    folder_path: str | None = None,
    source_path: str | None = None,
    metadata: dict[str, Any] | None = None,
    existing_job_id: str | None = None,
) -> None:
    owner_id = user_id or guest_session_id
    if not owner_id:
        raise ValueError('Either user_id or guest_session_id must be provided')
    owner_type = 'user' if user_id else 'guest'

    jobs = _ingest_jobs_service()
    ingest = _ingest_service()
    effective_metadata = dict(metadata or {})
    if folder_path is not None:
        effective_metadata.setdefault('folder_path', folder_path)
    effective_metadata.setdefault('storage_path', file_path)
    job_id = existing_job_id
    if job_id is None:
        job_id = jobs.create_job(
            owner_type=owner_type,
            owner_id=owner_id,
            user_id=user_id,
            file_id=file_id,
            filename=filename,
            mime=mime,
            source_path=source_path,
            folder_id=folder_id,
            metadata=effective_metadata,
        )
        jobs.mark_processing(job_id=job_id, attempt=1)
    else:
        existing = jobs.get_job(job_id=job_id, user_id=user_id)
        next_attempt = int(existing.get('attempt') or 0) + 1 if existing else 1
        jobs.mark_processing(job_id=job_id, attempt=next_attempt)
    try:
        ingest.ingest_file(
            file_id=file_id,
            file_path=file_path,
            filename=filename,
            mime=mime,
            user_id=user_id,
            guest_session_id=guest_session_id,
            folder_id=folder_id,
            folder_name=folder_name,
            folder_path=folder_path,
            source_path=source_path,
        )
    except Exception as exc:
        jobs.mark_failed(job_id=job_id, error=str(exc))
        raise
    jobs.mark_completed(job_id=job_id)


def _normalize_relative_path(relative_path: str, fallback_name: str) -> str:
    """Normalize user-supplied relative path for folder upload."""
    raw = (relative_path or '').replace('\\', '/').strip()
    normalized = raw.strip('/')
    if not normalized:
        normalized = fallback_name

    parsed = PurePosixPath(normalized)
    parts = [part for part in parsed.parts if part not in {'', '.'}]
    if not parts or any(part == '..' for part in parts):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                'detail': f'Invalid relative path: {relative_path!r}',
                'error_code': 'invalid_relative_path',
            },
        )
    return '/'.join(parts)


async def _parse_auth_request(
    request: Request,
    email_form: str | None,
    password_form: str | None,
) -> AuthRequest:
    """Parse auth payload from JSON or form-encoded requests."""
    content_type = (request.headers.get('content-type') or '').lower()
    if 'application/json' in content_type:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    'detail': 'JSON body must be an object',
                    'error_code': 'json_body_must_be_object',
                },
            )
        email = (
            payload.get('email')
            or payload.get('login')
            or payload.get('username')
        )
        password = payload.get('password') or payload.get('pass')
    else:
        email = email_form
        password = password_form

    try:
        return AuthRequest(email=email, password=password)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


async def _parse_query_request(
    request: Request,
    query_form: str | None,
    top_k_form: int,
    image_form: str | None,
    model_form: str | None,
    attachment_id_form: str | None,
    guest_session_id_form: str | None,
    folder_ids_form: str | None,
    file_ids_form: str | None,
) -> QueryRequest:
    """Parse query payload from JSON or multipart form."""
    content_type = (request.headers.get('content-type') or '').lower()
    if 'application/json' in content_type:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    'detail': 'JSON body must be an object',
                    'error_code': 'json_body_must_be_object',
                },
            )
        try:
            return QueryRequest(**payload)
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from exc

    folder_ids = []
    if folder_ids_form:
        folder_ids = [
            f.strip() for f in folder_ids_form.split(',') if f.strip()
        ]
    file_ids = []
    if file_ids_form:
        file_ids = [f.strip() for f in file_ids_form.split(',') if f.strip()]

    payload = {
        'query': query_form,
        'top_k': top_k_form,
        'image': image_form,
        'model': model_form,
        'attachment_id': attachment_id_form,
        'guest_session_id': guest_session_id_form,
        'folder_ids': folder_ids,
        'file_ids': file_ids,
    }
    try:
        return QueryRequest(**payload)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


def _resolve_attachment(
    *,
    attachment_id: str,
    user_id: str | None,
) -> dict[str, Any]:
    supabase = get_supabase_client(role='service')
    query = supabase.table('kb_files').select('*').eq('id', attachment_id)
    if user_id:
        query = query.eq('user_id', user_id)
    resp = query.limit(1).execute()
    data = getattr(resp, 'data', None) or []
    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                'detail': 'attachment_id not found',
                'error_code': 'attachment_id_not_found',
            },
        )
    return data[0]


async def _prepare_attachment_data(
    *,
    request_payload: QueryRequest,
    attachment_file: UploadFile | None,
    user_id: str | None,
    guest_session_id: str | None = None,
) -> tuple[list[dict[str, str]], str | None, str | None, str | None]:
    """Prepare extra context docs from attachment id or direct multipart file."""
    ingest = _ingest_service()

    if attachment_file is not None:
        stored = await save_upload_file(attachment_file)
        if user_id:
            kb = _kb_service()
            usage = kb.get_user_storage_usage(user_id=user_id)
            _enforce_quota_capacity(
                total_files_after=usage['total_files'] + 1,
                total_size_after=usage['total_size'] + stored.size,
            )
            existing = kb.find_existing_file_by_hash(
                user_id=user_id,
                folder_id=None,
                content_hash=stored.content_hash,
            )
            if existing:
                delete_stored_file(stored.storage_path)
                return [], None, existing['id']
            kb.create_uploaded_file_record(
                user_id=user_id,
                file_id=stored.file_id,
                filename=stored.filename,
                mime=stored.mime,
                size=stored.size,
                storage_path=stored.storage_path,
                content_hash=stored.content_hash,
                folder_id=None,
            )
        attachment_context_chunks = ingest.extract_attachment_context(
            stored.storage_path, stored.mime
        )
        if user_id:
            _ingest_with_job(
                file_id=stored.file_id,
                file_path=stored.storage_path,
                filename=stored.filename,
                mime=stored.mime,
                user_id=user_id,
                folder_id=None,
                folder_name=None,
                folder_path='root',
                source_path=stored.filename,
                metadata={'origin': 'chat_attachment', 'transient': False},
            )
        else:
            _ingest_with_job(
                file_id=stored.file_id,
                file_path=stored.storage_path,
                filename=stored.filename,
                mime=stored.mime,
                user_id=None,
                guest_session_id=guest_session_id,
                folder_id=None,
                folder_name=None,
                folder_path='root',
                source_path=stored.filename,
                metadata={'origin': 'chat_attachment', 'transient': True},
            )
        if user_id:
            image_query_path = (
                stored.storage_path
                if stored.mime in IMAGE_MIME_TYPES
                else None
            )
            return [], None, stored.file_id, image_query_path
        docs = [
            {
                'text': chunk,
                'source': stored.filename,
                'file_id': stored.file_id,
                'modality': 'text',
                'score': None,
                'preview_ref': stored.storage_path,
            }
            for chunk in attachment_context_chunks
        ]
        image_query_path = (
            stored.storage_path if stored.mime in IMAGE_MIME_TYPES else None
        )
        return docs, stored.storage_path, stored.file_id, image_query_path

    if request_payload.attachment_id:
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='attachment_id is only available for authenticated users',
            )
        attachment = _resolve_attachment(
            attachment_id=request_payload.attachment_id, user_id=user_id
        )
        image_query_path = None
        if attachment.get('mime') in IMAGE_MIME_TYPES:
            image_query_path = (
                str(attachment.get('storage_path') or '') or None
            )
        return [], None, request_payload.attachment_id, image_query_path

    return [], None, None, None


@router.post('/auth/signup')
async def signup(
    request: Request,
    email: str | None = Form(default=None),
    password: str | None = Form(default=None),
) -> dict:
    """Sign up a user with email/password via Supabase."""
    auth_request = await _parse_auth_request(request, email, password)
    supabase = get_supabase_client(role='anon')
    try:
        resp = supabase.auth.sign_up(
            {'email': auth_request.email, 'password': auth_request.password}
        )
    except Exception as exc:
        raise _to_auth_http_exception(exc) from exc
    return {
        'user': _serialize(getattr(resp, 'user', None)),
        'session': _serialize(getattr(resp, 'session', None)),
    }


@router.post('/auth/signin')
async def signin(
    request: Request,
    email: str | None = Form(default=None),
    password: str | None = Form(default=None),
) -> dict:
    """Sign in a user with email/password via Supabase."""
    auth_request = await _parse_auth_request(request, email, password)
    supabase = get_supabase_client(role='anon')
    try:
        resp = supabase.auth.sign_in_with_password(
            {'email': auth_request.email, 'password': auth_request.password}
        )
    except Exception as exc:
        raise _to_auth_http_exception(exc) from exc
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


@router.get('/models')
def get_models(
    authorization: str | None = Header(default=None),
) -> dict:
    """Return available LLM models for model picker in UI."""
    is_authenticated = False
    if authorization:
        try:
            token = get_access_token(authorization)
            _ = get_current_user(token)
            is_authenticated = True
        except HTTPException:
            is_authenticated = False

    models = [
        {'id': model_id, 'label': model_id}
        for model_id in Config.llm_available_models
    ]
    return {
        'authenticated': is_authenticated,
        'models': models,
        'data': models,
        'items': models,
        'default_model': Config.llm_model_name,
    }


@router.post('/files/upload')
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = REQUIRED_UPLOAD_FILE,
    user: Annotated[dict, Depends(get_current_user)] = None,
) -> dict:
    """Upload a file from chat attachment and persist metadata."""
    upload_limit = _env_int(
        'UPLOAD_RATE_LIMIT_PER_MINUTE_AUTH',
        DEFAULT_UPLOAD_RATE_LIMIT_PER_MINUTE_AUTH,
    )
    _enforce_request_rate_limit(
        scope=f'upload_auth:{user["id"]}',
        limit=upload_limit,
        message='Upload rate limit exceeded',
    )

    stored = await save_upload_file(file)
    kb = _kb_service()
    usage = kb.get_user_storage_usage(user_id=user['id'])
    _enforce_quota_capacity(
        total_files_after=usage['total_files'] + 1,
        total_size_after=usage['total_size'] + stored.size,
    )
    existing = kb.find_existing_file_by_hash(
        user_id=user['id'],
        folder_id=None,
        content_hash=stored.content_hash,
    )
    if existing:
        delete_stored_file(stored.storage_path)
        return {
            'file_id': existing['id'],
            'filename': existing['filename'],
            'mime': existing['mime'],
            'size': existing['size'],
            'storage_path': existing['storage_path'],
            'ingest_job_id': None,
            'deduplicated': True,
        }

    kb.create_uploaded_file_record(
        user_id=user['id'],
        file_id=stored.file_id,
        filename=stored.filename,
        mime=stored.mime,
        size=stored.size,
        storage_path=stored.storage_path,
        content_hash=stored.content_hash,
        folder_id=None,
    )

    job_id = _enqueue_ingest_job(
        background_tasks=background_tasks,
        file_id=stored.file_id,
        file_path=stored.storage_path,
        filename=stored.filename,
        mime=stored.mime,
        user_id=user['id'],
        folder_id=None,
        folder_path='root',
        source_path=stored.filename,
        metadata={'origin': 'files_upload', 'transient': False},
    )

    return {
        'file_id': stored.file_id,
        'filename': stored.filename,
        'mime': stored.mime,
        'size': stored.size,
        'storage_path': stored.storage_path,
        'ingest_job_id': job_id,
        'deduplicated': False,
    }


@router.post('/kb/folders/upload')
async def upload_folder_files(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = REQUIRED_UPLOAD_FILES,
    relative_paths: list[str] | None = OPTIONAL_RELATIVE_PATHS,
    parent_id: str | None = Form(default=None),
    user: Annotated[dict, Depends(get_current_user)] = None,
) -> dict:
    """Upload multiple files and recreate provided folder structure in KB."""
    if not files:
        raise _api_error(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='At least one file is required',
            error_code='at_least_one_file_is_required',
        )
    max_folder_upload_files = _env_int(
        'MAX_FILES_PER_FOLDER_UPLOAD',
        DEFAULT_MAX_FILES_PER_FOLDER_UPLOAD,
    )
    if len(files) > max_folder_upload_files:
        raise _api_error(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail='Too many files in one folder upload request',
            error_code='too_many_files_in_one_folder_upload_request',
        )
    upload_limit = _env_int(
        'UPLOAD_RATE_LIMIT_PER_MINUTE_AUTH',
        DEFAULT_UPLOAD_RATE_LIMIT_PER_MINUTE_AUTH,
    )
    _enforce_request_rate_limit(
        scope=f'upload_auth:{user["id"]}',
        limit=upload_limit,
        message='Upload rate limit exceeded',
    )

    if relative_paths is not None and len(relative_paths) != len(files):
        raise _api_error(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='relative_paths length must match files length',
            error_code='relative_paths_length_must_match_files_length',
        )

    kb = _kb_service()
    usage = kb.get_user_storage_usage(user_id=user['id'])
    projected_files = usage['total_files']
    projected_size = usage['total_size']

    folder_rows = kb.list_folders(user_id=user['id'])
    folder_by_parent_and_name: dict[tuple[str | None, str], str] = {}
    folder_name_by_id: dict[str, str] = {}
    for folder in folder_rows:
        folder_id = folder['id']
        folder_by_parent_and_name[
            (folder.get('parent_id'), folder.get('name'))
        ] = folder_id
        folder_name_by_id[folder_id] = folder.get('name') or ''

    if parent_id is not None:
        parent_folder = kb.get_folder(folder_id=parent_id, user_id=user['id'])
        folder_name_by_id[parent_id] = parent_folder.get('name') or ''

    uploaded_items: list[dict[str, Any]] = []

    for idx, upload in enumerate(files):
        client_relative_path = (
            relative_paths[idx]
            if relative_paths is not None and idx < len(relative_paths)
            else ''
        )
        normalized_relative_path = _normalize_relative_path(
            client_relative_path,
            upload.filename or '',
        )
        parsed_path = PurePosixPath(normalized_relative_path)
        filename = parsed_path.name
        folder_parts = parsed_path.parts[:-1]

        current_parent_id = parent_id
        for part in folder_parts:
            key = (current_parent_id, part)
            folder_id = folder_by_parent_and_name.get(key)
            if folder_id is None:
                created = kb.create_folder(
                    user_id=user['id'],
                    name=part,
                    parent_id=current_parent_id,
                )
                folder_id = created['id']
                folder_by_parent_and_name[key] = folder_id
                folder_name_by_id[folder_id] = created.get('name') or part
            current_parent_id = folder_id

        target_folder_id = current_parent_id

        # Browser folder drag/drop may send empty or extensionless upload.filename.
        # Reuse basename from relative path to keep MIME validation deterministic.
        upload_filename = (upload.filename or '').strip()
        if not Path(upload_filename).suffix and Path(filename).suffix:
            upload.filename = filename

        stored = await save_upload_file(upload)
        existing = kb.find_existing_file_by_hash(
            user_id=user['id'],
            folder_id=target_folder_id,
            content_hash=stored.content_hash,
        )
        if existing:
            delete_stored_file(stored.storage_path)
            uploaded_items.append(
                {
                    'file_id': existing['id'],
                    'filename': filename,
                    'mime': existing['mime'],
                    'size': existing['size'],
                    'relative_path': normalized_relative_path,
                    'folder_id': existing.get('folder_id'),
                    'ingest_job_id': None,
                    'deduplicated': True,
                }
            )
            continue
        _enforce_quota_capacity(
            total_files_after=projected_files + 1,
            total_size_after=projected_size + stored.size,
        )
        projected_files += 1
        projected_size += stored.size

        created_file = kb.create_uploaded_file_record(
            user_id=user['id'],
            file_id=stored.file_id,
            filename=stored.filename,
            mime=stored.mime,
            size=stored.size,
            storage_path=stored.storage_path,
            content_hash=stored.content_hash,
            folder_id=target_folder_id,
        )

        if target_folder_id is not None:
            created_file = kb.attach_file_to_folder(
                user_id=user['id'],
                file_id=stored.file_id,
                folder_id=target_folder_id,
            )

        job_id = _enqueue_ingest_job(
            background_tasks=background_tasks,
            file_id=stored.file_id,
            file_path=stored.storage_path,
            filename=stored.filename,
            mime=stored.mime,
            user_id=user['id'],
            folder_id=target_folder_id,
            folder_path='/'.join(folder_parts) if folder_parts else 'root',
            source_path=normalized_relative_path,
            metadata={'origin': 'folders_upload', 'transient': False},
        )

        uploaded_items.append(
            {
                'file_id': stored.file_id,
                'filename': filename,
                'mime': stored.mime,
                'size': stored.size,
                'relative_path': normalized_relative_path,
                'folder_id': created_file.get('folder_id'),
                'ingest_job_id': job_id,
                'deduplicated': False,
            }
        )

    return {'uploaded': uploaded_items}


@router.delete('/files/{file_id}')
def delete_file(
    file_id: str,
    user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """Delete uploaded file, metadata and indexed vectors."""
    kb = _kb_service()
    kb.delete_file(file_id=file_id, user_id=user['id'])
    return {'ok': True}


@router.get('/files/{file_id}')
def get_file_metadata(
    file_id: str,
    user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """Return uploaded file metadata for the current user."""
    kb = _kb_service()
    file_row = kb.get_file(file_id=file_id, user_id=user['id'])
    return {'file': file_row}


@router.get('/files/{file_id}/processing')
def get_file_processing(
    file_id: str,
    user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """Return latest ingest processing status for a file."""
    kb = _kb_service()
    _ = kb.get_file(file_id=file_id, user_id=user['id'])
    jobs = _ingest_jobs_service()
    file_jobs = jobs.list_jobs_for_file(
        user_id=user['id'],
        file_id=file_id,
        limit=20,
    )
    latest = file_jobs[0] if file_jobs else None
    status_value = latest.get('status') if latest else 'not_indexed'
    return {'file_id': file_id, 'status': status_value, 'jobs': file_jobs}


@router.post('/files/{file_id}/reindex')
def reindex_file(
    file_id: str,
    background_tasks: BackgroundTasks,
    user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """Schedule reindex for one user file and enqueue worker processing."""
    kb = _kb_service()
    _ = kb.get_file(file_id=file_id, user_id=user['id'])
    consistency = _consistency_service()
    result = consistency.schedule_reindex(
        user_id=user['id'],
        file_ids=[file_id],
        limit=1,
        only_missing_vectors=False,
    )
    worker = _ingest_worker()
    for job in result.get('jobs', []):
        job_id = str(job.get('job_id') or '')
        owner_user_id = str(job.get('user_id') or '')
        if not job_id or not owner_user_id:
            continue
        background_tasks.add_task(
            worker.process_job,
            job_id=job_id,
            user_id=owner_user_id,
        )
    return result


@router.get('/files/{file_id}/download')
def download_file(
    file_id: str,
    user: Annotated[dict, Depends(get_current_user)],
) -> FileResponse:
    """Download uploaded file with original filename and MIME type."""
    kb = _kb_service()
    file_row = kb.get_file(file_id=file_id, user_id=user['id'])
    file_path = Path(file_row['storage_path'])
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Stored file not found',
        )
    return FileResponse(
        path=str(file_path),
        media_type=file_row.get('mime') or 'application/octet-stream',
        filename=file_row.get('filename') or file_path.name,
    )


@router.get('/kb/tree')
def kb_tree(user: Annotated[dict, Depends(get_current_user)]) -> dict:
    """Get KB tree (folders and attached files) for the current user."""
    kb = _kb_service()
    return kb.build_tree(user_id=user['id'])


@router.post('/kb/folders')
def create_kb_folder(
    request: KBFolderCreateRequest,
    user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """Create a knowledge-base folder."""
    kb = _kb_service()
    folder = kb.create_folder(
        user_id=user['id'],
        name=request.name,
        parent_id=request.parent_id,
    )
    return {'folder': folder}


@router.delete('/kb/folders/{folder_id}')
def delete_kb_folder(
    folder_id: str,
    user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """Recursively delete folder subtree and linked files."""
    kb = _kb_service()
    kb.delete_folder_recursive(folder_id=folder_id, user_id=user['id'])
    return {'ok': True}


@router.post('/kb/files')
def attach_kb_file(
    background_tasks: BackgroundTasks,
    request: KBFileAttachRequest,
    user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """Attach an uploaded file to a folder and update index metadata."""
    kb = _kb_service()
    if request.folder_id:
        _ = kb.get_folder(folder_id=request.folder_id, user_id=user['id'])

    attached = kb.attach_file_to_folder(
        user_id=user['id'],
        file_id=request.file_id,
        folder_id=request.folder_id,
    )
    kb.delete_vectors_for_file(request.file_id)
    folder_path = kb.get_folder_path(
        user_id=user['id'],
        folder_id=request.folder_id,
    )

    job_id = _enqueue_ingest_job(
        background_tasks=background_tasks,
        file_id=attached['id'],
        file_path=attached['storage_path'],
        filename=attached['filename'],
        mime=attached['mime'],
        user_id=user['id'],
        folder_id=request.folder_id,
        folder_path=folder_path,
        source_path=attached['filename'],
        metadata={'origin': 'attach_kb_file', 'transient': False},
    )

    return {'file': attached, 'ingest_job_id': job_id}


@router.get('/kb/files')
def list_kb_files(
    folder_id: str | None = None,
    user: Annotated[dict, Depends(get_current_user)] = None,
) -> dict:
    """List files in folder or root when folder is not provided."""
    kb = _kb_service()
    files = kb.list_files(user_id=user['id'], folder_id=folder_id)
    return {'data': files}


@router.delete('/kb/files/{file_id}')
def delete_kb_file(
    file_id: str,
    user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """Delete KB file and its vector entries."""
    kb = _kb_service()
    kb.delete_file(file_id=file_id, user_id=user['id'])
    return {'ok': True}


@router.post('/ask')
async def ask_mixed(
    request: Request,
    query: str | None = Form(default=None),
    top_k: int = Form(default=5),
    image: str | None = Form(default=None),
    model: str | None = Form(default=None),
    attachment_id: str | None = Form(default=None),
    guest_session_id: str | None = Form(default=None),
    folder_ids: str | None = Form(default=None),
    file_ids: str | None = Form(default=None),
    attachment: UploadFile | None = OPTIONAL_UPLOAD_FILE,
) -> dict:
    """Handle chat request with optional model selection and attachment."""
    payload = await _parse_query_request(
        request,
        query,
        top_k,
        image,
        model,
        attachment_id,
        guest_session_id,
        folder_ids,
        file_ids,
    )
    effective_guest_session_id = (
        payload.guest_session_id or f'guest-{uuid.uuid4()}'
    )
    guest_subject = effective_guest_session_id
    if not guest_subject:
        guest_subject = _client_identifier(request)
    ask_limit = _env_int(
        'ASK_RATE_LIMIT_PER_MINUTE_GUEST',
        DEFAULT_ASK_RATE_LIMIT_PER_MINUTE_GUEST,
    )
    _enforce_request_rate_limit(
        scope=f'ask_guest:{guest_subject}',
        limit=ask_limit,
        message='Ask rate limit exceeded',
    )
    if attachment is not None:
        upload_limit_guest = _env_int(
            'UPLOAD_RATE_LIMIT_PER_MINUTE_GUEST',
            DEFAULT_UPLOAD_RATE_LIMIT_PER_MINUTE_GUEST,
        )
        _enforce_request_rate_limit(
            scope=f'upload_guest:{guest_subject}',
            limit=upload_limit_guest,
            message='Upload rate limit exceeded',
        )

    (
        extra_docs,
        transient_storage_path,
        attachment_file_id,
        image_query_path,
    ) = await _prepare_attachment_data(
        request_payload=payload,
        attachment_file=attachment,
        user_id=None,
        guest_session_id=effective_guest_session_id,
    )

    effective_file_ids = list(dict.fromkeys(payload.file_ids))
    if attachment_file_id and not image_query_path:
        effective_file_ids.append(attachment_file_id)

    try:
        result = rag.generate_answer(
            payload.query,
            top_k=payload.top_k,
            image=payload.image,
            image_query_path=image_query_path,
            model=payload.model,
            file_ids=effective_file_ids or None,
            exclude_file_ids=[attachment_file_id]
            if attachment_file_id
            else None,
            extra_docs=extra_docs,
        )
        result['guest_session_id'] = effective_guest_session_id
        return result
    finally:
        if attachment_file_id and transient_storage_path:
            ingest = _ingest_service()
            ingest.delete_vectors_for_file(file_id=attachment_file_id)
        if transient_storage_path:
            delete_stored_file(transient_storage_path)


@router.post('/ask_auth')
async def ask_mixed_auth(
    request: Request,
    user: Annotated[dict, Depends(get_current_user)],
    query: str | None = Form(default=None),
    top_k: int = Form(default=5),
    image: str | None = Form(default=None),
    model: str | None = Form(default=None),
    attachment_id: str | None = Form(default=None),
    guest_session_id: str | None = Form(default=None),
    folder_ids: str | None = Form(default=None),
    file_ids: str | None = Form(default=None),
    attachment: UploadFile | None = OPTIONAL_UPLOAD_FILE,
) -> dict:
    """Run RAG with user filtering and optional attachment/model."""
    payload = await _parse_query_request(
        request,
        query,
        top_k,
        image,
        model,
        attachment_id,
        guest_session_id,
        folder_ids,
        file_ids,
    )
    ask_limit = _env_int(
        'ASK_RATE_LIMIT_PER_MINUTE_AUTH',
        DEFAULT_ASK_RATE_LIMIT_PER_MINUTE_AUTH,
    )
    _enforce_request_rate_limit(
        scope=f'ask_auth:{user["id"]}',
        limit=ask_limit,
        message='Ask rate limit exceeded',
    )
    if attachment is not None:
        upload_limit = _env_int(
            'UPLOAD_RATE_LIMIT_PER_MINUTE_AUTH',
            DEFAULT_UPLOAD_RATE_LIMIT_PER_MINUTE_AUTH,
        )
        _enforce_request_rate_limit(
            scope=f'upload_auth:{user["id"]}',
            limit=upload_limit,
            message='Upload rate limit exceeded',
        )

    kb = _kb_service()
    (
        extra_docs,
        transient_storage_path,
        attachment_file_id,
        image_query_path,
    ) = await _prepare_attachment_data(
        request_payload=payload,
        attachment_file=attachment,
        user_id=user['id'],
    )

    effective_file_ids = list(dict.fromkeys(payload.file_ids))
    if attachment_file_id and not image_query_path:
        effective_file_ids.append(attachment_file_id)

    folder_scopes: list[str] | None = None
    if not effective_file_ids and payload.folder_ids:
        expanded = kb.get_descendant_folder_ids(
            user_id=user['id'], folder_ids=payload.folder_ids
        )
        folder_scopes = ['root', *expanded]

    try:
        result = rag.generate_answer(
            payload.query,
            top_k=payload.top_k,
            image=payload.image,
            image_query_path=image_query_path,
            user_id=user['id'],
            model=payload.model,
            folder_scopes=folder_scopes,
            file_ids=effective_file_ids or None,
            exclude_file_ids=[attachment_file_id]
            if attachment_file_id
            else None,
            extra_docs=extra_docs,
        )

        supabase = get_supabase_client(role='service')
        supabase.table('query_history').insert(
            {
                'user_id': user['id'],
                'query': payload.query,
                'answer': result.get('answer', ''),
                'retrieved_docs': result.get('retrieved_docs', []),
            }
        ).execute()

        return result
    finally:
        if transient_storage_path:
            delete_stored_file(transient_storage_path)


@router.get('/ingest/jobs')
def list_ingest_jobs(
    status: str | None = None,
    limit: int = 50,
    user: Annotated[dict, Depends(get_current_user)] = None,
) -> dict:
    """List ingest jobs for the current authenticated user."""
    resolved_limit = max(1, min(limit, 200))
    jobs = _ingest_jobs_service()
    jobs.refresh_depth_metrics()
    data = jobs.list_jobs(
        user_id=user['id'],
        status=status,
        limit=resolved_limit,
    )
    return {'data': data}


@router.get('/ingest/jobs/{job_id}')
def get_ingest_job(
    job_id: str,
    user: Annotated[dict, Depends(get_current_user)] = None,
) -> dict:
    """Get one ingest job for the current authenticated user."""
    jobs = _ingest_jobs_service()
    job = jobs.get_job(job_id=job_id, user_id=user['id'])
    if not job:
        raise _api_error(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Ingest job not found',
            error_code='ingest_job_not_found',
        )
    return {'job': job}


@router.get('/ingest/dlq')
def list_ingest_dlq(
    limit: int = 50,
    user: Annotated[dict, Depends(get_current_user)] = None,
) -> dict:
    """List DLQ items for the current authenticated user."""
    resolved_limit = max(1, min(limit, 200))
    jobs = _ingest_jobs_service()
    jobs.refresh_depth_metrics()
    data = jobs.list_dlq(user_id=user['id'], limit=resolved_limit)
    return {'data': data}


@router.get('/ingest/dlq/{dlq_id}')
def get_ingest_dlq_item(
    dlq_id: int,
    user: Annotated[dict, Depends(get_current_user)] = None,
) -> dict:
    """Get one DLQ item for the current authenticated user."""
    jobs = _ingest_jobs_service()
    item = jobs.get_dlq_item(dlq_id=dlq_id, user_id=user['id'])
    if not item:
        raise _api_error(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Ingest DLQ item not found',
            error_code='ingest_dlq_item_not_found',
        )
    return {'item': item}


@router.post('/ingest/dlq/{dlq_id}/requeue')
def requeue_ingest_dlq_item(
    dlq_id: int,
    background_tasks: BackgroundTasks,
    user: Annotated[dict, Depends(get_current_user)] = None,
) -> dict:
    """Requeue DLQ item back to ingest queue and schedule processing."""
    jobs = _ingest_jobs_service()
    job = jobs.requeue_from_dlq(dlq_id=dlq_id, user_id=user['id'])
    if not job:
        raise _api_error(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Ingest DLQ item not found or cannot be requeued',
            error_code='ingest_dlq_item_not_requeueable',
        )

    worker = _ingest_worker()
    background_tasks.add_task(
        worker.process_job,
        job_id=str(job['id']),
        user_id=user['id'],
    )
    return {'job': job}


@router.post('/ingest/jobs/{job_id}/retry')
def retry_ingest_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    user: Annotated[dict, Depends(get_current_user)] = None,
) -> dict:
    """Retry failed ingest job for a persisted user file."""
    jobs = _ingest_jobs_service()
    job = jobs.get_job(job_id=job_id, user_id=user['id'])
    if not job:
        raise _api_error(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Ingest job not found',
            error_code='ingest_job_not_found',
        )
    if job.get('owner_type') != 'user':
        raise _api_error(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Only user-owned jobs can be retried',
            error_code='only_user_owned_jobs_can_be_retried',
        )
    if job.get('status') != 'failed':
        raise _api_error(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Only failed jobs can be retried',
            error_code='only_failed_jobs_can_be_retried',
        )

    jobs.mark_queued(job_id=job_id)
    worker = _ingest_worker()
    background_tasks.add_task(
        worker.process_job,
        job_id=job_id,
        user_id=user['id'],
    )

    updated = jobs.get_job(job_id=job_id, user_id=user['id'])
    return {'job': updated}


@router.post('/admin/ingest/replay')
def admin_replay_ingest_jobs(
    request: Request,
    background_tasks: BackgroundTasks,
    status_filter: str = 'failed',
    limit: int = 50,
    user_id: str | None = None,
    _: Annotated[bool, Depends(get_admin_access)] = False,
) -> dict:
    """Admin: replay queued/failed jobs in batches."""
    allowed = {'queued', 'failed'}
    if status_filter not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'status_filter must be one of: {sorted(allowed)}',
        )

    jobs = _ingest_jobs_service()
    worker = _ingest_worker()
    selected = jobs.list_jobs_admin(
        status=status_filter,
        limit=max(1, min(limit, 500)),
        user_id=user_id,
    )

    replayed_ids: list[str] = []
    for job in selected:
        job_id = str(job.get('id') or '')
        owner_user_id = job.get('user_id')
        if not job_id or not owner_user_id:
            continue
        jobs.mark_queued(job_id=job_id)
        background_tasks.add_task(
            worker.process_job,
            job_id=job_id,
            user_id=str(owner_user_id),
        )
        replayed_ids.append(job_id)

    jobs.refresh_depth_metrics()
    _audit_admin_action(
        request=request,
        action='admin_ingest_replay',
        details={
            'status_filter': status_filter,
            'limit': limit,
            'user_id': user_id,
            'selected': len(selected),
            'replayed': len(replayed_ids),
        },
    )
    return {
        'selected': len(selected),
        'replayed': len(replayed_ids),
        'ids': replayed_ids,
    }


@router.post('/admin/ingest/purge')
def admin_purge_ingest_data(
    request: Request,
    older_than_hours: int = 24 * 7,
    statuses: str = 'completed,failed',
    purge_dlq: bool = True,
    limit: int = 500,
    _: Annotated[bool, Depends(get_admin_access)] = False,
) -> dict:
    """Admin: purge old ingest jobs and DLQ entries in batches."""
    parsed_statuses = [s.strip() for s in statuses.split(',') if s.strip()]
    allowed_statuses = {'completed', 'failed'}
    invalid = [s for s in parsed_statuses if s not in allowed_statuses]
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Unsupported statuses: {invalid}',
        )
    if not parsed_statuses:
        parsed_statuses = ['completed', 'failed']

    jobs = _ingest_jobs_service()
    purged_jobs = jobs.purge_jobs(
        statuses=parsed_statuses,
        older_than_hours=max(1, older_than_hours),
        limit=max(1, min(limit, 2000)),
    )
    purged_dlq = 0
    if purge_dlq:
        purged_dlq = jobs.purge_dlq(
            older_than_hours=max(1, older_than_hours),
            limit=max(1, min(limit, 2000)),
        )
    jobs.refresh_depth_metrics()
    _audit_admin_action(
        request=request,
        action='admin_ingest_purge',
        details={
            'older_than_hours': older_than_hours,
            'statuses': parsed_statuses,
            'purge_dlq': purge_dlq,
            'limit': limit,
            'purged_jobs': purged_jobs,
            'purged_dlq': purged_dlq,
        },
    )
    return {'purged_jobs': purged_jobs, 'purged_dlq': purged_dlq}


@router.post('/admin/consistency/reindex')
def admin_reindex_files(
    request: Request,
    background_tasks: BackgroundTasks,
    user_id: str | None = None,
    limit: int = 100,
    only_missing_vectors: bool = False,
    _: Annotated[bool, Depends(get_admin_access)] = False,
) -> dict:
    """Admin: schedule reindex jobs for selected user files."""
    consistency = _consistency_service()
    result = consistency.schedule_reindex(
        user_id=user_id,
        file_ids=None,
        limit=max(1, min(limit, 2_000)),
        only_missing_vectors=only_missing_vectors,
    )
    worker = _ingest_worker()
    for job in result.get('jobs', []):
        job_id = str(job.get('job_id') or '')
        owner_user_id = str(job.get('user_id') or '')
        if not job_id or not owner_user_id:
            continue
        background_tasks.add_task(
            worker.process_job,
            job_id=job_id,
            user_id=owner_user_id,
        )
    _audit_admin_action(
        request=request,
        action='admin_consistency_reindex',
        details={
            'user_id': user_id,
            'limit': limit,
            'only_missing_vectors': only_missing_vectors,
            'selected': result.get('selected'),
            'scheduled': result.get('scheduled'),
        },
    )
    return result


@router.post('/admin/consistency/cleanup')
def admin_cleanup_consistency(
    request: Request,
    dry_run: bool = True,
    cleanup_missing_storage_records: bool = True,
    cleanup_orphan_uploads: bool = True,
    cleanup_orphan_vectors: bool = True,
    uploads_min_age_seconds: int = 1_800,
    limit: int = 500,
    _: Annotated[bool, Depends(get_admin_access)] = False,
) -> dict:
    """Admin: cleanup orphan records/files/vectors and return report."""
    consistency = _consistency_service()
    report: dict[str, Any] = {'dry_run': dry_run}

    if cleanup_missing_storage_records:
        report['missing_storage_records'] = (
            consistency.cleanup_missing_storage_records(
                dry_run=dry_run,
                user_id=None,
                limit=max(1, min(limit, 2_000)),
            )
        )
    if cleanup_orphan_uploads:
        report['orphan_uploads'] = consistency.cleanup_orphan_uploads(
            dry_run=dry_run,
            min_age_seconds=max(0, uploads_min_age_seconds),
            limit=max(1, min(limit, 2_000)),
        )
    if cleanup_orphan_vectors:
        report['orphan_vectors'] = consistency.cleanup_orphan_vectors(
            dry_run=dry_run,
            limit_orphan_file_ids=max(1, min(limit, 5_000)),
        )

    _audit_admin_action(
        request=request,
        action='admin_consistency_cleanup',
        details={
            'dry_run': dry_run,
            'cleanup_missing_storage_records': cleanup_missing_storage_records,
            'cleanup_orphan_uploads': cleanup_orphan_uploads,
            'cleanup_orphan_vectors': cleanup_orphan_vectors,
            'uploads_min_age_seconds': uploads_min_age_seconds,
            'limit': limit,
        },
    )
    return report


@router.post('/embed/text')
def embed_text(request: TextEmbeddingRequest) -> dict:
    """Generate embedding for a text payload."""
    vector = text_embedding(request.text, provider_name=request.provider)
    return {
        'provider': request.provider,
        'modality': 'text',
        'dimension': len(vector),
        'embedding': vector,
    }


@router.post('/embed/image')
def embed_image(request: ImageEmbeddingRequest) -> dict:
    """Generate embedding for a local image file."""
    vector = image_embedding_from_path(
        request.image_path,
        provider_name=request.provider,
    )
    return {
        'provider': request.provider,
        'modality': 'image',
        'dimension': len(vector),
        'embedding': vector,
    }


@router.post('/embed/video')
def embed_video(request: VideoEmbeddingRequest) -> dict:
    """Generate embedding for a local video file."""
    vector = video_embedding_from_path(
        request.video_path,
        sample_fps=request.sample_fps,
        provider_name=request.provider,
    )
    return {
        'provider': request.provider,
        'modality': 'video',
        'dimension': len(vector),
        'embedding': vector,
        'sample_fps': request.sample_fps,
    }


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
