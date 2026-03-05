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
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from backend.core.embeddings import (
    image_embedding_from_path,
    text_embedding,
    video_embedding_from_path,
)
from backend.core.multimodal_rag import LocalRAG
from backend.services.admin_rate_limiter import AdminRateLimiter
from backend.services.ingest import IngestService
from backend.services.ingest_jobs import IngestJobsService
from backend.services.ingest_worker import IngestWorker
from backend.services.kb import KBService
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
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Missing or invalid Authorization header',
        )
    return authorization.split(' ', 1)[1].strip()


def get_admin_access(x_admin_key: str = Header(default='')) -> bool:
    """Protect admin-only endpoints with static admin key."""
    admin_key = (os.getenv('ADMIN_API_KEY') or '').strip()
    if not admin_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Admin API is not configured',
        )
    if x_admin_key != admin_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Invalid admin key',
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
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail='Admin rate limit exceeded',
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
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid or expired token',
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
    metadata: dict[str, Any] | None = None,
) -> str | None:
    owner_id = user_id or guest_session_id
    if not owner_id:
        raise ValueError('Either user_id or guest_session_id must be provided')
    owner_type = 'user' if user_id else 'guest'

    jobs = _ingest_jobs_service()
    job_id = jobs.create_job(
        owner_type=owner_type,
        owner_id=owner_id,
        user_id=user_id,
        file_id=file_id,
        filename=filename,
        mime=mime,
        source_path=source_path,
        folder_id=folder_id,
        metadata=metadata,
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
            metadata=metadata,
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
            detail=f'Invalid relative path: {relative_path!r}',
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
                detail='JSON body must be an object',
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

    return AuthRequest(email=email, password=password)


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
                detail='JSON body must be an object',
            )
        return QueryRequest(**payload)

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
    return QueryRequest(**payload)


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
            detail='attachment_id not found',
        )
    return data[0]


async def _prepare_attachment_data(
    *,
    request_payload: QueryRequest,
    attachment_file: UploadFile | None,
    user_id: str | None,
    guest_session_id: str | None = None,
) -> tuple[list[dict[str, str]], str | None, str | None]:
    """Prepare extra context docs from attachment id or direct multipart file."""
    ingest = _ingest_service()

    if attachment_file is not None:
        stored = await save_upload_file(attachment_file)
        if user_id:
            kb = _kb_service()
            kb.create_uploaded_file_record(
                user_id=user_id,
                file_id=stored.file_id,
                filename=stored.filename,
                mime=stored.mime,
                size=stored.size,
                storage_path=stored.storage_path,
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
                source_path=stored.filename,
                metadata={'origin': 'chat_attachment', 'transient': True},
            )
        if user_id:
            return [], None, stored.file_id
        docs = [
            {
                'text': chunk,
                'source': stored.filename,
            }
            for chunk in attachment_context_chunks
        ]
        return docs, stored.storage_path, stored.file_id

    if request_payload.attachment_id:
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='attachment_id is only available for authenticated users',
            )
        _ = _resolve_attachment(
            attachment_id=request_payload.attachment_id, user_id=user_id
        )
        return [], None, request_payload.attachment_id

    return [], None, None


@router.post('/auth/signup')
async def signup(
    request: Request,
    email: str | None = Form(default=None),
    password: str | None = Form(default=None),
) -> dict:
    """Sign up a user with email/password via Supabase."""
    auth_request = await _parse_auth_request(request, email, password)
    supabase = get_supabase_client(role='anon')
    resp = supabase.auth.sign_up(
        {'email': auth_request.email, 'password': auth_request.password}
    )
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
    resp = supabase.auth.sign_in_with_password(
        {'email': auth_request.email, 'password': auth_request.password}
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
    stored = await save_upload_file(file)
    kb = _kb_service()
    kb.create_uploaded_file_record(
        user_id=user['id'],
        file_id=stored.file_id,
        filename=stored.filename,
        mime=stored.mime,
        size=stored.size,
        storage_path=stored.storage_path,
    )

    job_id = _enqueue_ingest_job(
        background_tasks=background_tasks,
        file_id=stored.file_id,
        file_path=stored.storage_path,
        filename=stored.filename,
        mime=stored.mime,
        user_id=user['id'],
        folder_id=None,
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='At least one file is required',
        )

    if relative_paths is not None and len(relative_paths) != len(files):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='relative_paths length must match files length',
        )

    kb = _kb_service()

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
        created_file = kb.create_uploaded_file_record(
            user_id=user['id'],
            file_id=stored.file_id,
            filename=stored.filename,
            mime=stored.mime,
            size=stored.size,
            storage_path=stored.storage_path,
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

    job_id = _enqueue_ingest_job(
        background_tasks=background_tasks,
        file_id=attached['id'],
        file_path=attached['storage_path'],
        filename=attached['filename'],
        mime=attached['mime'],
        user_id=user['id'],
        folder_id=request.folder_id,
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

    (
        extra_docs,
        transient_storage_path,
        attachment_file_id,
    ) = await _prepare_attachment_data(
        request_payload=payload,
        attachment_file=attachment,
        user_id=None,
        guest_session_id=effective_guest_session_id,
    )

    effective_file_ids = list(dict.fromkeys(payload.file_ids))
    if attachment_file_id:
        effective_file_ids.append(attachment_file_id)

    try:
        result = rag.generate_answer(
            payload.query,
            top_k=payload.top_k,
            image=payload.image,
            model=payload.model,
            file_ids=effective_file_ids or None,
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

    kb = _kb_service()
    (
        extra_docs,
        transient_storage_path,
        attachment_file_id,
    ) = await _prepare_attachment_data(
        request_payload=payload,
        attachment_file=attachment,
        user_id=user['id'],
    )

    effective_file_ids = list(dict.fromkeys(payload.file_ids))
    if attachment_file_id:
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
            user_id=user['id'],
            model=payload.model,
            folder_scopes=folder_scopes,
            file_ids=effective_file_ids or None,
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Ingest job not found',
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Ingest DLQ item not found',
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Ingest DLQ item not found or cannot be requeued',
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Ingest job not found',
        )
    if job.get('owner_type') != 'user':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Only user-owned jobs can be retried',
        )
    if job.get('status') != 'failed':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Only failed jobs can be retried',
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
    return {
        'selected': len(selected),
        'replayed': len(replayed_ids),
        'ids': replayed_ids,
    }


@router.post('/admin/ingest/purge')
def admin_purge_ingest_data(
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
    return {'purged_jobs': purged_jobs, 'purged_dlq': purged_dlq}


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
