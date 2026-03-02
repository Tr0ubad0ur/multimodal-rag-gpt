from pathlib import Path
from typing import Annotated, Any, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field, field_validator

from backend.core.embeddings import (
    image_embedding_from_path,
    text_embedding,
    video_embedding_from_path,
)
from backend.core.multimodal_rag import LocalRAG
from backend.services.ingest import IngestService
from backend.services.kb import KBService
from backend.services.storage import delete_stored_file, save_upload_file
from backend.utils.config_handler import Config
from backend.utils.supabase_client import get_supabase_client

router = APIRouter()
rag = LocalRAG()
REQUIRED_UPLOAD_FILE = File(...)
OPTIONAL_UPLOAD_FILE = File(default=None)


class QueryRequest(BaseModel):
    """Schema for user query requests."""

    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)
    image: Optional[str] = None
    model: Optional[str] = Field(default=None, max_length=256)
    attachment_id: Optional[str] = Field(default=None, max_length=64)
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
            ingest.ingest_file(
                file_id=stored.file_id,
                file_path=stored.storage_path,
                filename=stored.filename,
                mime=stored.mime,
                user_id=user_id,
                folder_id=None,
                folder_name=None,
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

    ingest = _ingest_service()
    ingest.ingest_file(
        file_id=stored.file_id,
        file_path=stored.storage_path,
        filename=stored.filename,
        mime=stored.mime,
        user_id=user['id'],
        folder_id=None,
        folder_name=None,
    )

    return {
        'file_id': stored.file_id,
        'filename': stored.filename,
        'mime': stored.mime,
        'size': stored.size,
        'storage_path': stored.storage_path,
    }


@router.delete('/files/{file_id}')
def delete_file(
    file_id: str,
    user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """Delete uploaded file, metadata and indexed vectors."""
    kb = _kb_service()
    kb.delete_file(file_id=file_id, user_id=user['id'])
    return {'ok': True}


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
    request: KBFileAttachRequest,
    user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """Attach an uploaded file to a folder and update index metadata."""
    kb = _kb_service()
    folder_name: str | None = None
    if request.folder_id:
        folder = kb.get_folder(folder_id=request.folder_id, user_id=user['id'])
        folder_name = folder.get('name')

    attached = kb.attach_file_to_folder(
        user_id=user['id'],
        file_id=request.file_id,
        folder_id=request.folder_id,
    )
    kb.delete_vectors_for_file(request.file_id)

    ingest = _ingest_service()
    ingest.ingest_file(
        file_id=attached['id'],
        file_path=attached['storage_path'],
        filename=attached['filename'],
        mime=attached['mime'],
        user_id=user['id'],
        folder_id=request.folder_id,
        folder_name=folder_name,
    )

    return {'file': attached}


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
        folder_ids,
        file_ids,
    )

    (
        extra_docs,
        transient_storage_path,
        attachment_file_id,
    ) = await _prepare_attachment_data(
        request_payload=payload,
        attachment_file=attachment,
        user_id=None,
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
        return result
    finally:
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


# TODO write this requests
# @router.post('/test_llm')
# @router.post('/test_qdrant')
