from __future__ import annotations

from collections import defaultdict
from typing import Any

from fastapi import HTTPException, status
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from backend.services.storage import delete_stored_file
from backend.utils.config_handler import Config
from backend.utils.supabase_client import get_supabase_client


class KBService:
    """Service layer for KB folders/files and related vector cleanup."""

    def __init__(self) -> None:
        """Initialize DB and vector DB clients."""
        self.supabase = get_supabase_client(role='service')
        self.qdrant = QdrantClient(url=Config.qdrant_url)

    def create_uploaded_file_record(
        self,
        *,
        user_id: str,
        file_id: str,
        filename: str,
        mime: str,
        size: int,
        storage_path: str,
    ) -> dict[str, Any]:
        """Persist uploaded file metadata in kb_files."""
        resp = (
            self.supabase.table('kb_files')
            .insert(
                {
                    'id': file_id,
                    'user_id': user_id,
                    'folder_id': None,
                    'filename': filename,
                    'mime': mime,
                    'size': size,
                    'storage_path': storage_path,
                }
            )
            .execute()
        )
        return (getattr(resp, 'data', None) or [{}])[0]

    def get_file(self, *, file_id: str, user_id: str) -> dict[str, Any]:
        """Return one file row for a user or raise 404."""
        resp = (
            self.supabase.table('kb_files')
            .select('*')
            .eq('id', file_id)
            .eq('user_id', user_id)
            .limit(1)
            .execute()
        )
        data = getattr(resp, 'data', None) or []
        if not data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='File not found',
            )
        return data[0]

    def delete_file(self, *, file_id: str, user_id: str) -> None:
        """Delete file metadata, local file and related vectors."""
        file_row = self.get_file(file_id=file_id, user_id=user_id)
        delete_stored_file(file_row['storage_path'])
        self._delete_vectors_by_file_id(file_id)
        (
            self.supabase.table('kb_files')
            .delete()
            .eq('id', file_id)
            .eq('user_id', user_id)
            .execute()
        )

    def create_folder(
        self, *, user_id: str, name: str, parent_id: str | None = None
    ) -> dict[str, Any]:
        """Create a KB folder for the user."""
        payload = {
            'user_id': user_id,
            'name': name,
            'parent_id': parent_id,
        }
        resp = self.supabase.table('kb_folders').insert(payload).execute()
        return (getattr(resp, 'data', None) or [{}])[0]

    def list_folders(self, *, user_id: str) -> list[dict[str, Any]]:
        """List all folders for the user."""
        resp = (
            self.supabase.table('kb_folders')
            .select('*')
            .eq('user_id', user_id)
            .order('created_at')
            .execute()
        )
        return getattr(resp, 'data', None) or []

    def list_files(
        self, *, user_id: str, folder_id: str | None = None
    ) -> list[dict[str, Any]]:
        """List files in a folder or in root when folder_id is None."""
        query = (
            self.supabase.table('kb_files')
            .select('*')
            .eq('user_id', user_id)
            .order('created_at', desc=True)
        )
        if folder_id is None:
            query = query.is_('folder_id', None)
        else:
            query = query.eq('folder_id', folder_id)

        resp = query.execute()
        return getattr(resp, 'data', None) or []

    def attach_file_to_folder(
        self, *, user_id: str, file_id: str, folder_id: str | None
    ) -> dict[str, Any]:
        """Attach an uploaded file to a specific folder."""
        if folder_id is not None:
            _ = self.get_folder(folder_id=folder_id, user_id=user_id)
        self.get_file(file_id=file_id, user_id=user_id)
        resp = (
            self.supabase.table('kb_files')
            .update({'folder_id': folder_id})
            .eq('id', file_id)
            .eq('user_id', user_id)
            .execute()
        )
        updated = (getattr(resp, 'data', None) or [{}])[0]
        return updated

    def get_folder(self, *, folder_id: str, user_id: str) -> dict[str, Any]:
        """Return one folder row for a user or raise 404."""
        resp = (
            self.supabase.table('kb_folders')
            .select('*')
            .eq('id', folder_id)
            .eq('user_id', user_id)
            .limit(1)
            .execute()
        )
        data = getattr(resp, 'data', None) or []
        if not data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Folder not found',
            )
        return data[0]

    def delete_folder_recursive(self, *, folder_id: str, user_id: str) -> None:
        """Delete folder subtree and all files in it."""
        folders = self.list_folders(user_id=user_id)
        children_by_parent: dict[str | None, list[str]] = defaultdict(list)
        for folder in folders:
            children_by_parent[folder.get('parent_id')].append(folder['id'])

        to_delete: set[str] = set()
        stack = [folder_id]
        while stack:
            current = stack.pop()
            if current in to_delete:
                continue
            to_delete.add(current)
            stack.extend(children_by_parent.get(current, []))

        files_resp = (
            self.supabase.table('kb_files')
            .select('*')
            .eq('user_id', user_id)
            .in_('folder_id', list(to_delete))
            .execute()
        )
        files = getattr(files_resp, 'data', None) or []
        for file_row in files:
            delete_stored_file(file_row['storage_path'])
            self._delete_vectors_by_file_id(file_row['id'])

        (
            self.supabase.table('kb_files')
            .delete()
            .eq('user_id', user_id)
            .in_('folder_id', list(to_delete))
            .execute()
        )
        (
            self.supabase.table('kb_folders')
            .delete()
            .eq('user_id', user_id)
            .in_('id', list(to_delete))
            .execute()
        )

    def build_tree(self, *, user_id: str) -> dict[str, Any]:
        """Build nested folder tree with attached files."""
        folders = self.list_folders(user_id=user_id)
        files = (
            self.supabase.table('kb_files')
            .select('*')
            .eq('user_id', user_id)
            .order('created_at', desc=True)
            .execute()
        )
        files_data = getattr(files, 'data', None) or []

        nodes: dict[str, dict[str, Any]] = {
            folder['id']: {**folder, 'children': [], 'files': []}
            for folder in folders
        }
        roots: list[dict[str, Any]] = []

        for folder in folders:
            parent_id = folder.get('parent_id')
            if parent_id and parent_id in nodes:
                nodes[parent_id]['children'].append(nodes[folder['id']])
            else:
                roots.append(nodes[folder['id']])

        for kb_file in files_data:
            folder_id = kb_file.get('folder_id')
            if folder_id and folder_id in nodes:
                nodes[folder_id]['files'].append(kb_file)

        root_files = [f for f in files_data if not f.get('folder_id')]
        return {'folders': roots, 'root_files': root_files}

    def get_descendant_folder_ids(
        self, *, user_id: str, folder_ids: list[str]
    ) -> list[str]:
        """Expand selected folders to include all descendants."""
        if not folder_ids:
            return []

        folders = self.list_folders(user_id=user_id)
        children_by_parent: dict[str | None, list[str]] = defaultdict(list)
        all_folder_ids = {folder['id'] for folder in folders}
        for folder in folders:
            children_by_parent[folder.get('parent_id')].append(folder['id'])

        expanded: set[str] = set()
        stack = [
            folder_id
            for folder_id in folder_ids
            if folder_id in all_folder_ids
        ]
        while stack:
            current = stack.pop()
            if current in expanded:
                continue
            expanded.add(current)
            stack.extend(children_by_parent.get(current, []))
        return sorted(expanded)

    def _delete_vectors_by_file_id(self, file_id: str) -> None:
        selector = Filter(
            must=[
                FieldCondition(key='file_id', match=MatchValue(value=file_id))
            ]
        )
        for collection in self._all_collection_names():
            try:
                self.qdrant.delete(
                    collection_name=collection, points_selector=selector
                )
            except Exception:
                continue

    def delete_vectors_for_file(self, file_id: str) -> None:
        """Delete file vectors from all collections."""
        self._delete_vectors_by_file_id(file_id)

    def _update_vectors_folder_id(
        self, *, file_id: str, folder_id: str | None
    ) -> None:
        selector = Filter(
            must=[
                FieldCondition(key='file_id', match=MatchValue(value=file_id))
            ]
        )
        for collection in self._all_collection_names():
            try:
                points, _ = self.qdrant.scroll(
                    collection_name=collection,
                    scroll_filter=selector,
                    with_payload=True,
                    with_vectors=True,
                    limit=10_000,
                )
                if not points:
                    continue
                for point in points:
                    point.payload = {
                        **(point.payload or {}),
                        'folder_id': folder_id,
                        'folder_scope': folder_id or 'root',
                    }
                self.qdrant.upsert(collection_name=collection, points=points)
            except Exception:
                continue

    def _all_collection_names(self) -> list[str]:
        try:
            collections = self.qdrant.get_collections().collections
            return [collection.name for collection in collections]
        except Exception:
            return [
                Config.qdrant_text_collection,
                Config.qdrant_image_collection,
                Config.qdrant_video_collection,
            ]
