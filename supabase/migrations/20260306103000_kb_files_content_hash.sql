alter table public.kb_files
  add column if not exists content_hash text;

create index if not exists kb_files_user_folder_hash_idx
on public.kb_files(user_id, folder_id, content_hash);
