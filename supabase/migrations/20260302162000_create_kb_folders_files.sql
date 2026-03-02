create table if not exists public.kb_folders (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  parent_id uuid references public.kb_folders(id) on delete cascade,
  name text not null,
  created_at timestamptz not null default now()
);

create index if not exists kb_folders_user_id_idx on public.kb_folders(user_id);
create index if not exists kb_folders_parent_id_idx on public.kb_folders(parent_id);

create table if not exists public.kb_files (
  id uuid primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  folder_id uuid references public.kb_folders(id) on delete set null,
  filename text not null,
  mime text not null,
  size bigint not null,
  storage_path text not null,
  created_at timestamptz not null default now()
);

create index if not exists kb_files_user_id_idx on public.kb_files(user_id);
create index if not exists kb_files_folder_id_idx on public.kb_files(folder_id);

alter table public.kb_folders enable row level security;
alter table public.kb_files enable row level security;

create policy "select_own_kb_folders"
on public.kb_folders
for select
using (auth.uid() = user_id);

create policy "insert_own_kb_folders"
on public.kb_folders
for insert
with check (auth.uid() = user_id);

create policy "update_own_kb_folders"
on public.kb_folders
for update
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

create policy "delete_own_kb_folders"
on public.kb_folders
for delete
using (auth.uid() = user_id);

create policy "select_own_kb_files"
on public.kb_files
for select
using (auth.uid() = user_id);

create policy "insert_own_kb_files"
on public.kb_files
for insert
with check (auth.uid() = user_id);

create policy "update_own_kb_files"
on public.kb_files
for update
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

create policy "delete_own_kb_files"
on public.kb_files
for delete
using (auth.uid() = user_id);
