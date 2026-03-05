create table if not exists public.ingest_jobs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade,
  owner_type text not null check (owner_type in ('user', 'guest')),
  owner_id text not null,
  file_id uuid not null,
  filename text not null,
  mime text not null,
  source_path text,
  folder_id uuid references public.kb_folders(id) on delete set null,
  status text not null default 'queued' check (status in ('queued', 'processing', 'completed', 'failed')),
  attempt int not null default 0,
  error text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  started_at timestamptz,
  finished_at timestamptz
);

create index if not exists ingest_jobs_owner_idx on public.ingest_jobs(owner_type, owner_id);
create index if not exists ingest_jobs_status_idx on public.ingest_jobs(status);
create index if not exists ingest_jobs_file_id_idx on public.ingest_jobs(file_id);
create index if not exists ingest_jobs_user_id_idx on public.ingest_jobs(user_id);
create index if not exists ingest_jobs_created_at_idx on public.ingest_jobs(created_at desc);

alter table public.ingest_jobs enable row level security;

create policy "select_own_ingest_jobs"
on public.ingest_jobs
for select
using (user_id is not null and auth.uid() = user_id);

create policy "insert_own_ingest_jobs"
on public.ingest_jobs
for insert
with check (user_id is not null and auth.uid() = user_id);

create policy "update_own_ingest_jobs"
on public.ingest_jobs
for update
using (user_id is not null and auth.uid() = user_id)
with check (user_id is not null and auth.uid() = user_id);
