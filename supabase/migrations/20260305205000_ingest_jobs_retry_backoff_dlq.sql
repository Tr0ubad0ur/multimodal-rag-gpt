alter table public.ingest_jobs
  add column if not exists next_retry_at timestamptz,
  add column if not exists dead_lettered_at timestamptz;

create index if not exists ingest_jobs_next_retry_idx
on public.ingest_jobs(next_retry_at);

create table if not exists public.ingest_jobs_dlq (
  id bigserial primary key,
  job_id uuid not null references public.ingest_jobs(id) on delete cascade,
  user_id uuid references auth.users(id) on delete cascade,
  owner_type text not null,
  owner_id text not null,
  reason text not null,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists ingest_jobs_dlq_job_id_idx
on public.ingest_jobs_dlq(job_id);

create index if not exists ingest_jobs_dlq_user_id_idx
on public.ingest_jobs_dlq(user_id);

alter table public.ingest_jobs_dlq enable row level security;

create policy "select_own_ingest_jobs_dlq"
on public.ingest_jobs_dlq
for select
using (user_id is not null and auth.uid() = user_id);

create policy "insert_own_ingest_jobs_dlq"
on public.ingest_jobs_dlq
for insert
with check (user_id is not null and auth.uid() = user_id);

create or replace function public.claim_ingest_jobs(
  p_worker_id text,
  p_batch_size int default 10,
  p_lock_seconds int default 300
)
returns setof public.ingest_jobs
language plpgsql
security definer
as $$
declare
  v_now timestamptz := now();
  v_batch int := greatest(1, coalesce(p_batch_size, 10));
  v_lock_seconds int := greatest(30, coalesce(p_lock_seconds, 300));
begin
  return query
  with candidates as (
    select j.id
    from public.ingest_jobs j
    where j.owner_type = 'user'
      and j.status = 'queued'
      and (j.next_retry_at is null or j.next_retry_at <= v_now)
    order by j.created_at
    for update skip locked
    limit v_batch
  ),
  claimed as (
    update public.ingest_jobs j
    set
      status = 'processing',
      claimed_by = p_worker_id,
      started_at = v_now,
      claim_expires_at = v_now + make_interval(secs => v_lock_seconds),
      attempt = coalesce(j.attempt, 0) + 1,
      next_retry_at = null,
      error = null
    where j.id in (select id from candidates)
    returning j.*
  )
  select * from claimed;
end;
$$;

grant execute on function public.claim_ingest_jobs(text, int, int) to authenticated;
grant execute on function public.claim_ingest_jobs(text, int, int) to service_role;
