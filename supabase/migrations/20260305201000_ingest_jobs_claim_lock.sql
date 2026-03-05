alter table public.ingest_jobs
  add column if not exists claimed_by text,
  add column if not exists claim_expires_at timestamptz;

create index if not exists ingest_jobs_claim_expires_idx
on public.ingest_jobs(claim_expires_at);

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
    order by j.created_at
    for update skip locked
    limit v_batch
  ),
  claimed as (
    update public.ingest_jobs j
    set
      status = 'processing',
      claimed_by = p_worker_id,
      started_at = coalesce(j.started_at, v_now),
      claim_expires_at = v_now + make_interval(secs => v_lock_seconds)
    where j.id in (select id from candidates)
    returning j.*
  )
  select * from claimed;
end;
$$;

grant execute on function public.claim_ingest_jobs(text, int, int) to authenticated;
grant execute on function public.claim_ingest_jobs(text, int, int) to service_role;
