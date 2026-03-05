create table if not exists public.admin_rate_limits (
  scope text not null,
  bucket_start timestamptz not null,
  count int not null default 0,
  primary key (scope, bucket_start)
);

create index if not exists admin_rate_limits_bucket_idx
on public.admin_rate_limits(bucket_start);

create or replace function public.check_admin_rate_limit(
  p_scope text,
  p_limit int default 60,
  p_window_seconds int default 60
)
returns boolean
language plpgsql
security definer
as $$
declare
  v_now timestamptz := now();
  v_window_seconds int := greatest(1, coalesce(p_window_seconds, 60));
  v_bucket_start timestamptz;
  v_limit int := greatest(1, coalesce(p_limit, 60));
  v_count int;
begin
  v_bucket_start := date_trunc('minute', v_now);

  insert into public.admin_rate_limits (scope, bucket_start, count)
  values (p_scope, v_bucket_start, 1)
  on conflict (scope, bucket_start)
  do update set count = public.admin_rate_limits.count + 1
  returning count into v_count;

  delete from public.admin_rate_limits
  where bucket_start < (v_now - make_interval(secs => v_window_seconds * 2));

  return v_count <= v_limit;
end;
$$;

grant execute on function public.check_admin_rate_limit(text, int, int) to authenticated;
grant execute on function public.check_admin_rate_limit(text, int, int) to service_role;
