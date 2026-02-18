create table if not exists public.query_history (
  id bigserial primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  query text not null,
  answer text not null,
  retrieved_docs jsonb,
  created_at timestamptz not null default now()
);

alter table public.query_history enable row level security;

create policy "select_own_history"
on public.query_history
for select
using (auth.uid() = user_id);

create policy "insert_own_history"
on public.query_history
for insert
with check (auth.uid() = user_id);
