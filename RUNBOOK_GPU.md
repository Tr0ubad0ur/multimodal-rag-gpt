# GPU Runbook

## 1. Prerequisites on the Linux server

- NVIDIA driver installed
- Docker with NVIDIA Container Toolkit
- `docker compose` available
- Repo cloned

Quick checks:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 2. Prepare environment

```bash
cp .env.gpu.example .env.gpu
```

Fill in:

- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`
- `ADMIN_API_KEY`

If Supabase runs outside Docker on the same server, replace `host.docker.internal`
with the real hostname or IP.

## 3. Preflight

Local `uv` run on the server:

```bash
uv sync
uv run python scripts/check_gpu_env.py
```

Expected: `cuda_available: true` and two GPUs in the report.

## 4. Build and start

```bash
docker compose -f docker-compose.gpu.yml build
docker compose -f docker-compose.gpu.yml up -d
```

## 5. Health checks

```bash
curl -s http://localhost:18000/health/ready | jq .
curl -s http://localhost:18010/health/ready | jq .
```

## 6. Model smoke test

Inside the repo on the server:

```bash
uv run python scripts/check_model_gpu.py --model Qwen/Qwen2.5-VL-7B-Instruct
```

Recommended model order is documented in [MODEL_SHORTLIST_GPU.md](MODEL_SHORTLIST_GPU.md).

If you want the container path instead:

```bash
docker compose -f docker-compose.gpu.yml exec rag-web-gpu \
  uv run python scripts/check_model_gpu.py --model Qwen/Qwen2.5-VL-7B-Instruct
```

## 7. API smoke test

```bash
curl -X POST "http://localhost:18000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query":"Кратко опиши, что умеет система","top_k":3,"model":"Qwen/Qwen2.5-VL-7B-Instruct"}'
```

## 8. Logs

```bash
docker compose -f docker-compose.gpu.yml logs -f rag-web-gpu
docker compose -f docker-compose.gpu.yml logs -f rag-worker-gpu
```

## 9. Stop

```bash
docker compose -f docker-compose.gpu.yml down
```

Model cache is preserved in Docker volumes `hf_cache` and `torch_cache`.

## 10. Tuning knobs

Main env variables for large models:

- `LLM_TORCH_DTYPE=float16`
- `LLM_DEVICE_MAP=auto`
- `LLM_ATTN_IMPLEMENTATION=sdpa`
- `LLM_LOW_CPU_MEM_USAGE=true`
- `LLM_LOAD_IN_4BIT=false`

For a 7B model on 2x RTX 3080, start with these defaults before trying 4-bit.
