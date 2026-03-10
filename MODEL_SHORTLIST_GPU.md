# GPU Model Shortlist

Target environment:

- Linux server
- 2x RTX 3080
- short-lived run, not permanent hosting

## Test order

1. `Qwen/Qwen2-VL-2B-Instruct`
2. `Qwen/Qwen2.5-VL-7B-Instruct`
3. `llava-hf/llava-onevision-qwen2-7b-ov-hf`

## Why this order

### 1. `Qwen/Qwen2-VL-2B-Instruct`

Use it as the sanity model:

- already integrated in the project
- lowest VRAM risk
- confirms end-to-end path: auth, retrieval, image input, answer generation

Official model card:

- https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct

Notes:

- Qwen recommends latest `transformers` from source for this family

### 2. `Qwen/Qwen2.5-VL-7B-Instruct`

Use it as the main stronger multimodal candidate:

- materially stronger than the 2B model
- still aligned with the current backend contract: text + image, and optional video understanding
- same Qwen family, so the integration path is close to the already working one

Official model card:

- https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

Notes from the model card:

- Qwen recommends latest `transformers` from source
- for video, `qwen-vl-utils[decord]` is preferred on Linux
- for non-Linux environments, fallback without `decord` is possible

### 3. `llava-hf/llava-onevision-qwen2-7b-ov-hf`

Use it as the alternative 7B candidate:

- multimodal image/video capable
- official `transformers` support exists
- model card includes 4-bit and Flash-Attention optimization paths

Official model card:

- https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf

Notes from the model card:

- requires `transformers >= 4.45.0`
- examples use `torch.float16` on GPU
- 4-bit quantization through `bitsandbytes` is supported

## Not first-pass candidates

### `openai/gpt-oss-20b`

Reason:

- text-only
- much heavier than needed for the first university-server trial
- not aligned with the current multimodal path you want to validate

## Practical recommendation

On the university server:

1. smoke test the whole stack with `Qwen/Qwen2-VL-2B-Instruct`
2. switch to `Qwen/Qwen2.5-VL-7B-Instruct`
3. if VRAM or latency is bad, try `llava-hf/llava-onevision-qwen2-7b-ov-hf`

This gives you one safe baseline and two stronger candidates without jumping into Linux-only omni stacks.
