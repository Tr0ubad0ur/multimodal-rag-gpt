import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, pipeline

from backend.utils.config_handler import Config

logger = logging.getLogger(__name__)

_BACKENDS: dict[str, 'BaseLLMBackend'] = {}


def _default_torch_dtype() -> torch.dtype:
    """Pick a conservative dtype for the current runtime."""
    if torch.cuda.is_available():
        return torch.float16
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def _resolve_torch_dtype() -> torch.dtype:
    """Resolve target dtype from env or runtime defaults."""
    raw = os.getenv('LLM_TORCH_DTYPE', 'auto').strip().lower()
    if raw in {'', 'auto'}:
        return _default_torch_dtype()
    mapping = {
        'float16': torch.float16,
        'fp16': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
        'float32': torch.float32,
        'fp32': torch.float32,
    }
    return mapping.get(raw, _default_torch_dtype())


def _resolve_device_map() -> str:
    """Resolve device_map override for model loading."""
    return os.getenv('LLM_DEVICE_MAP', 'auto').strip() or 'auto'


def _resolve_attn_implementation() -> str | None:
    """Resolve attention implementation override."""
    raw = os.getenv('LLM_ATTN_IMPLEMENTATION', '').strip().lower()
    if not raw or raw == 'auto':
        if torch.cuda.is_available():
            return 'sdpa'
        return None
    return raw


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean env flag."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {'1', 'true', 'yes', 'on'}


def _build_common_model_kwargs() -> dict[str, Any]:
    """Build common model loading kwargs for large GPU-hosted models."""
    kwargs: dict[str, Any] = {
        'torch_dtype': _resolve_torch_dtype(),
        'device_map': _resolve_device_map(),
        'low_cpu_mem_usage': _env_flag('LLM_LOW_CPU_MEM_USAGE', True),
    }
    attn_implementation = _resolve_attn_implementation()
    if attn_implementation:
        kwargs['attn_implementation'] = attn_implementation
    if _env_flag('LLM_LOAD_IN_4BIT', False):
        kwargs['load_in_4bit'] = True
    return kwargs


def _resolve_auto_model_class():
    """Resolve compatible transformers AutoModel class for vision-text."""
    try:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq
    except ImportError:
        pass

    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    except ImportError as exc:
        raise ImportError(
            'No compatible vision-text auto model class found in transformers. '
            'Install a newer transformers version.'
        ) from exc


class BaseLLMBackend(ABC):
    """Common interface for text and multimodal backends."""

    supports_image: bool = False

    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: list[dict[str, Any]] | None = None,
        image: str | Image.Image | None = None,
    ) -> str:
        """Generate answer for prompt and optional context/image."""


class QwenVisionLLM(BaseLLMBackend):
    """Vision-text backend for multimodal requests."""

    supports_image = True

    def __init__(self, model_name: str) -> None:
        """Initialize the vision backend."""
        logger.info('Loading vision model %s...', model_name)
        self.model_name = model_name
        self.model_kwargs = _build_common_model_kwargs()
        self.processor = AutoProcessor.from_pretrained(model_name)
        model_cls = _resolve_auto_model_class()
        self.model = model_cls.from_pretrained(
            model_name,
            **self.model_kwargs,
        )
        logger.info('Vision model %s loaded', model_name)

    def build_messages(
        self, prompt: str, image: str | Image.Image | None = None
    ) -> List[Dict[str, Any]]:
        """Build chat template payload for Qwen vision models."""
        content: list[dict[str, Any]] = []
        if image is not None:
            if isinstance(image, str):
                if image.startswith('http'):
                    response = requests.get(image, stream=True, timeout=10)
                    response.raise_for_status()
                    resolved_image = Image.open(response.raw).convert('RGB')
                else:
                    resolved_image = Image.open(image).convert('RGB')
            else:
                resolved_image = image
            content.append({'type': 'image', 'image': resolved_image})
        content.append({'type': 'text', 'text': prompt})
        return [{'role': 'user', 'content': content}]

    def generate(
        self,
        prompt: str,
        context: list[dict[str, Any]] | None = None,
        image: str | Image.Image | None = None,
    ) -> str:
        """Generate answer with optional image input."""
        full_prompt = prompt
        if context:
            context_text = '\n'.join(item.get('text', '') for item in context)
            full_prompt = (
                'Используя следующий контекст:\n'
                f'{context_text}\n\n'
                'Ответьте на вопрос:\n'
                f'{prompt}'
            )

        messages = self.build_messages(full_prompt, image)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors='pt',
        ).to(self.model.device)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=Config.llm_max_new_tokens,
                do_sample=False,
            )

        return self.processor.decode(
            output[0][inputs['input_ids'].shape[-1] :],
            skip_special_tokens=True,
        )


class LlavaOneVisionBackend(BaseLLMBackend):
    """LLaVA-OneVision backend for image-text-to-text requests."""

    supports_image = True

    def __init__(self, model_name: str) -> None:
        """Initialize a Llava-OneVision model with GPU-friendly kwargs."""
        from transformers import LlavaOnevisionForConditionalGeneration

        logger.info('Loading Llava-OneVision model %s...', model_name)
        self.model_name = model_name
        self.model_kwargs = _build_common_model_kwargs()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name,
            **self.model_kwargs,
        )
        logger.info('Llava-OneVision model %s loaded', model_name)

    def _resolve_image(
        self, image: str | Image.Image | None
    ) -> Image.Image | None:
        """Resolve image input into a PIL image."""
        if image is None:
            return None
        if isinstance(image, Image.Image):
            return image.convert('RGB')
        if image.startswith('http'):
            response = requests.get(image, stream=True, timeout=10)
            response.raise_for_status()
            return Image.open(response.raw).convert('RGB')
        return Image.open(image).convert('RGB')

    def generate(
        self,
        prompt: str,
        context: list[dict[str, Any]] | None = None,
        image: str | Image.Image | None = None,
    ) -> str:
        """Generate answer using Llava-OneVision."""
        full_prompt = prompt
        if context:
            context_text = '\n'.join(item.get('text', '') for item in context)
            full_prompt = (
                'Используя следующий контекст:\n'
                f'{context_text}\n\n'
                'Ответьте на вопрос:\n'
                f'{prompt}'
            )

        content: list[dict[str, str]] = [{'type': 'text', 'text': full_prompt}]
        resolved_image = self._resolve_image(image)
        if resolved_image is not None:
            content.append({'type': 'image'})
        prompt_text = self.processor.apply_chat_template(
            [{'role': 'user', 'content': content}],
            add_generation_prompt=True,
        )

        inputs = self.processor(
            images=resolved_image,
            text=prompt_text,
            return_tensors='pt',
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=Config.llm_max_new_tokens,
                do_sample=False,
            )

        generated = output[0][inputs['input_ids'].shape[-1] :]
        return self.processor.decode(
            generated, skip_special_tokens=True
        ).strip()


class GPTOSSBackend(BaseLLMBackend):
    """Text-only backend for OpenAI gpt-oss models hosted on Hugging Face."""

    supports_image = False

    def __init__(self, model_name: str) -> None:
        """Initialize text generation pipeline."""
        logger.info('Loading text model %s...', model_name)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        pipeline_kwargs = {
            'torch_dtype': _resolve_torch_dtype(),
            'device_map': _resolve_device_map(),
        }
        self.pipeline = pipeline(
            task='text-generation',
            model=model_name,
            tokenizer=self.tokenizer,
            **pipeline_kwargs,
        )
        logger.info('Text model %s loaded', model_name)

    def generate(
        self,
        prompt: str,
        context: list[dict[str, Any]] | None = None,
        image: str | Image.Image | None = None,
    ) -> str:
        """Generate answer for text-only prompts."""
        if image is not None:
            raise ValueError(
                f'Model "{self.model_name}" does not support image input'
            )

        full_prompt = prompt
        if context:
            context_text = '\n'.join(item.get('text', '') for item in context)
            full_prompt = (
                'Используя следующий контекст:\n'
                f'{context_text}\n\n'
                'Ответьте на вопрос:\n'
                f'{prompt}'
            )

        outputs = self.pipeline(
            [{'role': 'user', 'content': full_prompt}],
            max_new_tokens=Config.llm_max_new_tokens,
        )
        generated = outputs[0].get('generated_text')
        if isinstance(generated, list) and generated:
            last_message = generated[-1]
            if isinstance(last_message, dict):
                return str(last_message.get('content', '')).strip()
            return str(last_message).strip()
        return str(generated or '').strip()


def _build_backend(model_name: str) -> BaseLLMBackend:
    """Instantiate backend implementation for model name."""
    normalized = model_name.lower()
    if 'gpt-oss' in normalized:
        return GPTOSSBackend(model_name)
    if 'llava-onevision' in normalized:
        return LlavaOneVisionBackend(model_name)
    return QwenVisionLLM(model_name)


def _resolve_llm_backend(model: str | None) -> BaseLLMBackend:
    """Resolve backend by requested model name with lazy caching."""
    model_name = model or Config.llm_model_name
    backend = _BACKENDS.get(model_name)
    if backend is None:
        backend = _build_backend(model_name)
        _BACKENDS[model_name] = backend
    return backend


def get_llm_response(prompt, context=None, image=None, model=None) -> str:
    """Generate a response using the configured backend."""
    backend = _resolve_llm_backend(model)

    if Config.llm_fast_mode:
        if context:
            snippets: List[str] = []
            for item in context[: max(1, Config.rag_max_context_docs)]:
                text = (item.get('text') or '').strip()
                if text:
                    snippets.append(text[:300])
            if snippets:
                joined = ' '.join(snippets)
                return f'Быстрый режим: {joined[:600]}'
        return 'Быстрый режим: данных недостаточно для уверенного ответа.'

    if context:
        limited_docs = context[: max(1, Config.rag_max_context_docs)]
        context_blocks: List[str] = []
        remaining = max(200, Config.rag_max_context_chars)
        for idx, item in enumerate(limited_docs, start=1):
            source = item.get('source') or 'unknown'
            text = (item.get('text') or '').strip()
            if not text:
                continue
            chunk = text[:remaining]
            context_blocks.append(f'[{idx}] {source}\n{chunk}')
            remaining -= len(chunk)
            if remaining <= 0:
                break

        combined_context = '\n\n'.join(context_blocks)
        combined_prompt = (
            'Ответь кратко и по делу, опираясь только на контекст ниже. '
            'Если данных недостаточно, так и напиши.\n\n'
            f'Контекст:\n{combined_context}\n\n'
            f'Вопрос: {prompt}'
        )
        return backend.generate(combined_prompt, context=None, image=image)

    return backend.generate(prompt, context=None, image=image)
