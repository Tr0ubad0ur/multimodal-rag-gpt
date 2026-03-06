import logging
from typing import Any, Dict, List

import requests
import torch
from PIL import Image
from transformers import AutoProcessor

from backend.utils.config_handler import Config

logger = logging.getLogger(__name__)

MODEL_NAME = Config.llm_model_name


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


class QwenVisionLLM:
    """Wrapper for a multimodal Vision-Text LLM (Qwen2-VL) to generate text from images and prompts.

    Attributes:
        processor: Processor for preparing images and text for the model.
        model: The loaded Vision2Seq model for multimodal inference.
    """

    def __init__(self) -> None:
        """Initialize the Vision LLM, loading the model and processor."""
        logger.info(f'Loading {Config.llm_model_name}...')
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model_cls = _resolve_auto_model_class()
        self.model = model_cls.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16
            if torch.backends.mps.is_available()
            else torch.float32,
            device_map='auto',
        )
        logger.info(f'{Config.llm_model_name} loaded successfully')

    def build_messages(self, prompt, image=None) -> List[Dict[str, Any]]:
        """Constructs a chat-style message payload for the model.

        Args:
            prompt (str): Text prompt to send to the model.
            image (str or PIL.Image.Image, optional): Path, URL, or PIL Image to include in the message.

        Returns:
            list[dict]: A list of message dictionaries ready for the processor.
        """
        content = []
        if image is not None:
            if isinstance(image, str):
                if image.startswith('http'):
                    response = requests.get(image, stream=True, timeout=10)
                    response.raise_for_status()
                    img = Image.open(response.raw).convert('RGB')
                else:
                    img = Image.open(image).convert('RGB')
            else:
                img = image
            content.append({'type': 'image', 'image': img})
        content.append({'type': 'text', 'text': prompt})
        messages = [{'role': 'user', 'content': content}]
        return messages

    def generate(self, prompt, context=None, image=None) -> str:
        """Generate a text response using the Vision LLM, optionally with context and/or image.

        Args:
            prompt (str): The main text prompt or question.
            context (list[dict], optional): A list of retrieved documents for RAG context. Each dict must contain 'text'.
            image (str or PIL.Image.Image, optional): Path, URL, or PIL image to include in generation.

        Returns:
            str: The generated text output from the model.
        """
        full_prompt = prompt
        if context:
            context_text = '\n'.join([d['text'] for d in context])
            full_prompt = f'Используя следующий контекст:\n{context_text}\n\nОтветьте на вопрос:\n{prompt}'

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

        result = self.processor.decode(
            output[0][inputs['input_ids'].shape[-1] :],
            skip_special_tokens=True,
        )
        return result


qwen_llm = QwenVisionLLM()


def _resolve_llm_backend(model: str | None) -> QwenVisionLLM:
    """Resolve an LLM backend by model name with fallback to default."""
    if not model or model == Config.llm_model_name:
        return qwen_llm
    logger.warning(
        'Unknown model "%s". Falling back to default model "%s".',
        model,
        Config.llm_model_name,
    )
    return qwen_llm


def get_llm_response(prompt, context=None, image=None, model=None) -> str:
    """Helper function to generate a response using the global QwenVisionLLM instance.

    Args:
        prompt (str): User prompt or question.
        context (list[dict], optional): Retrieved documents for context.
        image (str or PIL.Image.Image, optional): Image to include in generation.
        model (str | None, optional): Requested model identifier.

    Returns:
        str: Generated text from the LLM.
    """
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
