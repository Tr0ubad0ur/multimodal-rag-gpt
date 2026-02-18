import logging
from typing import Any, Dict, List

import requests
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from backend.utils.config_handler import Config

logger = logging.getLogger(__name__)

MODEL_NAME = Config.llm_model_name


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
        self.model = AutoModelForVision2Seq.from_pretrained(
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


def get_llm_response(prompt, context=None, image=None) -> str:
    """Helper function to generate a response using the global QwenVisionLLM instance.

    Args:
        prompt (str): User prompt or question.
        context (list[dict], optional): Retrieved documents for context.
        image (str or PIL.Image.Image, optional): Image to include in generation.

    Returns:
        str: Generated text from the LLM.
    """
    if context:
        by_source: Dict[str, List[str]] = {}
        for item in context:
            source = item.get('source') or 'unknown'
            by_source.setdefault(source, []).append(item.get('text', ''))

        summaries: List[str] = []
        for source, texts in by_source.items():
            source_text = '\n'.join(texts).strip()
            source_prompt = (
                'Сформулируй ОДНО короткое предложение, о чем этот источник. '
                'Используй только информацию из текста. '
                'Не повторяй слова "Источник", "Текст", "Запрос", не цитируй текст. '
                'Не добавляй лишние заголовки.\n\n'
                f'Текст:\n{source_text}\n\n'
                f'Запрос пользователя: {prompt}'
            )
            raw_summary = qwen_llm.generate(
                source_prompt, context=None, image=image
            )
            # Sanitize common prompt-echo artifacts
            summary = (
                raw_summary.replace('Источник:', '')
                .replace('Текст:', '')
                .replace('Запрос пользователя:', '')
                .replace('Summary:', '')
                .strip()
            )
            summaries.append(f'- {source}: {summary}')

        return '\n'.join(summaries)

    return qwen_llm.generate(prompt, context=None, image=image)
