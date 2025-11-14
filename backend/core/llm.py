# Загрузка модели
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import requests
import logging

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

class QwenVisionLLM:
    """Wrapper for a multimodal Vision-Text LLM (Qwen2-VL) to generate text from images and prompts.

    Attributes:
        processor: Processor for preparing images and text for the model.
        model: The loaded Vision2Seq model for multimodal inference.
    """
    def __init__(self):
        """Initialize the Vision LLM, loading the model and processor."""
        logger.info("🔄 Loading Qwen2-VL-2B-Instruct...")
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
            device_map="auto"
        )
        logger.info("✅ Qwen2-VL-2B loaded successfully!")

    def build_messages(self, prompt, image=None):
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
                if image.startswith("http"):
                    img = Image.open(requests.get(image, stream=True).raw).convert("RGB")
                else:
                    img = Image.open(image).convert("RGB")
            else:
                img = image
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        return messages

    def generate(self, prompt, context=None, image=None):
        """Generate a text response using the Vision LLM, optionally with context and/or image.

        Args:
            prompt (str): The main text prompt or question.
            context (list[dict], optional): A list of retrieved documents for RAG context. Each dict must contain 'text'.
            image (str or PIL.Image.Image, optional): Path, URL, or PIL image to include in generation.

        Returns:
            str: The generated text output from the model.
        """
        # Добавляем контекст retrieved_docs
        full_prompt = prompt
        if context:
            context_text = "\n".join([d['text'] for d in context])
            full_prompt = f"Используя следующий контекст:\n{context_text}\n\nОтветьте на вопрос:\n{prompt}"

        messages = self.build_messages(full_prompt, image)

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False
        )

        result = self.processor.decode(output[0][inputs["input_ids"].shape[-1]:])
        return result

# Singleton
qwen_llm = QwenVisionLLM()

def get_llm_response(prompt, context=None, image=None):
    """Helper function to generate a response using the global QwenVisionLLM instance.

    Args:
        prompt (str): User prompt or question.
        context (list[dict], optional): Retrieved documents for context.
        image (str or PIL.Image.Image, optional): Image to include in generation.

    Returns:
        str: Generated text from the LLM.
    """
    return qwen_llm.generate(prompt, context=context, image=image)
