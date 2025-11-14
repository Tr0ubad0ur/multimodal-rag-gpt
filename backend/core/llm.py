import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import requests

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

class QwenVisionLLM:
    def __init__(self):
        print("🔄 Loading Qwen2-VL-2B-Instruct...")
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
            device_map="auto"
        )
        print("✅ Qwen2-VL-2B loaded successfully!")

    def build_messages(self, prompt, image=None):
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
    return qwen_llm.generate(prompt, context=context, image=image)
