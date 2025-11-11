from backend.core.embeddings import text_embedding
from backend.core.image_embeddings import image_embedding_from_path
from backend.core import vectordb
from backend.core.llm import get_llm
from typing import List, Dict, Any
import heapq

# Параметры поиска
TEXT_K = 5
IMAGE_K = 5
COMBINED_K = 6  # сколько элементов в итоговом контексте передавать в LLM

def merge_and_rank(text_hits: List[Dict], image_hits: List[Dict], top_k: int = COMBINED_K):
    """
    Объединяем результаты из text_hits и image_hits в единый ранжированный список.
    Удаляем дубликаты по payload (например, path или source).
    """
    merged = []
    seen = set()
    for h in text_hits + image_hits:
        # Уникальность по (type + id или path)
        payload = h.get("payload", {})
        unique_key = None
        if payload.get("type") == "text":
            unique_key = f"text::{payload.get('source')}::{payload.get('chunk_id', '')}"
        else:
            unique_key = f"image::{payload.get('path')}"
        if unique_key in seen:
            continue
        seen.add(unique_key)
        merged.append(h)
    # Сортировка по score (возрастающий? qdrant score - чем выше лучше)
    merged.sort(key=lambda x: x.get("score", 0), reverse=True)
    return merged[:top_k]

def build_context(merged_hits: List[Dict]):
    """
    Собираем текстовый контекст из merged_hits: для текстов - берем сам текст в payload (если есть),
    или нужно хранить текст отдельно в БД/файлах и подтягивать по source+chunk_id.
    Для изображений - добавляем метаданные (путь, подпись), возможно OCR-выдержки.
    """
    ctx_pieces = []
    sources = []
    for i, h in enumerate(merged_hits, start=1):
        payload = h.get("payload", {})
        if payload.get("type") == "text":
            text = payload.get("text", "<текст отсутствует в payload>")
            src = payload.get("source", "unknown")
            ctx_pieces.append(f"[{i}] (text) {text}")
            sources.append(f"[{i}] {src}")
        else:
            path = payload.get("path", "unknown")
            caption = payload.get("caption", "")
            ctx_pieces.append(f"[{i}] (image) path={path}, caption={caption}")
            sources.append(f"[{i}] {path}")
    context = "\n\n".join(ctx_pieces)
    sources_str = "\n".join(sources)
    return context, sources_str

_PROMPT_TEMPLATE = """
Ты — помощник, который отвечает на вопросы, используя только предоставленные факты из контекста ниже.
Если ответ недоступен в контексте — честно скажи, что информации недостаточно.

Контекст:
{context}

Вопрос: {question}

Дай краткий, точный ответ. В конце укажи номера источников в формате [1], [2], ... из списка источников:
{sources}

Если ответ не однозначен — предложи варианты/следующие шаги.
"""

def ask_text_query(question: str):
    # 1) эмбеддинг текста
    q_vec = text_embedding(question)
    # 2) поиск в тексте и изображениях
    text_hits = vectordb.search_text(q_vec, limit=TEXT_K)
    image_hits = vectordb.search_images(q_vec, limit=IMAGE_K)
    # 3) объединяем и формируем контекст
    merged = merge_and_rank(text_hits, image_hits)
    context, sources = build_context(merged)
    prompt = _PROMPT_TEMPLATE.format(context=context, question=question, sources=sources)
    llm = get_llm()
    resp = llm.generate([{"role":"user","content":prompt}])
    # LangChain ChatOpenAI returns different shapes; здесь простой способ:
    # Если get_llm() — LangChain ChatOpenAI, лучше вызывать llm.chat or llm.call
    # Для совместимости: берем first content
    try:
        answer = resp.generations[0][0].text
    except Exception:
        # попытка альтернативного доступа
        answer = str(resp)
    return {"answer": answer, "sources": sources}

def ask_image_query(image_path: str, question: str = None):
    # 1) эмбедд изображения
    v = image_embedding_from_path(image_path)
    # 2) поиск похожих изображений и текстов
    image_hits = vectordb.search_images(v, limit=IMAGE_K)
    text_hits = vectordb.search_text(v, limit=TEXT_K)
    merged = merge_and_rank(text_hits, image_hits)
    context, sources = build_context(merged)
    if question is None:
        # если пользователь загрузил только изображение, создаём вопрос-подсказку
        question = "Опиши, что видно на изображении и какие релевантные текстовые фрагменты связаны с ним."
    prompt = _PROMPT_TEMPLATE.format(context=context, question=question, sources=sources)
    llm = get_llm()
    resp = llm.generate([{"role":"user","content":prompt}])
    try:
        answer = resp.generations[0][0].text
    except Exception:
        answer = str(resp)
    return {"answer": answer, "sources": sources}
