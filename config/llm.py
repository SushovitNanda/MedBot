"""
config/llm.py
LLM calls with fallback: Gemini 3.1 Flash Lite → Gemini 2.0 Flash → Groq Llama 3.3.
"""

import logging
from collections.abc import Iterator

import google.generativeai as genai
from groq import Groq

from config.settings import (
    FALLBACK_LLM,
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    PRIMARY_LLM,
    SECONDARY_LLM,
)

logger = logging.getLogger(__name__)

genai.configure(api_key=GOOGLE_API_KEY)
_groq = Groq(api_key=GROQ_API_KEY)

GEMINI_MODEL_CHAIN: list[str] = [PRIMARY_LLM, SECONDARY_LLM]


def generate_text(prompt: str, max_tokens: int = 1024) -> str:
    """Non-streaming completion through the model cascade."""
    for model_name in GEMINI_MODEL_CHAIN:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response.text:
                return response.text.strip()
        except Exception as exc:
            logger.warning("Gemini %s failed: %s", model_name, exc)

    logger.warning("All Gemini models failed; using Groq %s.", FALLBACK_LLM)
    try:
        resp = _groq.chat.completions.create(
            model=FALLBACK_LLM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content
        return (content or "").strip()
    except Exception as exc:
        raise RuntimeError(
            f"All LLM providers failed (Gemini + Groq {FALLBACK_LLM}). "
            f"Check GOOGLE_API_KEY and GROQ_API_KEY in .env. Groq error: {exc}"
        ) from exc


def stream_text(
    prompt: str,
    *,
    system_prompt: str | None = None,
) -> Iterator[str]:
    """Streaming completion through the model cascade."""
    for model_name in GEMINI_MODEL_CHAIN:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            return
        except Exception as exc:
            logger.warning("Gemini stream %s failed: %s", model_name, exc)

    logger.warning("All Gemini streams failed; using Groq %s.", FALLBACK_LLM)
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        stream = _groq.chat.completions.create(
            model=FALLBACK_LLM,
            messages=messages,
            stream=True,
            max_tokens=1500,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as exc:
        raise RuntimeError(
            f"All LLM providers failed (Gemini + Groq {FALLBACK_LLM}). "
            f"Check GOOGLE_API_KEY and GROQ_API_KEY in .env. Groq error: {exc}"
        ) from exc


def generate_with_image(prompt: str, image_bytes: bytes, mime_type: str = "image/png") -> str:
    """Vision captioning with the same Gemini cascade (no Groq vision)."""
    part = {"mime_type": mime_type, "data": image_bytes}
    for model_name in GEMINI_MODEL_CHAIN:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt, part])
            if response.text:
                return response.text.strip()
        except Exception as exc:
            logger.warning("Gemini vision %s failed: %s", model_name, exc)
    return "Medical diagram (caption unavailable)"
