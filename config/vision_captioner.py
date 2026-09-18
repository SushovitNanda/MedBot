"""
Local vision captioning for PDF ingest (Hugging Face BLIP on GPU/CPU).
Chat/RAG continues to use cloud APIs — this module is ingest-only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from config.settings import (
    HF_TOKEN,
    MODELS_CACHE_DIR,
    USE_LOCAL_VISION_CAPTION,
    VISION_CAPTION_BATCH_SIZE,
    VISION_CAPTION_MODEL,
    resolve_local_device,
)

logger = logging.getLogger(__name__)

_lock = Lock()
_processor: BlipProcessor | None = None
_model: BlipForConditionalGeneration | None = None


def _cache_path() -> Path:
    return MODELS_CACHE_DIR / VISION_CAPTION_MODEL.replace("/", "__")


def _hf_kwargs() -> dict:
    return {"token": HF_TOKEN} if HF_TOKEN else {}


def get_vision_captioner() -> tuple[BlipProcessor, BlipForConditionalGeneration]:
    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model

    with _lock:
        if _processor is not None and _model is not None:
            return _processor, _model

        device = resolve_local_device()
        cache = _cache_path()
        kw = _hf_kwargs()
        source = str(cache) if cache.exists() else VISION_CAPTION_MODEL

        logger.info("Loading vision caption model '%s' on %s ...", source, device)

        _processor = BlipProcessor.from_pretrained(source, **kw)
        _model = BlipForConditionalGeneration.from_pretrained(source, **kw)

        if source == VISION_CAPTION_MODEL:
            cache.mkdir(parents=True, exist_ok=True)
            _processor.save_pretrained(str(cache))
            _model.save_pretrained(str(cache))

        _model.eval()
        _model.to(device)
        if device == "cuda":
            _model = _model.half()

        logger.info("Vision caption model ready on %s.", device)

    return _processor, _model


def caption_images_batch(images: list[Image.Image]) -> list[str]:
    if not images:
        return []

    if not USE_LOCAL_VISION_CAPTION:
        return [_fallback_caption(img) for img in images]

    processor, model = get_vision_captioner()
    device = resolve_local_device()
    results: list[str] = []

    for i in range(0, len(images), VISION_CAPTION_BATCH_SIZE):
        batch = [img.convert("RGB") for img in images[i : i + VISION_CAPTION_BATCH_SIZE]]
        try:
            inputs = processor(images=batch, return_tensors="pt").to(device)
            if device == "cuda":
                inputs = {
                    k: v.half() if v.is_floating_point() else v
                    for k, v in inputs.items()
                }

            with torch.no_grad():
                ids = model.generate(**inputs, max_length=140, num_beams=3)

            for raw in processor.batch_decode(ids, skip_special_tokens=True):
                text = (raw or "").strip() or "Medical textbook figure."
                results.append(
                    f"[Medical figure] {text} "
                    "(diagram, flowchart, table graphic, or illustration.)"
                )
        except Exception as exc:
            logger.warning("BLIP batch caption failed: %s", exc)
            results.extend(_fallback_caption(img) for img in batch)

    return results


def caption_image(pil_img: Image.Image) -> str:
    return caption_images_batch([pil_img])[0]


def _fallback_caption(img: Image.Image) -> str:
    return (
        f"Medical textbook figure ({img.width}x{img.height}px). "
        "Caption unavailable — see image file."
    )


def warmup_vision_captioner() -> None:
    if USE_LOCAL_VISION_CAPTION:
        get_vision_captioner()
