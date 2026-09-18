"""
config/model_loader.py
Loads and caches local embedder, reranker, and CLIP (GPU when available).
"""

import logging
from pathlib import Path
from threading import Lock

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from config.settings import (
    CLIP_MODEL,
    EMBEDDING_MODEL,
    MODELS_CACHE_DIR,
    RERANKER_MODEL,
    resolve_local_device,
)

logger = logging.getLogger(__name__)

DEVICE = resolve_local_device()

_embed_lock = Lock()
_rerank_lock = Lock()
_clip_lock = Lock()

_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_clip_model: CLIPModel | None = None
_clip_proc: CLIPProcessor | None = None


def _model_cache_path(model_name: str) -> Path:
    safe = model_name.replace("/", "__")
    return MODELS_CACHE_DIR / safe


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is not None:
        return _embedder

    with _embed_lock:
        if _embedder is not None:
            return _embedder

        cache_path = _model_cache_path(EMBEDDING_MODEL)
        if cache_path.exists():
            logger.info("Loading embedder from cache: %s (%s)", cache_path, DEVICE)
            _embedder = SentenceTransformer(str(cache_path), device=DEVICE)
        else:
            logger.info("Downloading embedder: %s → %s", EMBEDDING_MODEL, DEVICE)
            _embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
            _embedder.save(str(cache_path))

    return _embedder


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is not None:
        return _reranker

    with _rerank_lock:
        if _reranker is not None:
            return _reranker

        cache_path = _model_cache_path(RERANKER_MODEL)
        if cache_path.exists():
            logger.info("Loading reranker from cache: %s (%s)", cache_path, DEVICE)
            _reranker = CrossEncoder(str(cache_path), device=DEVICE)
        else:
            logger.info("Downloading reranker: %s → %s", RERANKER_MODEL, DEVICE)
            _reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE)
            _reranker.model.save_pretrained(str(cache_path))
            _reranker.tokenizer.save_pretrained(str(cache_path))

    return _reranker


def get_clip() -> tuple[CLIPModel, CLIPProcessor]:
    global _clip_model, _clip_proc
    if _clip_model is not None and _clip_proc is not None:
        return _clip_model, _clip_proc

    with _clip_lock:
        if _clip_model is not None and _clip_proc is not None:
            return _clip_model, _clip_proc

        cache_path = _model_cache_path(CLIP_MODEL)
        if cache_path.exists():
            logger.info("Loading CLIP from cache: %s (%s)", cache_path, DEVICE)
            _clip_model = CLIPModel.from_pretrained(str(cache_path))
            _clip_proc = CLIPProcessor.from_pretrained(str(cache_path))
        else:
            logger.info("Downloading CLIP: %s → %s", CLIP_MODEL, DEVICE)
            _clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
            _clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
            _clip_model.save_pretrained(str(cache_path))
            _clip_proc.save_pretrained(str(cache_path))

        _clip_model.eval()
        _clip_model.to(DEVICE)

    return _clip_model, _clip_proc


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=len(texts) > 64,
        normalize_embeddings=True,
        device=DEVICE,
    )
    return vectors.tolist()


def embed_single(text: str) -> list[float]:
    return embed_texts([text])[0]


def embed_image(image) -> list[float]:
    clip_model, clip_proc = get_clip()
    inputs = clip_proc(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().cpu().tolist()


def embed_images_batch(images: list) -> list[list[float]]:
    if not images:
        return []
    clip_model, clip_proc = get_clip()
    inputs = clip_proc(images=images, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return [row.cpu().tolist() for row in features]


def embed_text_for_image_search(text: str) -> list[float]:
    clip_model, clip_proc = get_clip()
    inputs = clip_proc(text=[text], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().cpu().tolist()


def rerank(query: str, documents: list[str]) -> list[float]:
    reranker = get_reranker()
    pairs = [(query, doc) for doc in documents]
    scores = reranker.predict(pairs)
    return scores.tolist()


def warmup_models() -> None:
    logger.info("Warming up local models on device=%s ...", DEVICE)
    if DEVICE == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    get_embedder()
    get_reranker()
    get_clip()
    from config.vision_captioner import warmup_vision_captioner

    warmup_vision_captioner()
    logger.info("All local models loaded and ready.")
