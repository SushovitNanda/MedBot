"""
config/settings.py
All configuration loaded from MedRAG/.env (single source of truth).
"""

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
_ENV_PATH = ROOT_DIR / ".env"
if not _ENV_PATH.is_file():
    raise EnvironmentError(f"Missing {_ENV_PATH} — copy keys from project docs or create .env")
load_dotenv(_ENV_PATH, override=True)


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_bool(key: str, default: str = "false") -> bool:
    return _env(key, default).lower() in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    raw = _env(key, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise EnvironmentError(f".env {key} must be an integer, got: {raw!r}") from exc


def _env_float(key: str, default: float) -> float:
    raw = _env(key, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise EnvironmentError(f".env {key} must be a number, got: {raw!r}") from exc


# ── API keys (required) ──────────────────────────────────────────────────────
GOOGLE_API_KEY: str = _env("GOOGLE_API_KEY")
GROQ_API_KEY: str = _env("GROQ_API_KEY")
API_SECRET_TOKEN: str = _env("API_SECRET_TOKEN")
HF_TOKEN: str = _env("HF_TOKEN") or _env("HUGGINGFACE_HUB_TOKEN")

if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY missing from .env")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY missing from .env")
if not API_SECRET_TOKEN:
    raise EnvironmentError("API_SECRET_TOKEN missing from .env")

# Propagate HF token for transformers / huggingface_hub
if HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", HF_TOKEN)

# Hugging Face cache warnings (Windows symlinks)
if _env("HF_HUB_DISABLE_SYMLINKS_WARNING"):
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = _env("HF_HUB_DISABLE_SYMLINKS_WARNING")

# ── API server ───────────────────────────────────────────────────────────────
API_HOST: str = _env("API_HOST", "0.0.0.0")
API_PORT: int = _env_int("API_PORT", 8000)

# ── Frontend (Next.js reads NEXT_PUBLIC_* from frontend/.env.local at build/dev)
NEXT_PUBLIC_API_URL: str = _env("NEXT_PUBLIC_API_URL", "http://localhost:8000")
NEXT_PUBLIC_API_TOKEN: str = _env("NEXT_PUBLIC_API_TOKEN") or API_SECRET_TOKEN

# ── LLM models (chat / RAG — cloud APIs only) ───────────────────────────────
PRIMARY_LLM: str = _env("PRIMARY_LLM", "gemini-3.1-flash-lite-preview")
SECONDARY_LLM: str = _env("SECONDARY_LLM", "gemini-2.0-flash")
FALLBACK_LLM: str = _env("FALLBACK_LLM", "llama-3.3-70b-versatile")
VISION_LLM: str = _env("VISION_LLM") or PRIMARY_LLM

# ── Local models (ingest + retrieval) ────────────────────────────────────────
LOCAL_MODEL_DEVICE: str = _env("LOCAL_MODEL_DEVICE", "auto")
EMBEDDING_MODEL: str = _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM: int = _env_int("EMBEDDING_DIM", 384)
RERANKER_MODEL: str = _env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CLIP_MODEL: str = _env("CLIP_MODEL", "openai/clip-vit-base-patch32")
CLIP_DIM: int = _env_int("CLIP_DIM", 512)

USE_LOCAL_VISION_CAPTION: bool = _env_bool("USE_LOCAL_VISION_CAPTION", "true")
VISION_CAPTION_MODEL: str = _env(
    "VISION_CAPTION_MODEL", "Salesforce/blip-image-captioning-large"
)
VISION_CAPTION_BATCH_SIZE: int = _env_int("VISION_CAPTION_BATCH_SIZE", 8)

# PDF figure extraction
INGEST_MIN_IMAGE_PIXELS: int = _env_int("INGEST_MIN_IMAGE_PIXELS", 3600)
PAGE_RENDER_DPI: int = _env_int("PAGE_RENDER_DPI", 150)
FIGURE_PAGE_MAX_TEXT_CHARS: int = _env_int("FIGURE_PAGE_MAX_TEXT_CHARS", 400)
SCANNED_PAGE_TEXT_THRESHOLD: int = _env_int("SCANNED_PAGE_TEXT_THRESHOLD", 120)

MODELS_CACHE_DIR: Path = ROOT_DIR / _env("MODELS_CACHE_DIR", "models_cache")
MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── PDF knowledge base ───────────────────────────────────────────────────────
PDF_DIRS: dict[str, Path] = {
    "DHA": ROOT_DIR / _env("DHA_DATA_DIR", "DHA Data"),
    "MDS": ROOT_DIR / _env("MDS_DATA_DIR", "MDS Data"),
    "ORE": ROOT_DIR / _env("ORE_DATA_DIR", "ORE Data"),
}

# ── Runtime artifacts ────────────────────────────────────────────────────────
EXTRACTED_IMAGES_DIR: Path = ROOT_DIR / _env("EXTRACTED_IMAGES_DIR", "extracted_images")
BM25_DIR: Path = ROOT_DIR / _env("BM25_DIR", "bm25_indexes")
CHUNK_CACHE_DIR: Path = ROOT_DIR / _env("CHUNK_CACHE_DIR", "chunk_cache")
FINGERPRINT_FILE: Path = ROOT_DIR / _env("FINGERPRINT_FILE", "kb_fingerprint.json")
QDRANT_STORAGE_DIR: Path = ROOT_DIR / _env("QDRANT_STORAGE_DIR", "qdrant_data")

for _d in (EXTRACTED_IMAGES_DIR, BM25_DIR, CHUNK_CACHE_DIR, QDRANT_STORAGE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Qdrant ───────────────────────────────────────────────────────────────────
QDRANT_MODE: str = _env("QDRANT_MODE", "auto")
QDRANT_HOST: str = _env("QDRANT_HOST", "localhost")
QDRANT_PORT: int = _env_int("QDRANT_PORT", 6333)
QDRANT_LOCAL_PATH: Path = QDRANT_STORAGE_DIR / "local_embedded"
QDRANT_COLLECTIONS: dict[str, str] = {
    "DHA": _env("QDRANT_COLLECTION_DHA", "dha"),
    "MDS": _env("QDRANT_COLLECTION_MDS", "mds"),
    "ORE": _env("QDRANT_COLLECTION_ORE", "ore"),
}

# ── Chunking ─────────────────────────────────────────────────────────────────
SECTION_CHUNK_TOKENS: int = _env_int("SECTION_CHUNK_TOKENS", 800)
SEMANTIC_CHUNK_TOKENS: int = _env_int("SEMANTIC_CHUNK_TOKENS", 400)
CHUNK_OVERLAP_TOKENS: int = _env_int("CHUNK_OVERLAP_TOKENS", 50)

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K_DENSE: int = _env_int("TOP_K_DENSE", 15)
TOP_K_SPARSE: int = _env_int("TOP_K_SPARSE", 15)
TOP_K_IMAGE: int = _env_int("TOP_K_IMAGE", 5)
TOP_K_FUSED: int = _env_int("TOP_K_FUSED", 20)
TOP_K_RERANKED: int = _env_int("TOP_K_RERANKED", 5)

# ── Agentic loop ─────────────────────────────────────────────────────────────
MAX_RETRY_COUNT: int = _env_int("MAX_RETRY_COUNT", 2)
CONFIDENCE_THRESHOLD: float = _env_float("CONFIDENCE_THRESHOLD", 0.70)

# ── Langfuse (optional) ──────────────────────────────────────────────────────
LANGFUSE_PUBLIC_KEY: str = _env("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY: str = _env("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST: str = _env("LANGFUSE_HOST", "http://localhost:3001")
LANGFUSE_ENABLED: bool = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)


def resolve_local_device() -> str:
    import torch

    pref = LOCAL_MODEL_DEVICE.lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise EnvironmentError(
                "LOCAL_MODEL_DEVICE=cuda but PyTorch has no CUDA. "
                "Reinstall: pip uninstall torch torchvision torchaudio -y && pip install -r requirements.txt"
            )
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _file_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_kb_fingerprint(exam: str) -> dict[str, str]:
    pdf_dir = PDF_DIRS.get(exam)
    if not pdf_dir or not pdf_dir.exists():
        return {}
    return {
        p.name: _file_fingerprint(p)
        for p in sorted(pdf_dir.rglob("*.pdf"))
    }


def compute_all_kb_fingerprints() -> dict[str, dict[str, str]]:
    return {exam: compute_kb_fingerprint(exam) for exam in PDF_DIRS}
