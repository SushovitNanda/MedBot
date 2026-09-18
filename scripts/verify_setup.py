"""
Smoke-test MedRAG dependencies: Qdrant collections, model cache, PDF folders.

Usage (from repo root):
    python scripts/verify_setup.py
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.qdrant_manager import (
    DENSE_VECTOR,
    IMAGE_VECTOR,
    build_named_vectors_config,
    get_qdrant_client,
    init_collections,
)
from config.settings import PDF_DIRS, compute_all_kb_fingerprints

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    errors: list[str] = []

    # PDF folders
    for exam, path in PDF_DIRS.items():
        pdfs = list(path.rglob("*.pdf")) if path.exists() else []
        logger.info("[%s] %d PDF(s) in %s", exam, len(pdfs), path)
        if not pdfs:
            logger.warning("[%s] No PDFs found — ingestion will skip this exam.", exam)

    # Qdrant
    try:
        cfg = build_named_vectors_config()
        assert DENSE_VECTOR in cfg and IMAGE_VECTOR in cfg
        client = get_qdrant_client()
        client.get_collections()
        init_collections()
        logger.info("Qdrant OK — collections initialized.")
    except Exception as exc:
        errors.append(f"Qdrant: {exc}")

    # GPU / PyTorch
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("CUDA available: %s", torch.cuda.get_device_name(0))
        else:
            logger.warning(
                "CUDA not available — ingest will use CPU. "
                "Run: pip uninstall torch torchvision torchaudio -y && pip install -r requirements.txt"
            )
    except Exception as exc:
        errors.append(f"PyTorch: {exc}")

    # Models (optional quick load)
    try:
        from config.model_loader import get_embedder

        get_embedder()
        logger.info("Embedder model OK.")
    except Exception as exc:
        errors.append(f"Models: {exc}")

    fp = compute_all_kb_fingerprints()
    total = sum(len(v) for v in fp.values())
    logger.info("Knowledge base fingerprint: %d PDF(s) total.", total)

    if errors:
        logger.error("Setup verification failed:")
        for e in errors:
            logger.error("  - %s", e)
        return 1

    logger.info("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
