"""
Pre-download and cache embedder, reranker, and CLIP on CPU.

Usage (from MedRAG repo root):
    python scripts/warmup_models.py
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.model_loader import warmup_models
from config.settings import resolve_local_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

if __name__ == "__main__":
    warmup_models()
    print(
        f"All models cached under models_cache/ — ready on {resolve_local_device()}."
    )
