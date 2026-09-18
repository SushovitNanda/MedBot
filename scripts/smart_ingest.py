"""
Smart ingestion: skip re-indexing when DHA / MDS / ORE PDF folders are unchanged.

Usage (from MedRAG repo root):
    python scripts/smart_ingest.py
    python scripts/smart_ingest.py --force
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.qdrant_manager import init_collections
from config.settings import FINGERPRINT_FILE, compute_all_kb_fingerprints
from ingestion.pdf_ingester import ingest_all_exams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_saved_fingerprint() -> dict | None:
    if not FINGERPRINT_FILE.exists():
        return None
    try:
        return json.loads(FINGERPRINT_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_fingerprint(fp: dict) -> None:
    FINGERPRINT_FILE.write_text(json.dumps(fp, indent=2), encoding="utf-8")


def kb_unchanged() -> bool:
    saved = load_saved_fingerprint()
    if saved is None:
        return False
    current = compute_all_kb_fingerprints()
    return saved == current


def main() -> None:
    parser = argparse.ArgumentParser(description="MedRAG smart PDF ingestion")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even when knowledge base fingerprints match",
    )
    args = parser.parse_args()

    try:
        init_collections()
    except Exception as exc:
        logger.error("Qdrant collection setup failed: %s", exc)
        raise SystemExit(1) from exc

    if not args.force and kb_unchanged():
        logger.info(
            "Knowledge base unchanged (DHA Data / MDS Data / ORE Data) — skipping ingestion."
        )
        return

    current = compute_all_kb_fingerprints()
    total_pdfs = sum(len(v) for v in current.values())
    if total_pdfs == 0:
        logger.warning(
            "No PDF files found in DHA Data, MDS Data, or ORE Data. "
            "Add PDFs and re-run, or ingestion will be skipped next time too."
        )
    else:
        logger.info("Starting ingestion for %d PDF(s) across all exams...", total_pdfs)

    try:
        ingest_all_exams()
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        raise SystemExit(1) from exc

    save_fingerprint(current)
    logger.info("Ingestion complete. Fingerprint saved to %s", FINGERPRINT_FILE)


if __name__ == "__main__":
    main()
