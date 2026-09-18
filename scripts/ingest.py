"""
Ingest PDFs into Qdrant and build BM25 indexes.

Usage (from MedRAG repo root):
    python scripts/ingest.py --exam DHA
    python scripts/ingest.py --all
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.qdrant_manager import init_collections
from ingestion.pdf_ingester import ingest_all_exams, ingest_exam

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="MedRAG PDF ingestion")
    parser.add_argument(
        "--exam",
        choices=["DHA", "MDS", "ORE"],
        help="Ingest a single exam dataset",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all exam datasets",
    )
    args = parser.parse_args()

    if not args.exam and not args.all:
        parser.error("Provide --exam DHA|MDS|ORE or --all")

    try:
        init_collections()

        if args.all:
            ingest_all_exams()
        else:
            ingest_exam(args.exam)
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
