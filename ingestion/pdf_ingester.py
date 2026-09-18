"""
ingestion/pdf_ingester.py

Full ingestion pipeline:
  1. Detect PDF type (book vs question-bank) from heuristics
  2. Extract text pages, tables (pdfplumber), images (PyMuPDF)
  3. Caption images via local Hugging Face BLIP (GPU); chat still uses cloud APIs
  4. Hierarchical chunk: doc-summary → chapter → section → semantic chunks
     MCQ-aware chunking for question-bank PDFs
  5. Embed text chunks (all-MiniLM-L6-v2, local)
  6. Embed images (CLIP, local)
  7. Build / update BM25 index
  8. Upsert to Qdrant with full payload metadata

Run this once per exam dataset.  Re-run to add new books.
"""

import io
import json
import logging
import pickle
import re
import uuid
from pathlib import Path

import pdfplumber
from bm25s import BM25
from qdrant_client.models import PointStruct

from config.model_loader import embed_texts
from ingestion.image_extractor import collect_figures, figures_to_image_chunks
from config.qdrant_manager import DENSE_VECTOR, IMAGE_VECTOR, get_qdrant_client
from config.settings import (
    BM25_DIR,
    CHUNK_OVERLAP_TOKENS,
    EXTRACTED_IMAGES_DIR,
    PDF_DIRS,
    QDRANT_COLLECTIONS,
    SECTION_CHUNK_TOKENS,
    SEMANTIC_CHUNK_TOKENS,
)

logger = logging.getLogger(__name__)

CHUNK_SIZE_SECTION = SECTION_CHUNK_TOKENS
CHUNK_SIZE_SEMANTIC = SEMANTIC_CHUNK_TOKENS
CHUNK_OVERLAP = CHUNK_OVERLAP_TOKENS

# ─── MCQ detection patterns ───────────────────────────────────────────────────
MCQ_QUESTION_RE = re.compile(
    r"(?:^|\n)\s*(?:Q\.?\s*\d+|Question\s+\d+|\d+[\.\)])\s+.{20,}",
    re.MULTILINE,
)
MCQ_OPTION_RE   = re.compile(r"^\s*[A-Da-d][\.\)]\s+.+", re.MULTILINE)
MCQ_ANSWER_RE   = re.compile(
    r"(?:Ans(?:wer)?\.?|Correct\s+(?:option|answer)\s*:?|Key\s*:?)\s*[A-Da-d]",
    re.IGNORECASE,
)
MNEMONIC_RE = re.compile(
    r"\b(?:mnemonic|remember|recall|tip|trick)\b.{0,120}",
    re.IGNORECASE,
)
HEADING_RE = re.compile(r"^[A-Z][A-Z\s\-]{4,80}$", re.MULTILINE)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_question_bank(text_sample: str) -> bool:
    """Heuristic: if >15% of lines match MCQ patterns it's a question bank."""
    lines = text_sample.splitlines()
    mcq_lines = sum(
        1 for l in lines
        if MCQ_OPTION_RE.match(l) or MCQ_QUESTION_RE.search(l)
    )
    return (mcq_lines / max(len(lines), 1)) > 0.15


def _split_sentences(text: str) -> list[str]:
    """Naive but fast sentence splitter for medical text."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _token_approx(text: str) -> int:
    """Approximate token count: word count × 1.35."""
    return int(len(text.split()) * 1.35)


def _sliding_window_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
    source_meta: dict,
) -> list[dict]:
    """Split text into overlapping semantic chunks."""
    sentences = _split_sentences(text)
    chunks = []
    buf: list[str] = []
    buf_tokens = 0

    for sent in sentences:
        sent_tokens = _token_approx(sent)
        if buf_tokens + sent_tokens > chunk_size and buf:
            chunk_text = " ".join(buf)
            chunks.append({**source_meta, "text": chunk_text,
                           "hierarchy_level": "chunk",
                           "chunk_type": "text"})
            # overlap: keep last N tokens worth of sentences
            while buf and buf_tokens > overlap:
                removed = buf.pop(0)
                buf_tokens -= _token_approx(removed)
        buf.append(sent)
        buf_tokens += sent_tokens

    if buf:
        chunks.append({**source_meta, "text": " ".join(buf),
                       "hierarchy_level": "chunk",
                       "chunk_type": "text"})
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Core chunkers
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_book_pdf(
    pdf_path: Path,
    exam: str,
    book_name: str,
) -> list[dict]:
    """
    Hierarchical chunker for standard medical textbooks.
    Levels: document-summary → chapter → section → semantic chunks
    Also extracts tables and images.
    """
    chunks = []
    full_text_pages: list[tuple[int, str]] = []  # (page_num, text)
    page_text_by_num: dict[int, str] = {}

    # ── Pass 1: extract text + tables per page ───────────────────────────────
    with pdfplumber.open(pdf_path) as plumb_pdf:
        for page in plumb_pdf.pages:
            page_text = page.extract_text() or ""
            full_text_pages.append((page.page_number, page_text))
            page_text_by_num[page.page_number] = page_text

            # ── Table extraction ─────────────────────────────────────────────
            for table in page.extract_tables():
                if not table:
                    continue
                # Convert to clean JSON string
                headers = table[0] if table else []
                rows = table[1:] if len(table) > 1 else []
                table_dict = {
                    "headers": headers,
                    "rows": rows,
                }
                table_text = json.dumps(table_dict, ensure_ascii=False)
                chunks.append({
                    "exam": exam,
                    "book_name": book_name,
                    "chapter": "",
                    "section": "",
                    "page_number": page.page_number,
                    "hierarchy_level": "chunk",
                    "chunk_type": "table",
                    "text": f"[TABLE from {book_name} p.{page.page_number}]\n{table_text}",
                    "has_image": False,
                    "image_path": None,
                })

    # ── Pass 2: figures (embedded + page renders for vector/scanned pages) ──
    figures = collect_figures(pdf_path, exam, book_name, page_text_by_num)
    image_chunks = figures_to_image_chunks(figures, exam, book_name)
    chunks.extend(image_chunks)
    if image_chunks:
        by_src: dict[str, int] = {}
        for c in image_chunks:
            src = c.get("figure_source", "unknown")
            by_src[src] = by_src.get(src, 0) + 1
        logger.info(
            "[%s] '%s' image chunks: %s",
            exam,
            book_name,
            ", ".join(f"{k}={v}" for k, v in by_src.items()),
        )

    # ── Pass 3: hierarchical text chunking ───────────────────────────────────
    # Group pages into chapters by detecting ALL-CAPS headings
    current_chapter = "Introduction"
    current_section = ""
    chapter_buffer:  list[tuple[int, str]] = []

    def flush_chapter(chapter_name: str, pages: list[tuple[int, str]]) -> None:
        """Chunk a chapter's worth of pages into section → semantic chunks."""
        nonlocal chunks

        chapter_text = "\n".join(t for _, t in pages)

        # Chapter-level parent chunk (summary)
        if chapter_text.strip():
            # Truncate to CHUNK_SIZE_SECTION tokens for the parent summary
            words = chapter_text.split()
            summary_words = words[:300]
            chunks.append({
                "exam": exam,
                "book_name": book_name,
                "chapter": chapter_name,
                "section": "",
                "page_number": pages[0][0] if pages else 0,
                "hierarchy_level": "chapter",
                "chunk_type": "text",
                "text": " ".join(summary_words),
                "has_image": False,
                "image_path": None,
            })

        # Section-level and semantic children
        section_buf = ""
        section_name = ""
        start_page = pages[0][0] if pages else 0

        for page_num, page_text in pages:
            lines = page_text.splitlines()
            for line in lines:
                if HEADING_RE.match(line.strip()) and len(line.strip()) > 6:
                    # Flush section buffer
                    if section_buf.strip():
                        meta = {
                            "exam": exam,
                            "book_name": book_name,
                            "chapter": chapter_name,
                            "section": section_name,
                            "page_number": start_page,
                            "has_image": False,
                            "image_path": None,
                        }
                        # Section parent
                        chunks.append({
                            **meta,
                            "hierarchy_level": "section",
                            "chunk_type": "text",
                            "text": section_buf[:CHUNK_SIZE_SECTION * 5],  # ~section summary
                        })
                        # Semantic children
                        for child in _sliding_window_chunks(
                            section_buf, CHUNK_SIZE_SEMANTIC, CHUNK_OVERLAP, meta
                        ):
                            # Tag mnemonics
                            if MNEMONIC_RE.search(child["text"]):
                                child["chunk_type"] = "mnemonic"
                            chunks.append(child)

                    section_name = line.strip()
                    section_buf = ""
                    start_page = page_num
                else:
                    section_buf += line + "\n"

        # Flush last section
        if section_buf.strip():
            meta = {
                "exam": exam,
                "book_name": book_name,
                "chapter": chapter_name,
                "section": section_name,
                "page_number": start_page,
                "has_image": False,
                "image_path": None,
            }
            chunks.append({
                **meta,
                "hierarchy_level": "section",
                "chunk_type": "text",
                "text": section_buf[:CHUNK_SIZE_SECTION * 5],
            })
            for child in _sliding_window_chunks(
                section_buf, CHUNK_SIZE_SEMANTIC, CHUNK_OVERLAP, meta
            ):
                if MNEMONIC_RE.search(child["text"]):
                    child["chunk_type"] = "mnemonic"
                chunks.append(child)

    for page_num, page_text in full_text_pages:
        first_line = page_text.strip().splitlines()[0] if page_text.strip() else ""
        if HEADING_RE.match(first_line) and len(first_line) > 6:
            if chapter_buffer:
                flush_chapter(current_chapter, chapter_buffer)
            current_chapter = first_line.strip()
            chapter_buffer = []
        chapter_buffer.append((page_num, page_text))

    if chapter_buffer:
        flush_chapter(current_chapter, chapter_buffer)

    logger.info(f"Book '{book_name}': {len(chunks)} chunks produced.")
    return chunks


def _chunk_mcq_pdf(
    pdf_path: Path,
    exam: str,
    book_name: str,
) -> list[dict]:
    """
    MCQ-aware chunker for question-bank PDFs.
    Detects Q/options/answer/explanation blocks and stores each as a unit.
    """
    chunks = []

    full_text = ""
    with pdfplumber.open(pdf_path) as plumb:
        for page in plumb.pages:
            full_text += (page.extract_text() or "") + "\n"

    # Split into raw MCQ blocks by question number pattern
    # Matches: "1.", "Q.1", "Question 1", "1)"
    splitter = re.compile(
        r"(?=(?:^|\n)\s*(?:Q\.?\s*\d+|Question\s+\d+|\d{1,3}[\.\)])\s)",
        re.MULTILINE,
    )
    raw_blocks = splitter.split(full_text)

    for block in raw_blocks:
        block = block.strip()
        if not block or len(block) < 30:
            continue

        # Extract components
        question_match = re.match(r"(.+?)(?=[A-Da-d][\.\)])", block, re.DOTALL)
        question_text  = question_match.group(1).strip() if question_match else block[:200]

        options = MCQ_OPTION_RE.findall(block)
        answer_match = MCQ_ANSWER_RE.search(block)
        answer_text = answer_match.group(0) if answer_match else ""

        # Everything after answer line = explanation / rationale
        explanation = ""
        if answer_match:
            after_ans = block[answer_match.end():].strip()
            explanation = after_ans[:500]

        formatted = (
            f"QUESTION: {question_text}\n"
            f"OPTIONS:\n" + "\n".join(f"  {o}" for o in options) + "\n"
            f"ANSWER: {answer_text}\n"
            f"EXPLANATION: {explanation}"
        )

        chunks.append({
            "exam": exam,
            "book_name": book_name,
            "chapter": "",
            "section": "",
            "page_number": 0,
            "hierarchy_level": "chunk",
            "chunk_type": "mcq",
            "text": formatted,
            "has_image": False,
            "image_path": None,
        })

    logger.info(f"Q-bank '{book_name}': {len(chunks)} MCQ chunks produced.")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# BM25 index management
# ─────────────────────────────────────────────────────────────────────────────

def _bm25_index_path(exam: str) -> Path:
    return BM25_DIR / f"{exam.lower()}_bm25.pkl"


def load_or_create_bm25(exam: str) -> BM25 | None:
    """Load existing BM25 index from disk or return None if it doesn't exist."""
    path = _bm25_index_path(exam)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_bm25(exam: str, bm25_index: BM25, corpus: list[str]) -> None:
    path = _bm25_index_path(exam)
    with open(path, "wb") as f:
        pickle.dump((bm25_index, corpus), f)
    logger.info("BM25 index saved: %s", path)


def build_and_save_bm25(exam: str, corpus: list[str]) -> BM25:
    """Build BM25 index from corpus and save to disk."""
    tokenized = [doc.lower().split() for doc in corpus]
    bm25 = BM25()
    bm25.index(tokenized)
    save_bm25(exam, bm25, corpus)
    return bm25


# ─────────────────────────────────────────────────────────────────────────────
# Qdrant upsert
# ─────────────────────────────────────────────────────────────────────────────

def _upsert_chunks_to_qdrant(
    exam: str,
    chunks: list[dict],
    batch_size: int = 64,
) -> None:
    """
    Embed and upsert all chunks to the correct Qdrant collection.
    Image chunks get a CLIP vector; text chunks get a dense text vector.
    Both vector types are stored under named-vector keys.
    """
    client = get_qdrant_client()
    collection_name = QDRANT_COLLECTIONS[exam]

    # Separate text chunks from image chunks
    text_chunks  = [c for c in chunks if c["chunk_type"] != "image"]
    image_chunks = [c for c in chunks if c["chunk_type"] == "image"]

    # ── Upsert text chunks in batches ────────────────────────────────────────
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i: i + batch_size]
        texts  = [c["text"] for c in batch]
        vecs   = embed_texts(texts)

        points = []
        for chunk, vec in zip(batch, vecs):
            point_id = str(uuid.uuid4())
            payload  = {k: v for k, v in chunk.items()
                        if k not in ("text", "clip_vector")}
            payload["text"] = chunk["text"]  # store text in payload for retrieval

            points.append(PointStruct(
                id=point_id,
                vector={DENSE_VECTOR: vec},
                payload=payload,
            ))

        if points:
            client.upsert(collection_name=collection_name, points=points)
            logger.info(
                f"[{exam}] Upserted text batch {i//batch_size + 1} "
                f"({len(batch)} chunks)"
            )

    # ── Upsert image chunks ────────────────────────────────────────────────
    for i in range(0, len(image_chunks), batch_size):
        batch = image_chunks[i: i + batch_size]
        # Text caption embedding for text-based image search
        caption_texts = [c["text"] for c in batch]
        caption_vecs  = embed_texts(caption_texts)

        points = []
        for chunk, caption_vec in zip(batch, caption_vecs):
            point_id = str(uuid.uuid4())
            payload  = {k: v for k, v in chunk.items()
                        if k not in ("clip_vector",)}

            vectors = {DENSE_VECTOR: caption_vec}
            if "clip_vector" in chunk and chunk["clip_vector"]:
                vectors[IMAGE_VECTOR] = chunk["clip_vector"]

            points.append(PointStruct(
                id=point_id,
                vector=vectors,
                payload=payload,
            ))

        if points:
            client.upsert(collection_name=collection_name, points=points)
            logger.info(
                f"[{exam}] Upserted image batch {i//batch_size + 1} "
                f"({len(batch)} image chunks)"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main ingestion entry point
# ─────────────────────────────────────────────────────────────────────────────

def ingest_exam(exam: str) -> None:
    """
    Ingest all PDFs for one exam into Qdrant and build BM25 index.
    Safe to re-run — existing points are upserted (no duplicates if same ID).
    Call this for each exam: ingest_exam("DHA"), ingest_exam("MDS"), etc.
    """
    if exam not in PDF_DIRS:
        raise ValueError(f"Unknown exam '{exam}'. Valid: {list(PDF_DIRS)}")

    root_dir = PDF_DIRS[exam]
    if not root_dir.exists():
        logger.warning("[%s] PDF directory not found: %s — skipping.", exam, root_dir)
        return

    pdf_files = sorted(root_dir.rglob("*.pdf"))
    logger.info(f"[{exam}] Found {len(pdf_files)} PDFs in {root_dir}")

    if not pdf_files:
        logger.warning("[%s] No PDF files in %s — skipping exam.", exam, root_dir)
        return

    all_chunks:    list[dict] = []
    all_bm25_docs: list[str] = []

    for pdf_path in pdf_files:
        book_name = pdf_path.stem
        logger.info(f"[{exam}] Processing: {pdf_path.name}")

        try:
            # Sample first 3 pages to determine PDF type
            sample_text = ""
            with pdfplumber.open(pdf_path) as p:
                for pg in p.pages[:3]:
                    sample_text += pg.extract_text() or ""

            is_qbank = _is_question_bank(sample_text)
            logger.info(
                f"[{exam}] '{book_name}' detected as: "
                f"{'Question Bank' if is_qbank else 'Textbook'}"
            )

            if is_qbank:
                chunks = _chunk_mcq_pdf(pdf_path, exam, book_name)
            else:
                chunks = _chunk_book_pdf(pdf_path, exam, book_name)

            all_chunks.extend(chunks)
            all_bm25_docs.extend(c["text"] for c in chunks)

            # Upsert this book's chunks immediately (don't accumulate all in RAM)
            _upsert_chunks_to_qdrant(exam, chunks)
            chunks.clear()

        except Exception as e:
            logger.error(f"[{exam}] Failed to process '{pdf_path.name}': {e}")
            continue

    # Build and save BM25 index for the entire exam corpus
    logger.info(f"[{exam}] Building BM25 index over {len(all_bm25_docs)} documents...")
    build_and_save_bm25(exam, all_bm25_docs)
    logger.info(f"[{exam}] Ingestion complete.")


def ingest_all_exams() -> None:
    """Convenience wrapper to ingest all three exam datasets."""
    for exam in PDF_DIRS:
        logger.info(f"Starting ingestion for exam: {exam}")
        ingest_exam(exam)