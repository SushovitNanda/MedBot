"""
ingestion/image_extractor.py

Extract figures from PDFs for RAG indexing:
  - Embedded raster images (PyMuPDF)
  - Full-page renders for scanned / low-text pages
  - Page renders when figures are vector-only (no embedded bitmap)
  - Batch local BLIP captions + CLIP embeddings
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import fitz
from PIL import Image

from config.model_loader import embed_images_batch
from config.settings import (
    EXTRACTED_IMAGES_DIR,
    FIGURE_PAGE_MAX_TEXT_CHARS,
    INGEST_MIN_IMAGE_PIXELS,
    PAGE_RENDER_DPI,
    SCANNED_PAGE_TEXT_THRESHOLD,
)
from config.vision_captioner import caption_images_batch

logger = logging.getLogger(__name__)

FIGURE_REF_RE = re.compile(
    r"\b("
    r"Fig\.|FIG\.|Figure\s+\d|"
    r"Table\s+\d|TABLE\s+\d|"
    r"Flow\s*chart|Chart\s+\d|"
    r"Diagram|Illustration|Algorithm\s+\d"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class ExtractedFigure:
    page_number: int
    pil_image: Image.Image
    save_path: Path
    source: str  # embedded | page_render | scanned_page


def _page_area(page: fitz.Page) -> float:
    r = page.rect
    return max(r.width * r.height, 1.0)


def _image_covers_page(pil_img: Image.Image, page: fitz.Page, threshold: float = 0.82) -> bool:
    page_area = _page_area(page)
    img_area = pil_img.width * pil_img.height
    return img_area / page_area >= threshold


def _render_page(page: fitz.Page, dpi: int) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def _is_scanned_pdf(doc: fitz.Document, sample: int = 12) -> bool:
    n = min(sample, len(doc))
    if n == 0:
        return False
    low = sum(1 for i in range(n) if len(doc[i].get_text().strip()) < SCANNED_PAGE_TEXT_THRESHOLD)
    return low >= n * 0.65


def _should_render_vector_figure_page(page_text: str, embedded_count: int) -> bool:
    text = page_text.strip()
    if embedded_count > 0:
        return False
    if len(text) < SCANNED_PAGE_TEXT_THRESHOLD:
        return True
    if FIGURE_REF_RE.search(text) and len(text) < FIGURE_PAGE_MAX_TEXT_CHARS:
        return True
    return False


def collect_figures(
    pdf_path: Path,
    exam: str,
    book_name: str,
    page_texts: dict[int, str],
) -> list[ExtractedFigure]:
    """
    Collect visual content from all pages without captioning yet.
    """
    doc = fitz.open(str(pdf_path))
    img_dir = EXTRACTED_IMAGES_DIR / exam / book_name.replace(" ", "_")
    img_dir.mkdir(parents=True, exist_ok=True)

    scanned = _is_scanned_pdf(doc)
    if scanned:
        logger.info(
            "[%s] '%s' detected as scanned/image-heavy PDF — page-render mode.",
            exam,
            book_name,
        )

    figures: list[ExtractedFigure] = []
    seen_pages_rendered: set[int] = set()

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        page_text = page_texts.get(page_num, "")

        if scanned:
            if page_num in seen_pages_rendered:
                continue
            pil = _render_page(page, PAGE_RENDER_DPI)
            path = img_dir / f"p{page_num}_scan.png"
            pil.save(path)
            figures.append(
                ExtractedFigure(page_num, pil, path, "scanned_page"),
            )
            seen_pages_rendered.add(page_num)
            continue

        embedded_on_page: list[tuple[Image.Image, int]] = []
        for img_idx, img_info in enumerate(page.get_images(full=True)):
            try:
                base = doc.extract_image(img_info[0])
                pil = Image.open(io.BytesIO(base["image"])).convert("RGB")
                if pil.width * pil.height < INGEST_MIN_IMAGE_PIXELS:
                    continue
                embedded_on_page.append((pil, img_idx))
            except Exception as exc:
                logger.debug("Embedded image skip p%s #%s: %s", page_num, img_idx, exc)

        if len(embedded_on_page) == 1 and _image_covers_page(embedded_on_page[0][0], page):
            pil, img_idx = embedded_on_page[0]
            path = img_dir / f"p{page_num}_full.png"
            pil.save(path)
            figures.append(
                ExtractedFigure(page_num, pil, path, "embedded"),
            )
            continue

        for pil, img_idx in embedded_on_page:
            path = img_dir / f"p{page_num}_img{img_idx}.png"
            pil.save(path)
            figures.append(
                ExtractedFigure(page_num, pil, path, "embedded"),
            )

        if _should_render_vector_figure_page(page_text, len(embedded_on_page)):
            if page_num not in seen_pages_rendered:
                pil = _render_page(page, PAGE_RENDER_DPI)
                path = img_dir / f"p{page_num}_render.png"
                pil.save(path)
                figures.append(
                    ExtractedFigure(page_num, pil, path, "page_render"),
                )
                seen_pages_rendered.add(page_num)

    doc.close()

    logger.info(
        "[%s] '%s': collected %d figures (embedded + renders).",
        exam,
        book_name,
        len(figures),
    )
    return figures


def figures_to_image_chunks(
    figures: list[ExtractedFigure],
    exam: str,
    book_name: str,
) -> list[dict]:
    """Caption and embed collected figures → chunk dicts for Qdrant."""
    if not figures:
        return []

    captions = caption_images_batch([f.pil_image for f in figures])
    clip_vectors = embed_images_batch([f.pil_image for f in figures])

    chunks: list[dict] = []
    for fig, caption, clip_vec in zip(figures, captions, clip_vectors):
        chunks.append({
            "exam": exam,
            "book_name": book_name,
            "chapter": "",
            "section": "",
            "page_number": fig.page_number,
            "hierarchy_level": "chunk",
            "chunk_type": "image",
            "text": caption,
            "has_image": True,
            "image_path": str(fig.save_path),
            "clip_vector": clip_vec,
            "figure_source": fig.source,
        })

    return chunks
