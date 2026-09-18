"""
retrieval/retriever.py
Hybrid retrieval: dense + BM25 + CLIP image search → RRF → cross-encoder rerank.
"""

import logging
import pickle
from dataclasses import dataclass

import numpy as np
from qdrant_client.models import FieldCondition, Filter, MatchValue

from config.model_loader import embed_single, embed_text_for_image_search, rerank
from config.qdrant_manager import DENSE_VECTOR, IMAGE_VECTOR, get_qdrant_client, get_search_params
from config.settings import (
    BM25_DIR,
    QDRANT_COLLECTIONS,
    TOP_K_DENSE,
    TOP_K_FUSED,
    TOP_K_IMAGE,
    TOP_K_RERANKED,
    TOP_K_SPARSE,
)

logger = logging.getLogger(__name__)

_bm25_cache: dict[str, object] = {}
_bm25_corpus_cache: dict[str, list[str]] = {}


@dataclass
class RetrievedChunk:
    text: str
    exam: str
    book_name: str
    chapter: str
    section: str
    page_number: int
    chunk_type: str
    hierarchy_level: str
    has_image: bool
    image_path: str | None
    score: float
    retrieval_source: str = "fused"


def _load_bm25(exam: str):
    if exam not in _bm25_cache:
        index_path = BM25_DIR / f"{exam.lower()}_bm25.pkl"
        if not index_path.exists():
            logger.warning("No BM25 index for exam '%s'. Run ingestion first.", exam)
            _bm25_cache[exam] = None
            _bm25_corpus_cache[exam] = []
            return None, []

        with open(index_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, tuple):
            _bm25_cache[exam], _bm25_corpus_cache[exam] = data
        else:
            _bm25_cache[exam] = data
            _bm25_corpus_cache[exam] = []

    return _bm25_cache[exam], _bm25_corpus_cache.get(exam, [])


def _bm25_search(exam: str, query: str, top_k: int) -> list[tuple[int, float]]:
    bm25, corpus = _load_bm25(exam)
    if bm25 is None or not corpus:
        return []

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


def _rrf_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    rrf_scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


IMAGE_KEYWORDS = {
    "show", "diagram", "image", "figure", "picture",
    "illustration", "photograph", "chart", "graph",
    "anatomy", "radiograph", "x-ray", "mri", "ct scan",
}


def _is_image_query(query: str) -> bool:
    tokens = set(query.lower().split())
    return bool(tokens & IMAGE_KEYWORDS)


def retrieve(
    query: str,
    exam: str,
    include_images: bool | None = None,
) -> list[RetrievedChunk]:
    if exam not in QDRANT_COLLECTIONS:
        raise ValueError(f"Unknown exam '{exam}'. Valid: {list(QDRANT_COLLECTIONS)}")

    collection = QDRANT_COLLECTIONS[exam]
    client = get_qdrant_client()
    search_params = get_search_params()
    do_image_search = include_images if include_images is not None else _is_image_query(query)

    query_vec = embed_single(query)
    dense_results = client.search(
        collection_name=collection,
        query_vector=(DENSE_VECTOR, query_vec),
        limit=TOP_K_DENSE,
        with_payload=True,
        search_params=search_params,
    )
    dense_list = [(str(r.id), r.score) for r in dense_results]
    payload_store = {str(r.id): r.payload for r in dense_results}

    bm25_hits = _bm25_search(exam, query, TOP_K_SPARSE)
    sparse_list: list[tuple[str, float]] = []

    if bm25_hits:
        sparse_qdrant = client.search(
            collection_name=collection,
            query_vector=(DENSE_VECTOR, query_vec),
            limit=TOP_K_SPARSE,
            with_payload=True,
            search_params=search_params,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="chunk_type",
                        match=MatchValue(value="text"),
                    )
                ]
            ),
        )
        for i, r in enumerate(sparse_qdrant):
            pid = str(r.id)
            bm25_score = bm25_hits[i][1] if i < len(bm25_hits) else 0.1
            sparse_list.append((pid, bm25_score))
            payload_store[pid] = r.payload

    image_list: list[tuple[str, float]] = []
    if do_image_search:
        clip_vec = embed_text_for_image_search(query)
        try:
            image_results = client.search(
                collection_name=collection,
                query_vector=(IMAGE_VECTOR, clip_vec),
                limit=TOP_K_IMAGE,
                with_payload=True,
                search_params=search_params,
                query_filter=Filter(
                    must=[
                        FieldCondition(key="has_image", match=MatchValue(value=True)),
                    ]
                ),
            )
            for r in image_results:
                pid = str(r.id)
                image_list.append((pid, r.score))
                payload_store[pid] = r.payload
        except Exception as exc:
            logger.warning("Image search failed: %s", exc)

    all_lists = [lst for lst in [dense_list, sparse_list, image_list] if lst]
    fused = _rrf_fusion(all_lists)[:TOP_K_FUSED]

    missing_ids = [pid for pid, _ in fused if pid not in payload_store]
    if missing_ids:
        fetched = client.retrieve(
            collection_name=collection,
            ids=missing_ids,
            with_payload=True,
        )
        for point in fetched:
            payload_store[str(point.id)] = point.payload

    fused_chunks = []
    for pid, rrf_score in fused:
        payload = payload_store.get(pid, {})
        if payload:
            fused_chunks.append((pid, rrf_score, payload))

    if not fused_chunks:
        logger.warning("No chunks retrieved for query: '%s' [%s]", query[:80], exam)
        return []

    doc_texts = [p["text"] for _, _, p in fused_chunks]
    rerank_scores = rerank(query, doc_texts)
    scored = sorted(
        zip(fused_chunks, rerank_scores),
        key=lambda x: x[1],
        reverse=True,
    )[:TOP_K_RERANKED]

    results: list[RetrievedChunk] = []
    for (_, _, payload), rerank_score in scored:
        results.append(
            RetrievedChunk(
                text=payload.get("text", ""),
                exam=payload.get("exam", exam),
                book_name=payload.get("book_name", ""),
                chapter=payload.get("chapter", ""),
                section=payload.get("section", ""),
                page_number=payload.get("page_number", 0),
                chunk_type=payload.get("chunk_type", "text"),
                hierarchy_level=payload.get("hierarchy_level", "chunk"),
                has_image=payload.get("has_image", False),
                image_path=payload.get("image_path"),
                score=float(rerank_score),
            )
        )

    logger.info("[%s] Retrieved %d chunks for: '%s...'", exam, len(results), query[:60])
    return results
