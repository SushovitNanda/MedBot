"""
config/qdrant_manager.py
Qdrant connection and collection setup (server or embedded local — no API key).
"""

import logging
from functools import lru_cache
from typing import Dict

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    BinaryQuantization,
    BinaryQuantizationConfig,
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PayloadSchemaType,
    QuantizationSearchParams,
    SearchParams,
    VectorParams,
)

from config.settings import (
    CLIP_DIM,
    EMBEDDING_DIM,
    QDRANT_COLLECTIONS,
    QDRANT_HOST,
    QDRANT_LOCAL_PATH,
    QDRANT_MODE,
    QDRANT_PORT,
)

logger = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
IMAGE_VECTOR = "image"

_client_mode: str | None = None


def _server_reachable() -> bool:
    try:
        r = httpx.get(
            f"http://{QDRANT_HOST}:{QDRANT_PORT}/healthz",
            timeout=2.0,
        )
        return r.status_code == 200
    except Exception:
        return False


def resolve_qdrant_mode() -> str:
    """Return 'server' or 'local' based on QDRANT_MODE and connectivity."""
    global _client_mode
    if _client_mode:
        return _client_mode

    if QDRANT_MODE == "local":
        _client_mode = "local"
    elif QDRANT_MODE == "server":
        _client_mode = "server"
    else:
        _client_mode = "server" if _server_reachable() else "local"

    logger.info("Qdrant mode resolved: %s", _client_mode)
    return _client_mode


def build_named_vectors_config() -> Dict[str, VectorParams]:
    """
    Multi-vector collection config for qdrant-client 1.12+.

    VectorsConfig is a typing.Union alias — pass a plain dict to create_collection(),
    not VectorsConfig(...).
    """
    return {
        DENSE_VECTOR: VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
            on_disk=True,
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100, on_disk=False),
        ),
        IMAGE_VECTOR: VectorParams(
            size=CLIP_DIM,
            distance=Distance.COSINE,
            on_disk=True,
        ),
    }


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    mode = resolve_qdrant_mode()
    if mode == "local":
        QDRANT_LOCAL_PATH.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(QDRANT_LOCAL_PATH))
        logger.info("Qdrant embedded client at %s", QDRANT_LOCAL_PATH)
    else:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
        logger.info("Qdrant server client at %s:%s", QDRANT_HOST, QDRANT_PORT)
    return client


def _collection_exists(client: QdrantClient, name: str) -> bool:
    existing = [c.name for c in client.get_collections().collections]
    return name in existing


def init_collections() -> None:
    """Create exam collections (dha, mds, ore) if they do not exist."""
    client = get_qdrant_client()
    vectors_config = build_named_vectors_config()

    for exam, collection_name in QDRANT_COLLECTIONS.items():
        if _collection_exists(client, collection_name):
            logger.info("Collection '%s' already exists — skipping.", collection_name)
            continue

        logger.info("Creating collection '%s' for exam '%s'...", collection_name, exam)

        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                quantization_config=BinaryQuantization(
                    binary=BinaryQuantizationConfig(always_ram=True),
                ),
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=20_000,
                    memmap_threshold=50_000,
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create Qdrant collection '{collection_name}' for exam '{exam}'. "
                f"Ensure Qdrant is running (Docker: docker compose up qdrant -d) or set "
                f"QDRANT_MODE=local. Original error: {exc}"
            ) from exc

        for field, schema in [
            ("exam", PayloadSchemaType.KEYWORD),
            ("chunk_type", PayloadSchemaType.KEYWORD),
            ("book_name", PayloadSchemaType.KEYWORD),
            ("chapter", PayloadSchemaType.KEYWORD),
            ("page_number", PayloadSchemaType.INTEGER),
            ("has_image", PayloadSchemaType.BOOL),
            ("hierarchy_level", PayloadSchemaType.KEYWORD),
        ]:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception as exc:
                logger.warning(
                    "Payload index '%s' on '%s' failed (non-fatal): %s",
                    field,
                    collection_name,
                    exc,
                )

        logger.info("Collection '%s' created successfully.", collection_name)


def get_search_params() -> SearchParams:
    return SearchParams(
        quantization=QuantizationSearchParams(
            ignore=False,
            rescore=True,
            oversampling=2.0,
        )
    )
