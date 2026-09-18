"""
models/schemas.py
Pydantic v2 schemas for API I/O and LangGraph state.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExamType(str, Enum):
    DHA = "DHA"
    MDS = "MDS"
    ORE = "ORE"


class AppMode(str, Enum):
    ANSWER = "answer"
    EVAL = "eval"


class QueryType(str, Enum):
    CONCEPT = "concept"
    MCQ = "mcq"
    COMPARE = "compare"
    IMAGE = "image"
    RECALL = "recall"
    GENERAL = "general"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    exam: ExamType
    mode: AppMode = AppMode.ANSWER
    session_id: str = ""
    history: list[dict[str, str]] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def clean_query(cls, v: str) -> str:
        return v.strip()


class Citation(BaseModel):
    book_name: str
    chapter: str
    section: str
    page_number: int
    chunk_type: str
    has_image: bool
    image_path: str | None = None
    relevance_score: float


class EvalMetrics(BaseModel):
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    hallucination_flag: bool = False


class ChatResponse(BaseModel):
    answer: str
    exam: ExamType
    mode: AppMode
    query_type: QueryType
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    citations: list[Citation]
    has_images: bool
    image_paths: list[str]
    eval_metrics: EvalMetrics | None = None
    retry_count: int = 0
    session_id: str = ""


class StreamChunk(BaseModel):
    type: str
    content: Any = None


class MedRAGState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str = ""
    exam: ExamType = ExamType.DHA
    mode: AppMode = AppMode.ANSWER
    session_id: str = ""
    history: list[dict[str, str]] = Field(default_factory=list)

    query_type: QueryType = QueryType.GENERAL
    rewritten_query: str = ""

    retrieved_docs: list[dict] = Field(default_factory=list)
    doc_grade: str = "pending"
    retry_count: int = 0

    answer: str = ""
    confidence: float = 0.0
    citations: list[dict] = Field(default_factory=list)
    has_images: bool = False
    image_paths: list[str] = Field(default_factory=list)

    hallucination_grade: str = "pending"
    eval_metrics: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    exam: ExamType
    admin_token: str


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    models_loaded: bool
    exams_available: list[str]
