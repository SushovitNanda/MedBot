"""
rag/pipeline.py

LangGraph agentic RAG pipeline.
Nodes:
  1. exam_router        — validate and lock exam namespace
  2. query_classifier   — detect query type (concept / MCQ / compare / image)
  3. query_rewriter     — rewrite query if retrieval was poor (conditional)
  4. retriever          — hybrid dense+sparse+image retrieval
  5. doc_grader         — LLM grades relevance of retrieved docs
  6. generator          — streaming LLM answer generation
  7. hallucination_checker — verifies answer is grounded in context
  8. confidence_scorer  — computes final confidence score

Conditional edges implement the self-correction loop:
  doc_grader → irrelevant → query_rewriter → retriever (max 2 retries)
  hallucination_checker → hallucinated → generator (with tighter prompt, once)
"""

import json
import logging
import math
import re
from typing import AsyncGenerator

from langgraph.graph import END, StateGraph

from config.llm import generate_text, stream_text
from config.settings import CONFIDENCE_THRESHOLD, MAX_RETRY_COUNT
from models.schemas import (
    AppMode,
    ConfidenceLevel,
    ExamType,
    MedRAGState,
    QueryType,
)
from retrieval.retriever import RetrievedChunk, retrieve

logger = logging.getLogger(__name__)


def _call_llm(prompt: str) -> str:
    """Synchronous LLM call — Gemini cascade then Groq."""
    return generate_text(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Exam Router
# ─────────────────────────────────────────────────────────────────────────────

def exam_router(state: MedRAGState) -> MedRAGState:
    """
    Validate exam selection and initialise rewritten_query.
    Hard namespace enforcement — no cross-exam retrieval possible.
    """
    if state.exam not in [e for e in ExamType]:
        raise ValueError(f"Invalid exam: {state.exam}")

    # Seed rewritten_query with original query
    state.rewritten_query = state.query
    logger.info(f"[Router] Exam={state.exam} | Query='{state.query[:60]}...'")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — Query Classifier
# ─────────────────────────────────────────────────────────────────────────────

def query_classifier(state: MedRAGState) -> MedRAGState:
    """Classify the query into one of the QueryType categories."""
    prompt = f"""Classify the following medical exam query into EXACTLY ONE category.

Categories:
- concept   : asking about a medical concept, mechanism, definition, pathophysiology
- mcq       : a multiple-choice question or asking to answer an MCQ
- compare   : comparing two or more drugs, diseases, conditions, or procedures
- image     : requesting a diagram, figure, image, or visual explanation
- recall    : asking to list, enumerate, or recall items (e.g. causes of X)
- general   : anything else

Query: "{state.query}"

Reply with ONLY the category name, nothing else."""

    result = _call_llm(prompt).lower().strip()
    # Map to enum
    type_map = {
        "concept": QueryType.CONCEPT,
        "mcq":     QueryType.MCQ,
        "compare": QueryType.COMPARE,
        "image":   QueryType.IMAGE,
        "recall":  QueryType.RECALL,
        "general": QueryType.GENERAL,
    }
    state.query_type = type_map.get(result, QueryType.GENERAL)
    logger.info(f"[Classifier] Query type: {state.query_type}")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — Query Rewriter
# ─────────────────────────────────────────────────────────────────────────────

def query_rewriter(state: MedRAGState) -> MedRAGState:
    """Rewrite the query to improve retrieval quality (triggered on poor retrieval)."""
    state.retry_count += 1
    logger.info(f"[Rewriter] Attempt {state.retry_count} — rewriting query.")

    prompt = f"""You are a medical exam assistant helping improve search queries.
The original query did not retrieve good results. Rewrite it to be more specific
and use precise medical terminology that would appear in medical textbooks.

Exam: {state.exam}
Original query: "{state.query}"
Previous rewrite (if any): "{state.rewritten_query}"

Provide ONLY the rewritten query, no explanation."""

    state.rewritten_query = _call_llm(prompt).strip()
    logger.info(f"[Rewriter] Rewritten: '{state.rewritten_query[:80]}'")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — Retriever
# ─────────────────────────────────────────────────────────────────────────────

def retriever_node(state: MedRAGState) -> MedRAGState:
    """Run hybrid retrieval using the (possibly rewritten) query."""
    include_images = state.query_type == QueryType.IMAGE

    chunks: list[RetrievedChunk] = retrieve(
        query=state.rewritten_query,
        exam=state.exam.value,
        include_images=include_images,
    )

    # Convert to dicts for state serialisation
    state.retrieved_docs = [
        {
            "text":            c.text,
            "exam":            c.exam,
            "book_name":       c.book_name,
            "chapter":         c.chapter,
            "section":         c.section,
            "page_number":     c.page_number,
            "chunk_type":      c.chunk_type,
            "hierarchy_level": c.hierarchy_level,
            "has_image":       c.has_image,
            "image_path":      c.image_path,
            "score":           c.score,
        }
        for c in chunks
    ]

    # Collect image paths
    state.has_images = any(c.has_image for c in chunks)
    state.image_paths = [
        c.image_path for c in chunks
        if c.has_image and c.image_path
    ]

    logger.info(
        f"[Retriever] {len(chunks)} chunks retrieved. "
        f"Images: {len(state.image_paths)}"
    )
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — Document Grader
# ─────────────────────────────────────────────────────────────────────────────

def doc_grader(state: MedRAGState) -> MedRAGState:
    """Grade whether retrieved documents are relevant to the query."""
    if not state.retrieved_docs:
        state.doc_grade = "irrelevant"
        return state

    # Use top 3 chunks for grading to save tokens
    sample_docs = state.retrieved_docs[:3]
    context_sample = "\n\n---\n\n".join(d["text"][:300] for d in sample_docs)

    prompt = f"""You are grading whether retrieved medical textbook excerpts are relevant
to a student's exam question.

Question: "{state.query}"
Exam: {state.exam}

Retrieved excerpts:
{context_sample}

Are these excerpts relevant and useful to answer the question?
Reply with ONLY 'relevant' or 'irrelevant'."""

    result = _call_llm(prompt).lower().strip()
    state.doc_grade = "relevant" if "relevant" in result else "irrelevant"
    logger.info(f"[DocGrader] Grade: {state.doc_grade}")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 6 — Generator (with streaming support)
# ─────────────────────────────────────────────────────────────────────────────

def _build_system_prompt(exam: str, query_type: str) -> str:
    exam_descriptions = {
        "DHA": "Dubai Health Authority (DHA) licensing exam",
        "MDS": "Master of Dental Surgery (MDS) entrance exam",
        "ORE": "Overseas Registration Exam (ORE) for dentists",
    }
    exam_desc = exam_descriptions.get(exam, exam)

    base = f"""You are MedRAG, a specialised AI tutor for the {exam_desc}.

CRITICAL RULES:
1. Answer ONLY using information from the provided context excerpts.
2. Do NOT use general medical knowledge not present in the context.
3. If the context does not contain sufficient information, say so clearly.
4. Always cite your sources (book name, chapter, page) in your answer.
5. For MCQ questions: identify the correct answer and explain why each option is right or wrong.
6. Confidence scoring: assess how well the context supports your answer.
7. Format your response clearly with appropriate headings for complex answers.
"""

    if query_type == "mcq":
        base += "\nFor MCQs: State the answer option first, then provide detailed explanation with rationale."
    elif query_type == "compare":
        base += "\nFor comparisons: Use a structured format with clear differentiating points."
    elif query_type == "recall":
        base += "\nFor recall questions: Use numbered or bulleted lists for clarity."
    elif query_type == "image":
        base += "\nImage context is provided. Describe what the image shows and its clinical significance."

    return base


def _build_context_block(docs: list[dict]) -> str:
    """Format retrieved docs into a context block for the LLM."""
    parts = []
    for i, doc in enumerate(docs, 1):
        src = (
            f"[Source {i}: {doc['book_name']} | "
            f"Chapter: {doc['chapter'] or 'N/A'} | "
            f"Page: {doc['page_number']} | "
            f"Type: {doc['chunk_type']}]"
        )
        parts.append(f"{src}\n{doc['text']}")
    return "\n\n" + ("\n\n---\n\n".join(parts))


def generator_node(state: MedRAGState, strict: bool = False) -> MedRAGState:
    """Generate answer from retrieved context (non-streaming, used in agentic loop)."""
    system_prompt = _build_system_prompt(state.exam.value, state.query_type.value)
    context_block = _build_context_block(state.retrieved_docs)

    strictness_note = (
        "\n\nIMPORTANT: A previous answer was flagged for possible hallucination. "
        "Be extremely conservative. Only state what is explicitly in the context."
    ) if strict else ""

    history_block = ""
    if state.history:
        turns = []
        for turn in state.history[-4:]:  # last 2 exchanges
            turns.append(f"{turn['role'].upper()}: {turn['content']}")
        history_block = "\n\nConversation history:\n" + "\n".join(turns)

    full_prompt = (
        f"{system_prompt}{strictness_note}\n\n"
        f"CONTEXT FROM {state.exam} STUDY MATERIALS:\n{context_block}"
        f"{history_block}\n\n"
        f"STUDENT QUESTION: {state.query}\n\n"
        f"Provide a comprehensive answer. At the end, rate your own confidence "
        f"in this answer as a decimal between 0.0 and 1.0 on a line like:\n"
        f"CONFIDENCE: 0.XX"
    )

    raw_answer = _call_llm(full_prompt)

    # Extract confidence from answer
    conf_match = re.search(r"CONFIDENCE:\s*(0?\.\d+|1\.0)", raw_answer, re.IGNORECASE)
    if conf_match:
        state.confidence = float(conf_match.group(1))
        state.answer = raw_answer[:conf_match.start()].strip()
    else:
        state.confidence = CONFIDENCE_THRESHOLD  # default
        state.answer = raw_answer.strip()

    # Build citation list
    state.citations = [
        {
            "book_name":    doc["book_name"],
            "chapter":      doc["chapter"],
            "section":      doc["section"],
            "page_number":  doc["page_number"],
            "chunk_type":   doc["chunk_type"],
            "has_image":    doc["has_image"],
            "image_path":   doc.get("image_path"),
            "relevance_score": doc["score"],
        }
        for doc in state.retrieved_docs
    ]

    logger.info(f"[Generator] Answer generated. Confidence: {state.confidence:.2f}")
    return state


async def stream_generator(state: MedRAGState) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE events as the LLM generates tokens.
    Call this directly from the FastAPI streaming endpoint.
    """
    system_prompt = _build_system_prompt(state.exam.value, state.query_type.value)
    context_block = _build_context_block(state.retrieved_docs)

    history_block = ""
    if state.history:
        turns = [
            f"{t['role'].upper()}: {t['content']}"
            for t in state.history[-4:]
        ]
        history_block = "\n\nConversation history:\n" + "\n".join(turns)

    full_prompt = (
        f"{system_prompt}\n\n"
        f"CONTEXT FROM {state.exam.value} STUDY MATERIALS:\n{context_block}"
        f"{history_block}\n\n"
        f"STUDENT QUESTION: {state.query}\n\n"
        f"Provide a comprehensive answer with citations."
    )

    for token in stream_text(full_prompt, system_prompt=system_prompt):
        event = json.dumps({"type": "token", "content": token})
        yield f"data: {event}\n\n"

    # ── Emit citations ────────────────────────────────────────────────────────
    citations_event = json.dumps({
        "type":    "citations",
        "content": state.citations,
    })
    yield f"data: {citations_event}\n\n"

    # ── Emit confidence ───────────────────────────────────────────────────────
    confidence_event = json.dumps({
        "type":    "confidence",
        "content": {
            "score": state.confidence,
            "level": _confidence_level(state.confidence).value,
        },
    })
    yield f"data: {confidence_event}\n\n"

    # ── Emit images if any ────────────────────────────────────────────────────
    if state.has_images and state.image_paths:
        images_event = json.dumps({
            "type":    "images",
            "content": state.image_paths,
        })
        yield f"data: {images_event}\n\n"

    # ── Done ──────────────────────────────────────────────────────────────────
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# Node 7 — Hallucination Checker
# ─────────────────────────────────────────────────────────────────────────────

def hallucination_checker(state: MedRAGState) -> MedRAGState:
    """Verify that the generated answer is grounded in the retrieved context."""
    context_texts = " ".join(d["text"][:200] for d in state.retrieved_docs[:3])

    prompt = f"""You are checking if an AI-generated answer is grounded in the source context.

Context excerpts:
{context_texts[:800]}

Generated answer:
{state.answer[:600]}

Does the answer contain ONLY information that is supported by the context?
Reply with ONLY 'grounded' or 'hallucinated'."""

    result = _call_llm(prompt).lower().strip()
    state.hallucination_grade = "grounded" if "grounded" in result else "hallucinated"
    logger.info(f"[HallucinationChecker] Grade: {state.hallucination_grade}")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Confidence Level Helper
# ─────────────────────────────────────────────────────────────────────────────

def _confidence_level(score: float) -> ConfidenceLevel:
    if score >= 0.85:
        return ConfidenceLevel.HIGH
    elif score >= CONFIDENCE_THRESHOLD:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW


# ─────────────────────────────────────────────────────────────────────────────
# Conditional edge functions
# ─────────────────────────────────────────────────────────────────────────────

def should_rewrite(state: MedRAGState) -> str:
    """After doc_grader: rewrite if irrelevant and retries remain."""
    if state.doc_grade == "irrelevant" and state.retry_count < MAX_RETRY_COUNT:
        return "rewrite"
    return "generate"


def should_retry_generation(state: MedRAGState) -> str:
    """After hallucination_checker: retry generation if hallucinated (once)."""
    if state.hallucination_grade == "hallucinated" and state.retry_count < 1:
        state.retry_count += 1
        return "regenerate"
    return "done"


# ─────────────────────────────────────────────────────────────────────────────
# Build the LangGraph
# ─────────────────────────────────────────────────────────────────────────────

def build_rag_graph():
    """
    Compile the LangGraph StateGraph.
    Returns a compiled graph ready for .invoke() or async iteration.
    """
    graph = StateGraph(MedRAGState)

    # Add nodes
    graph.add_node("exam_router",           exam_router)
    graph.add_node("query_classifier",      query_classifier)
    graph.add_node("query_rewriter",        query_rewriter)
    graph.add_node("retriever",             retriever_node)
    graph.add_node("doc_grader",            doc_grader)
    graph.add_node("generator",             generator_node)
    graph.add_node("generator_strict",      lambda s: generator_node(s, strict=True))
    graph.add_node("hallucination_checker", hallucination_checker)

    # Linear flow
    graph.set_entry_point("exam_router")
    graph.add_edge("exam_router",      "query_classifier")
    graph.add_edge("query_classifier", "retriever")
    graph.add_edge("retriever",        "doc_grader")

    # Conditional: grade docs → rewrite or generate
    graph.add_conditional_edges(
        "doc_grader",
        should_rewrite,
        {"rewrite": "query_rewriter", "generate": "generator"},
    )

    # Rewriter loops back to retriever
    graph.add_edge("query_rewriter", "retriever")

    # Generator → hallucination check
    graph.add_edge("generator", "hallucination_checker")

    # Conditional: hallucination → strict regenerate or done
    graph.add_conditional_edges(
        "hallucination_checker",
        should_retry_generation,
        {"regenerate": "generator_strict", "done": END},
    )

    graph.add_edge("generator_strict", END)

    return graph.compile()


# Singleton compiled graph
_compiled_graph = None

def get_rag_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_rag_graph()
    return _compiled_graph


# ─────────────────────────────────────────────────────────────────────────────
# Public API used by FastAPI endpoints
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_state(result) -> MedRAGState:
    if isinstance(result, MedRAGState):
        return result
    if isinstance(result, dict):
        return MedRAGState(**result)
    raise TypeError(f"Unexpected graph result type: {type(result)}")


def _populate_stream_metadata(state: MedRAGState) -> None:
    """Set citations and a retrieval-based confidence before SSE metadata events."""
    state.citations = [
        {
            "book_name": doc["book_name"],
            "chapter": doc["chapter"],
            "section": doc["section"],
            "page_number": doc["page_number"],
            "chunk_type": doc["chunk_type"],
            "has_image": doc["has_image"],
            "image_path": doc.get("image_path"),
            "relevance_score": doc.get("score", 0.0),
        }
        for doc in state.retrieved_docs
    ]
    if state.retrieved_docs:
        top_score = max(doc.get("score", 0.0) for doc in state.retrieved_docs)
        state.confidence = min(1.0, max(0.5, 1 / (1 + math.exp(-top_score))))
    elif state.doc_grade == "relevant":
        state.confidence = CONFIDENCE_THRESHOLD
    else:
        state.confidence = 0.5


async def run_rag_pipeline(
    query: str,
    exam: ExamType,
    mode: AppMode,
    history: list[dict],
    session_id: str = "",
) -> MedRAGState:
    """
    Run the full agentic pipeline synchronously (for eval mode).
    Returns final MedRAGState with answer, citations, confidence, eval_metrics.
    """
    initial_state = MedRAGState(
        query=query,
        exam=exam,
        mode=mode,
        session_id=session_id,
        history=history,
    )
    graph = get_rag_graph()
    result = graph.invoke(initial_state)
    return _coerce_state(result)


async def run_rag_streaming(
    query: str,
    exam: ExamType,
    mode: AppMode,
    history: list[dict],
    session_id: str = "",
) -> AsyncGenerator[str, None]:
    """
    Run retrieval + agentic grading synchronously, then stream generation.
    Yields SSE-formatted strings.
    """
    # Run all nodes except generator synchronously
    state = MedRAGState(
        query=query,
        exam=exam,
        mode=mode,
        session_id=session_id,
        history=history,
    )
    state = exam_router(state)
    state = query_classifier(state)
    state = retriever_node(state)
    state = doc_grader(state)

    while state.doc_grade == "irrelevant" and state.retry_count < MAX_RETRY_COUNT:
        state = query_rewriter(state)
        state = retriever_node(state)
        state = doc_grader(state)

    _populate_stream_metadata(state)

    async for event in stream_generator(state):
        yield event