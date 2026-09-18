"""
evaluation/rag_evaluator.py

RAGAS-based evaluation suite for eval mode.
Computes: faithfulness, answer_relevancy, context_precision, context_recall.
Also runs a custom hallucination flag via Gemini.

Only triggered when mode == "eval". Has zero overhead in "answer" mode.
"""

import logging
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from config.settings import GOOGLE_API_KEY, PRIMARY_LLM

logger = logging.getLogger(__name__)

genai.configure(api_key=GOOGLE_API_KEY)


def _get_ragas_llm():
    """LangChain-wrapped Gemini for RAGAS."""
    return ChatGoogleGenerativeAI(
        model=PRIMARY_LLM,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )


def _get_ragas_embeddings():
    """LangChain-wrapped Gemini embeddings for RAGAS."""
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY,
    )


def run_ragas_evaluation(
    query: str,
    answer: str,
    retrieved_docs: list[dict],
    ground_truth: str = "",
) -> dict[str, float | None]:
    """
    Run RAGAS evaluation on a single query-answer-context triple.

    Args:
        query:          The user's question
        answer:         The generated answer
        retrieved_docs: List of retrieved chunk dicts (with 'text' key)
        ground_truth:   Optional reference answer (improves context_recall score)

    Returns:
        Dict with metric names as keys and float scores (0–1) as values.
        None values indicate metric could not be computed.
    """
    contexts = [doc["text"] for doc in retrieved_docs if doc.get("text")]

    if not contexts:
        logger.warning("No contexts available for RAGAS evaluation.")
        return {
            "faithfulness":      None,
            "answer_relevancy":  None,
            "context_precision": None,
            "context_recall":    None,
        }

    # Build RAGAS dataset
    data = {
        "question":  [query],
        "answer":    [answer],
        "contexts":  [contexts],
    }
    if ground_truth:
        data["ground_truth"] = [ground_truth]

    dataset = Dataset.from_dict(data)

    # Select metrics
    metrics = [faithfulness, answer_relevancy, context_precision]
    if ground_truth:
        metrics.append(context_recall)

    try:
        llm        = _get_ragas_llm()
        embeddings = _get_ragas_embeddings()

        results = evaluate(
            dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
        )

        scores = results.to_pandas().iloc[0].to_dict()
        return {
            "faithfulness":      _safe_float(scores.get("faithfulness")),
            "answer_relevancy":  _safe_float(scores.get("answer_relevancy")),
            "context_precision": _safe_float(scores.get("context_precision")),
            "context_recall":    _safe_float(scores.get("context_recall")),
        }

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return {
            "faithfulness":      None,
            "answer_relevancy":  None,
            "context_precision": None,
            "context_recall":    None,
        }


def check_hallucination_quick(
    answer: str,
    contexts: list[str],
) -> bool:
    """
    Quick hallucination check using Gemini.
    Returns True if hallucination is suspected.
    Faster and cheaper than full RAGAS faithfulness.
    """
    if not contexts:
        return False

    context_sample = "\n---\n".join(c[:300] for c in contexts[:3])
    model  = genai.GenerativeModel(PRIMARY_LLM)

    prompt = f"""Does the following answer contain claims NOT supported by the context?

Context:
{context_sample}

Answer:
{answer[:500]}

Reply with ONLY 'yes' (hallucination detected) or 'no' (answer is grounded)."""

    try:
        response = model.generate_content(prompt)
        return "yes" in response.text.lower()
    except Exception as e:
        logger.warning(f"Hallucination check failed: {e}")
        return False


def compute_confidence(
    rerank_scores: list[float],
    doc_grade: str,
    hallucination_grade: str,
    ragas_faithfulness: float | None,
) -> float:
    """
    Composite confidence score combining multiple signals.

    Signals:
    - Reranker score of top retrieved chunk (retrieval quality)
    - Doc grader result
    - Hallucination checker result
    - RAGAS faithfulness (if available)

    Returns float in [0, 1].
    """
    score = 0.5  # base

    # Signal 1: top reranker score (0–1 normalised via sigmoid)
    if rerank_scores:
        top_score = max(rerank_scores)
        # Cross-encoder scores can be negative; sigmoid to [0,1]
        import math
        normalised = 1 / (1 + math.exp(-top_score))
        score += 0.2 * normalised

    # Signal 2: doc grader
    if doc_grade == "relevant":
        score += 0.15
    else:
        score -= 0.15

    # Signal 3: hallucination check
    if hallucination_grade == "grounded":
        score += 0.10
    else:
        score -= 0.20

    # Signal 4: RAGAS faithfulness (most reliable signal)
    if ragas_faithfulness is not None:
        score += 0.05 + 0.20 * ragas_faithfulness

    return max(0.0, min(1.0, score))


def _safe_float(val: Any) -> float | None:
    """Convert to float or return None."""
    try:
        return float(val) if val is not None else None
    except (ValueError, TypeError):
        return None