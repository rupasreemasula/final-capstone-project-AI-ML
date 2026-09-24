"""
Module 3 - Support Assistant: Tasks 3, 4 & 5.
LangGraph StateGraph with a TypedDict state and 3 nodes (classify_intent,
retrieve_and_answer, direct_answer), a conditional routing edge, and a
Pydantic-validated structured output.
"""

from __future__ import annotations

import os
from typing import TypedDict

from langgraph.graph import END, StateGraph

from ingest import retrieve_top_chunks
from prompts import build_direct_prompt, build_rag_prompt
from schemas import AskResponse

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]

DIRECT_ANSWER_FALLBACK = "I can only answer questions about Zepto policies right now."


def mock_llm_enabled() -> bool:
    """MOCK_LLM unset or '1' -> mock (graded baseline). Only '0' -> real LLM."""
    return os.environ.get("MOCK_LLM", "1") != "0"


class GraphState(TypedDict):
    query: str
    intent: str
    retrieved: list[dict]
    answer: str
    sources: list[str]
    confidence: float


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------

def _keyword_classify(query_lower: str) -> str:
    return (
        "policy_question"
        if any(keyword in query_lower for keyword in POLICY_KEYWORDS)
        else "general_question"
    )


def classify_intent(state: GraphState) -> dict:
    query_lower = state["query"].lower()

    if mock_llm_enabled():
        # Mock mode (graded baseline): keyword heuristic, no LLM call.
        intent = _keyword_classify(query_lower)
    else:
        # Optional MOCK_LLM=0 extension: classify via a real LLM call. If the
        # LLM call itself fails (e.g. no GROQ_API_KEY configured, or a
        # network error), fall back to the same keyword heuristic rather
        # than crashing the request.
        from llm import classify_intent_llm

        try:
            intent = classify_intent_llm(state["query"])
        except Exception:
            intent = _keyword_classify(query_lower)

    return {"intent": intent}


def route_after_classify(state: GraphState) -> str:
    """Conditional edge - routing logic itself does not depend on MOCK_LLM."""
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


# ---------------------------------------------------------------------------
# Node 2: retrieve_and_answer
# ---------------------------------------------------------------------------

def retrieve_and_answer(state: GraphState) -> dict:
    # Retrieval always runs for real, in BOTH modes - no API key/network needed.
    retrieved = retrieve_top_chunks(state["query"], n_results=3)
    retrieved_ids = [chunk["id"] for chunk in retrieved]

    if mock_llm_enabled():
        # Mock mode (graded baseline): canned templated answer, no LLM call.
        top_chunk_snippet = retrieved[0]["text"][:200]
        answer = f"Based on the retrieved context: {top_chunk_snippet}"
        return {
            "retrieved": retrieved,
            "answer": answer,
            "sources": retrieved_ids,
            "confidence": 1.0,
        }

    # Optional MOCK_LLM=0 extension: real LLM, grounded only in retrieved chunks.
    from llm import call_llm_and_validate

    context = "\n\n".join(f"[{c['id']}] {c['text']}" for c in retrieved)
    prompt = build_rag_prompt(context=context, query=state["query"])
    validated = call_llm_and_validate(prompt, fallback_sources=retrieved_ids)
    return {
        "retrieved": retrieved,
        "answer": validated.answer,
        "sources": validated.sources,
        "confidence": validated.confidence,
    }


# ---------------------------------------------------------------------------
# Node 3: direct_answer
# ---------------------------------------------------------------------------

def direct_answer(state: GraphState) -> dict:
    if mock_llm_enabled():
        # Mock mode (graded baseline): fixed canned string, no LLM call.
        return {"retrieved": [], "answer": DIRECT_ANSWER_FALLBACK, "sources": [], "confidence": 1.0}

    # Optional MOCK_LLM=0 extension: real LLM, no retrieval.
    from llm import call_llm_and_validate

    prompt = build_direct_prompt(query=state["query"])
    validated = call_llm_and_validate(prompt, fallback_sources=[])
    return {
        "retrieved": [],
        "answer": validated.answer,
        "sources": validated.sources,
        "confidence": validated.confidence,
    }


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_query(query: str) -> AskResponse:
    result = get_compiled_graph().invoke({"query": query})
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
    )
