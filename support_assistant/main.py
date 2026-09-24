"""
Module 3 - Support Assistant: Task 6.
FastAPI wrapper around the LangGraph pipeline. Run locally with:

    uvicorn main:app --host 0.0.0.0 --port 7860

MOCK_LLM defaults to mock mode (unset or "1") - the graded baseline. Set
MOCK_LLM=0 (and GROQ_API_KEY) to exercise the optional real-LLM extension.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from graph import run_query
from ingest import ingest
from schemas import AskRequest, AskResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure the ChromaDB collection is populated before the first request -
    # idempotent, so this is a no-op if ingest.py was already run.
    ingest()
    yield


app = FastAPI(
    title="Zepto Support Assistant",
    description="Grounded GenAI support assistant over Zepto's policy corpus.",
    lifespan=lifespan,
)


@app.get("/")
def health() -> dict:
    return {"status": "ok", "service": "zepto-support-assistant"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return run_query(request.query)
