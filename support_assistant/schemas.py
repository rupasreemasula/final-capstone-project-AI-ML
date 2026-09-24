"""
Module 3 - Support Assistant: Task 5 (structured output schema) and Task 6
(FastAPI request/response models).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
