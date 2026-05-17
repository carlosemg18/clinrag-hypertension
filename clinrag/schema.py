"""Structured schemas for citation-enforced RAG responses.

`LLMAnswer` is the contract the LLM must satisfy — both Claude and Gemini are
constrained to return exactly this shape. `RAGResponse` is the full pipeline
result (retrieval + generation + validation) returned to callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A single source reference backing a claim in the answer."""

    doc_id: str = Field(description="doc_id of a document from the provided context")
    location: str = Field(description="The location label of the chunk, e.g. 'p. 73'")
    quote: str = Field(description="Verbatim sentence(s) from the context that support the claim")


class LLMAnswer(BaseModel):
    """The exact JSON contract the LLM must return.

    Shared, model-agnostic schema so Claude and Gemini are graded on equal terms.
    """

    answerable: bool = Field(
        description="True only if the provided context actually contains the answer"
    )
    answer: str = Field(
        description=(
            "The answer, with an inline [doc_id] citation after every claim. "
            "If answerable is false, briefly explain that the guidelines provided "
            "do not cover the question."
        )
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="One entry per [doc_id] cited inline; empty if answerable is false",
    )


@dataclass
class RAGResponse:
    """Full pipeline result for one query against one model."""

    query: str
    model: str
    refused: bool
    answer: str
    citations: list[Citation] = field(default_factory=list)
    refusal_reason: str | None = None
    retrieval_score: float = 0.0
    context_doc_ids: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)

    @property
    def citation_doc_ids(self) -> list[str]:
        return [c.doc_id for c in self.citations]
