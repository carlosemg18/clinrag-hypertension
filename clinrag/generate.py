"""The RAG chain: retrieve -> gate for scope -> generate -> validate citations.

Two refusal paths:
  1. Retrieval gate — if the top chunk is below the relevance threshold, the
     question is out of scope and no LLM call is made.
  2. LLM gate — if the retrieved context does not actually answer the question,
     the model sets `answerable=false` and the response is treated as a refusal.

A citation is valid only if its doc_id appears in the retrieved context. If the
first generation cites an unknown doc_id (or cites nothing), one corrective
retry is attempted before the response is returned.
"""

from __future__ import annotations

from clinrag.config import PATHS
from clinrag.llm import generate_answer
from clinrag.schema import RAGResponse
from clinrag.retrieve import retrieve

SYSTEM_PROMPT = (PATHS.root / "clinrag" / "prompts" / "system_v1.md").read_text()


def _format_context(nodes) -> tuple[str, list[str], list[str]]:
    """Render retrieved chunks into a labelled context block for the prompt."""
    blocks, doc_ids, contexts = [], [], []
    for node in nodes:
        meta = node.metadata
        text = node.get_content()
        doc_ids.append(meta["doc_id"])
        contexts.append(text)
        header = (
            f"[doc_id: {meta['doc_id']}] {meta['title']} "
            f"({meta['source']}, {meta.get('location', '?')})"
        )
        blocks.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(blocks), doc_ids, contexts


def _user_message(query: str, context_block: str) -> str:
    return f"CONTEXT EXCERPTS:\n\n{context_block}\n\n---\n\nQUESTION: {query}"


def _invalid_citations(answer, allowed: set[str]) -> list[str]:
    """Cited doc_ids that are not present in the retrieved context."""
    return sorted({c.doc_id for c in answer.citations if c.doc_id not in allowed})


def _missing_inline(answer) -> list[str]:
    """Cited doc_ids that are never tagged inline as [doc_id] in the answer text."""
    return sorted(
        {c.doc_id for c in answer.citations if f"[{c.doc_id}]" not in answer.answer}
    )


def generate(query: str, model: str = "claude", top_k: int | None = None) -> RAGResponse:
    """Answer a hypertension question with enforced inline citations."""
    retrieval = retrieve(query, top_k=top_k)

    # Refusal path 1: nothing relevant retrieved -> out of scope.
    if not retrieval.in_scope:
        return RAGResponse(
            query=query,
            model=model,
            refused=True,
            answer="This question is outside the hypertension guidelines this system covers.",
            refusal_reason="retrieval_below_threshold",
            retrieval_score=retrieval.top_score,
        )

    context_block, doc_ids, contexts = _format_context(retrieval.nodes)
    allowed = set(doc_ids)
    unique_doc_ids = list(dict.fromkeys(doc_ids))
    user_message = _user_message(query, context_block)

    answer = generate_answer(model, SYSTEM_PROMPT, user_message)
    invalid = _invalid_citations(answer, allowed)

    # One corrective retry if the model cited an unknown doc_id, cited nothing,
    # or listed a citation without the matching inline [doc_id] marker.
    if answer.answerable and (invalid or _missing_inline(answer) or not answer.citations):
        correction = (
            "\n\n---\nYour previous response was rejected. Every citation must use "
            f"a doc_id from the context above ({', '.join(unique_doc_ids)}), each "
            "citation object must have a matching inline [doc_id] in the answer "
            "text, and every claim needs an inline [doc_id]. Answer again, correctly."
        )
        answer = generate_answer(model, SYSTEM_PROMPT, user_message + correction)
        invalid = _invalid_citations(answer, allowed)

    # Refusal path 2: the model judged the context insufficient.
    if not answer.answerable:
        return RAGResponse(
            query=query,
            model=model,
            refused=True,
            answer=answer.answer,
            refusal_reason="llm_insufficient_context",
            retrieval_score=retrieval.top_score,
            context_doc_ids=unique_doc_ids,
            contexts=contexts,
        )

    return RAGResponse(
        query=query,
        model=model,
        refused=False,
        answer=answer.answer,
        citations=answer.citations,
        retrieval_score=retrieval.top_score,
        context_doc_ids=unique_doc_ids,
        contexts=contexts,
        invalid_citations=invalid,
    )
