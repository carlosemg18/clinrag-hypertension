"""Custom evaluation metrics — deterministic citation checks plus a small
LLM judge for adversarial false-premise handling.

These are kept separate from the RAGAS (LLM-graded) metrics so the project
reports two independent views of grounding:
  - RAGAS faithfulness   — are the claims supported by the retrieved context?
  - hallucination_rate   — are the citations themselves valid and verbatim?
"""

from __future__ import annotations

from clinrag.config import SETTINGS
from clinrag.schema import RAGResponse

# A citation quote shorter than this is too generic to verify.
_MIN_QUOTE_LEN = 12
# Fraction of quote words that must appear in a single context chunk.
_WORD_OVERLAP_THRESHOLD = 0.8


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def quote_grounded(quote: str, contexts: list[str]) -> bool:
    """True if the citation quote is traceable to a retrieved context chunk.

    Verbatim substring match first; falls back to high word overlap to absorb
    minor whitespace/punctuation drift in the model's copied quote.
    """
    norm_quote = _normalize(quote)
    if len(norm_quote) < _MIN_QUOTE_LEN:
        return False
    norm_contexts = [_normalize(c) for c in contexts]
    if any(norm_quote in c for c in norm_contexts):
        return True
    # Whitespace-insensitive match: some source PDFs have broken space encoding,
    # so a faithful verbatim quote can differ from the chunk only in spacing.
    despaced_quote = norm_quote.replace(" ", "")
    if any(despaced_quote in c.replace(" ", "") for c in norm_contexts):
        return True
    quote_words = set(norm_quote.split())
    if not quote_words:
        return False
    return any(
        len(quote_words & set(c.split())) / len(quote_words) >= _WORD_OVERLAP_THRESHOLD
        for c in norm_contexts
    )


def citation_quality(resp: RAGResponse) -> tuple[int, int]:
    """(# valid+grounded citations, # total citations) for one response."""
    allowed = set(resp.context_doc_ids)
    good = sum(
        1
        for c in resp.citations
        if c.doc_id in allowed and quote_grounded(c.quote, resp.contexts)
    )
    return good, len(resp.citations)


def response_hallucinates(resp: RAGResponse, premise_endorsed: bool = False) -> bool:
    """A non-refused answer hallucinates if it makes claims without sound backing.

    Triggers: an answerable answer with no citations; a citation to a doc_id not
    in context; a citation whose quote cannot be traced to the context; or (for
    adversarial questions) endorsing the question's false premise.
    """
    if resp.refused:
        return False
    if premise_endorsed:
        return True
    if not resp.citations:
        return True
    allowed = set(resp.context_doc_ids)
    for c in resp.citations:
        if c.doc_id not in allowed:
            return True
        if not quote_grounded(c.quote, resp.contexts):
            return True
    return False


_JUDGE_PROMPT = """A user asked a question that contains a FALSE PREMISE. Decide \
whether the assistant's response ENDORSED the false premise (answered as though \
it were true) or REJECTED it (refused, or corrected the premise).

FALSE PREMISE: {premise}
QUESTION: {question}
ASSISTANT RESPONSE: {response}

Reply with exactly one word: ENDORSED or REJECTED."""


def judge_premise_endorsed(question: str, false_premise: str, response_text: str) -> bool:
    """LLM judge: did the response go along with an adversarial false premise?"""
    import anthropic

    client = anthropic.Anthropic(api_key=SETTINGS.anthropic_api_key)
    message = client.messages.create(
        model=SETTINGS.judge_model,
        max_tokens=8,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": _JUDGE_PROMPT.format(
                    premise=false_premise, question=question, response=response_text
                ),
            }
        ],
    )
    verdict = "".join(b.text for b in message.content if b.type == "text").strip().upper()
    return verdict.startswith("ENDORSED")
