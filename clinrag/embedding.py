"""The shared BGE embedding model, used by both ingestion and retrieval.

Kept in its own module so retrieval (and the Streamlit app) do not transitively
import the heavier ingestion/evaluation dependencies (mlflow, ragas).
"""

from __future__ import annotations

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from clinrag.config import SETTINGS

# BGE-v1.5 retrieval works best when queries carry this instruction prefix.
# Passages (the corpus chunks) are embedded without it.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"


def build_embed_model() -> HuggingFaceEmbedding:
    """The BGE-large embedding model, shared by ingestion and retrieval."""
    return HuggingFaceEmbedding(
        model_name=SETTINGS.embedding_model,
        query_instruction=BGE_QUERY_INSTRUCTION,
    )
