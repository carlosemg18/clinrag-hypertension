"""Retrieval over the LanceDB index, with an out-of-scope relevance gate.

The index must be built first (`python -m clinrag.ingest`).
"""

from __future__ import annotations

from dataclasses import dataclass

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.lancedb import LanceDBVectorStore

from clinrag.config import PATHS, SETTINGS
from clinrag.ingest import build_embed_model

_index: VectorStoreIndex | None = None


def _get_index() -> VectorStoreIndex:
    """Lazily connect to the LanceDB table; embedding model loaded once."""
    global _index
    if _index is None:
        if not PATHS.lancedb.exists():
            raise SystemExit(
                f"No index at {PATHS.lancedb}. Run: python -m clinrag.ingest"
            )
        Settings.embed_model = build_embed_model()
        vector_store = LanceDBVectorStore(
            uri=str(PATHS.lancedb), table_name=SETTINGS.lancedb_table
        )
        _index = VectorStoreIndex.from_vector_store(vector_store)
    return _index


@dataclass
class RetrievalResult:
    """Retrieved chunks for a query, plus the in-/out-of-scope verdict."""

    query: str
    nodes: list[NodeWithScore]

    @property
    def top_score(self) -> float:
        return self.nodes[0].score or 0.0 if self.nodes else 0.0

    @property
    def in_scope(self) -> bool:
        """A query is in scope when its best chunk clears the relevance gate.

        The threshold (RELEVANCE_THRESHOLD) is provisional — it is calibrated
        against the out-of-scope golden-set subset during evaluation (Days 8-10).
        """
        return self.top_score >= SETTINGS.relevance_threshold


def retrieve(query: str, top_k: int | None = None) -> RetrievalResult:
    """Return the top-k corpus chunks most relevant to `query`."""
    k = top_k or SETTINGS.top_k
    retriever = _get_index().as_retriever(similarity_top_k=k)
    return RetrievalResult(query=query, nodes=retriever.retrieve(query))
