"""Build the LanceDB vector index from the hypertension corpus.

Pipeline: parse -> chunk (512 tokens / 50 overlap) -> embed (BGE-large) -> LanceDB.
Re-running rebuilds the table from scratch, so the index is deterministic.

Usage:
    python -m clinrag.ingest
"""

from __future__ import annotations

import shutil
import time
from collections import Counter

import mlflow
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.lancedb import LanceDBVectorStore

from clinrag.config import PATHS, SETTINGS
from clinrag.embedding import build_embed_model
from clinrag.parsing import load_corpus
from clinrag.tracking import start_run

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def build_index() -> VectorStoreIndex:
    print("Loading corpus...")
    documents = load_corpus()
    if not documents:
        raise SystemExit("No documents parsed. Run scripts/download_corpus.py first.")

    Settings.embed_model = build_embed_model()
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Parsed {len(documents)} sections/pages -> {len(nodes)} chunks")

    # Rebuild the table from scratch so re-runs are deterministic.
    if PATHS.lancedb.exists():
        shutil.rmtree(PATHS.lancedb)
    vector_store = LanceDBVectorStore(
        uri=str(PATHS.lancedb), table_name=SETTINGS.lancedb_table
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"Embedding {len(nodes)} chunks with {SETTINGS.embedding_model}...")
    start = time.time()
    index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
    elapsed = time.time() - start

    per_source = Counter(n.metadata["source"] for n in nodes)
    print(f"\nIndexed {len(nodes)} chunks in {elapsed:.0f}s -> {PATHS.lancedb}")
    for source, count in sorted(per_source.items()):
        print(f"  {source:12s} {count} chunks")

    with start_run("ingest", phase="day3-5"):
        mlflow.log_params(
            {
                "embedding_model": SETTINGS.embedding_model,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
            }
        )
        mlflow.log_metrics(
            {
                "n_sections": len(documents),
                "n_chunks": len(nodes),
                "embed_seconds": elapsed,
            }
        )
    return index


if __name__ == "__main__":
    build_index()
