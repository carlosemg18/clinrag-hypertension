"""Central configuration: resolves repo paths and loads settings from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the repo root if present. Real environment variables win.
load_dotenv(REPO_ROOT / ".env", override=False)

# Silence the HuggingFace tokenizers fork warning during multi-process embedding.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _path(env_var: str, default: str) -> Path:
    raw = os.getenv(env_var, default)
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


@dataclass(frozen=True)
class Paths:
    root: Path = REPO_ROOT
    corpus: Path = field(default_factory=lambda: REPO_ROOT / "data" / "corpus")
    manifest: Path = field(default_factory=lambda: REPO_ROOT / "data" / "corpus_manifest.csv")
    eval: Path = field(default_factory=lambda: REPO_ROOT / "data" / "eval")
    golden_set: Path = field(default_factory=lambda: REPO_ROOT / "data" / "eval" / "golden_set.jsonl")
    lancedb: Path = field(default_factory=lambda: _path("LANCEDB_URI", "data/lancedb"))


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))

    claude_model: str = field(default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-pro"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    )

    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    relevance_threshold: float = field(
        default_factory=lambda: float(os.getenv("RELEVANCE_THRESHOLD", "0.50"))
    )

    lancedb_table: str = field(default_factory=lambda: os.getenv("LANCEDB_TABLE", "htn_chunks"))

    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    )
    mlflow_experiment: str = field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT", "clinrag-htn")
    )


PATHS = Paths()
SETTINGS = Settings()
