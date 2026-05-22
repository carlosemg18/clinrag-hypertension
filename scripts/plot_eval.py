"""Render the two-model evaluation comparison chart from the result CSVs.

Reads data/eval/results_<model>.csv and writes docs/eval_comparison.png.

Usage:
    python scripts/plot_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "data" / "eval"
OUT = REPO_ROOT / "docs" / "eval_comparison.png"

# Metric -> (column in CSV, target line). All are "higher is better", 0..1.
METRICS = [
    ("Faithfulness", "ragas_faithfulness", 0.85),
    ("Answer\nRelevancy", "ragas_answer_relevancy", 0.80),
    ("Context\nPrecision", "ragas_llm_context_precision_with_reference", 0.70),
    ("Context\nRecall", "ragas_context_recall", 0.75),
    ("Citation\nAccuracy", None, 0.90),  # computed from good/total
]
MODELS = {"claude": ("Claude Sonnet 4.6", "#c47b3f"), "gemini": ("Gemini 2.5 Pro", "#4a7fb5")}


def metric_value(df: pd.DataFrame, column: str | None) -> float:
    in_scope = df[(df.category == "in-scope") & (~df.refused)]
    if column is None:  # citation accuracy
        total = in_scope.n_citations.sum()
        return in_scope.n_good_citations.sum() / total if total else float("nan")
    return pd.to_numeric(in_scope[column], errors="coerce").mean()


def main() -> int:
    frames = {}
    for model in MODELS:
        path = EVAL_DIR / f"results_{model}.csv"
        if path.exists():
            frames[model] = pd.read_csv(path)
    if not frames:
        print("No result CSVs found. Run: python -m clinrag.evaluate")
        return 1

    labels = [m[0] for m in METRICS]
    x = np.arange(len(METRICS))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, (model, df) in enumerate(frames.items()):
        values = [metric_value(df, col) for _, col, _ in METRICS]
        bars = ax.bar(
            x + (i - 0.5) * width, values, width,
            label=MODELS[model][0], color=MODELS[model][1],
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

    # Target markers.
    for xi, (_, _, target) in zip(x, METRICS):
        ax.hlines(target, xi - 0.5, xi + 0.5, colors="#555", linestyles="dashed", linewidth=1)
    ax.plot([], [], color="#555", linestyle="dashed", label="target")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score (higher is better)")
    ax.set_title("ClinCite-HTN — RAGAS + citation accuracy (40-question golden set)")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
