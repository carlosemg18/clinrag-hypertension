"""Evaluation harness for the 40-question golden set (Days 8-10).

Runs the full RAG chain for a model over every golden-set question, then scores
the results two ways:
  - RAGAS (LLM-graded): faithfulness, answer relevancy, context precision/recall
  - Custom (deterministic + small judge): citation accuracy, hallucination rate,
    out-of-scope refusal rate, adversarial false-premise handling

Results are logged to MLflow and written to data/eval/results_<model>.csv.

Usage:
    python -m clinrag.evaluate                      # both models
    python -m clinrag.evaluate --model claude       # one model
    python -m clinrag.evaluate --model gemini --limit 6 --no-ragas   # quick test
"""

from __future__ import annotations

import argparse
import json
import math

import mlflow
import pandas as pd

from clinrag.config import PATHS, SETTINGS
from clinrag.generate import SYSTEM_PROMPT
from clinrag.ingest import CHUNK_OVERLAP, CHUNK_SIZE
from clinrag.generate import generate
from clinrag.metrics import citation_quality, judge_premise_endorsed, response_hallucinates
from clinrag.schema import RAGResponse
from clinrag.tracking import start_run

PROMPT_VERSION = "system_v1"


def load_golden_set() -> list[dict]:
    with PATHS.golden_set.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --- generation -------------------------------------------------------------

def run_generation(golden: list[dict], model: str) -> list[tuple[dict, RAGResponse]]:
    results: list[tuple[dict, RAGResponse]] = []
    for i, item in enumerate(golden, start=1):
        resp = generate(item["question"], model=model)
        verdict = "refused" if resp.refused else f"{len(resp.citations)} citations"
        print(f"  [{model}] {i:2d}/{len(golden)} {item['id']:8s} {verdict}")
        results.append((item, resp))
    return results


# --- custom metrics ---------------------------------------------------------

def compute_custom_metrics(
    results: list[tuple[dict, RAGResponse]],
) -> tuple[dict[str, float], dict[str, bool]]:
    by_cat = {"in-scope": [], "out-of-scope": [], "adversarial": []}
    for item, resp in results:
        by_cat[item["category"]].append((item, resp))

    # Citation accuracy over answered in-scope responses.
    good = total = 0
    for _, resp in by_cat["in-scope"]:
        if not resp.refused:
            g, t = citation_quality(resp)
            good += g
            total += t
    citation_accuracy = good / total if total else 0.0

    oos = by_cat["out-of-scope"]
    refusal_rate_oos = sum(r.refused for _, r in oos) / len(oos) if oos else 0.0

    in_scope = by_cat["in-scope"]
    false_refusal_rate = (
        sum(r.refused for _, r in in_scope) / len(in_scope) if in_scope else 0.0
    )

    # Adversarial false-premise handling (LLM-judged for answered responses).
    premise_endorsed: dict[str, bool] = {}
    adv = by_cat["adversarial"]
    adv_passed = 0
    for item, resp in adv:
        if resp.refused:
            premise_endorsed[item["id"]] = False
            adv_passed += 1
        else:
            endorsed = judge_premise_endorsed(
                item["question"], item["false_premise"], resp.answer
            )
            premise_endorsed[item["id"]] = endorsed
            adv_passed += not endorsed
    adversarial_pass_rate = adv_passed / len(adv) if adv else 0.0

    # Hallucination rate over answered in-scope + adversarial responses.
    answered = [(it, r) for it, r in in_scope + adv if not r.refused]
    hallucinated = sum(
        response_hallucinates(r, premise_endorsed.get(it["id"], False))
        for it, r in answered
    )
    hallucination_rate = hallucinated / len(answered) if answered else 0.0

    metrics = {
        "citation_accuracy": citation_accuracy,
        "hallucination_rate": hallucination_rate,
        "refusal_rate_oos": refusal_rate_oos,
        "false_refusal_rate": false_refusal_rate,
        "adversarial_pass_rate": adversarial_pass_rate,
    }
    return metrics, premise_endorsed


# --- RAGAS ------------------------------------------------------------------

def _ragas_judge():
    """A fixed judge (Claude) and a light embedding model for RAGAS scoring."""
    from langchain_anthropic import ChatAnthropic
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = LangchainLLMWrapper(
        ChatAnthropic(model=SETTINGS.judge_model, temperature=0, max_tokens=4096)
    )
    emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )
    return llm, emb


def run_ragas(results: list[tuple[dict, RAGResponse]]) -> tuple[dict[str, float], pd.DataFrame]:
    """Score answered in-scope responses with the four RAGAS metrics."""
    from ragas import EvaluationDataset, RunConfig
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    samples = [
        {
            "user_input": item["question"],
            "retrieved_contexts": resp.contexts,
            "response": resp.answer,
            "reference": item["ground_truth"],
        }
        for item, resp in results
        if item["category"] == "in-scope" and not resp.refused
    ]
    if not samples:
        return {}, pd.DataFrame()

    llm, emb = _ragas_judge()
    result = ragas_evaluate(
        dataset=EvaluationDataset.from_list(samples),
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=llm,
        embeddings=emb,
        run_config=RunConfig(max_workers=3, timeout=300, max_retries=15),
        show_progress=True,
    )
    df = result.to_pandas()

    inputs = {"user_input", "retrieved_contexts", "response", "reference"}
    scores: dict[str, float] = {}
    for col in df.columns:
        if col in inputs:
            continue
        mean = pd.to_numeric(df[col], errors="coerce").mean()
        if "faith" in col:
            scores["faithfulness"] = mean
        elif "relevan" in col:
            scores["answer_relevancy"] = mean
        elif "precision" in col:
            scores["context_precision"] = mean
        elif "recall" in col:
            scores["context_recall"] = mean
    return scores, df


# --- per-question results ---------------------------------------------------

def build_results_table(
    results: list[tuple[dict, RAGResponse]],
    premise_endorsed: dict[str, bool],
    ragas_df: pd.DataFrame,
    model: str,
) -> pd.DataFrame:
    ragas_by_question = {}
    if not ragas_df.empty:
        ragas_by_question = {
            row["user_input"]: row for _, row in ragas_df.iterrows()
        }

    rows = []
    for item, resp in results:
        good, total = citation_quality(resp)
        endorsed = premise_endorsed.get(item["id"], False)
        row = {
            "id": item["id"],
            "category": item["category"],
            "model": model,
            "question": item["question"],
            "refused": resp.refused,
            "refusal_reason": resp.refusal_reason or "",
            "retrieval_score": round(resp.retrieval_score, 3),
            "n_citations": total,
            "n_good_citations": good,
            "premise_endorsed": endorsed,
            "hallucinates": response_hallucinates(resp, endorsed),
            "answer": resp.answer,
        }
        ragas_row = ragas_by_question.get(item["question"])
        if ragas_row is not None:
            for col, val in ragas_row.items():
                if col not in ("user_input", "retrieved_contexts", "response", "reference"):
                    row[f"ragas_{col}"] = val
        rows.append(row)
    return pd.DataFrame(rows)


# --- orchestration ----------------------------------------------------------

def evaluate_model(model: str, golden: list[dict], use_ragas: bool) -> dict[str, float]:
    print(f"\n=== Evaluating {model} on {len(golden)} questions ===")
    results = run_generation(golden, model)

    print(f"  [{model}] scoring custom metrics...")
    custom, premise_endorsed = compute_custom_metrics(results)

    ragas_scores: dict[str, float] = {}
    ragas_df = pd.DataFrame()
    if use_ragas:
        print(f"  [{model}] running RAGAS...")
        ragas_scores, ragas_df = run_ragas(results)

    table = build_results_table(results, premise_endorsed, ragas_df, model)
    out_csv = PATHS.eval / f"results_{model}.csv"
    table.to_csv(out_csv, index=False)
    print(f"  [{model}] per-question results -> {out_csv}")

    all_metrics = {**ragas_scores, **custom}
    answered = sum(not r.refused for _, r in results)

    with start_run(f"eval-{model}", phase="day8-10", model=model):
        mlflow.log_params(
            {
                "model": model,
                "judge_model": SETTINGS.judge_model,
                "prompt_version": PROMPT_VERSION,
                "embedding_model": SETTINGS.embedding_model,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "top_k": SETTINGS.top_k,
                "relevance_threshold": SETTINGS.relevance_threshold,
                "n_questions": len(golden),
            }
        )
        mlflow.log_metric("n_answered", answered)
        for name, value in all_metrics.items():
            if value is not None and not math.isnan(value):
                mlflow.log_metric(name, value)
        mlflow.log_artifact(str(out_csv))

    return all_metrics


def _print_comparison(scores: dict[str, dict[str, float]]) -> None:
    targets = {
        "faithfulness": "> 0.85",
        "answer_relevancy": "> 0.80",
        "context_precision": "> 0.70",
        "context_recall": "> 0.75",
        "citation_accuracy": "> 0.90",
        "hallucination_rate": "< 0.05",
        "refusal_rate_oos": "> 0.95",
        "false_refusal_rate": "~ 0.00",
        "adversarial_pass_rate": "> 0.90",
    }
    models = list(scores)
    print("\n" + "=" * 72)
    print("EVALUATION SUMMARY")
    header = f"{'metric':24s} " + "".join(f"{m:>14s}" for m in models) + f"{'target':>12s}"
    print(header)
    print("-" * len(header))
    for metric, target in targets.items():
        cells = ""
        for m in models:
            val = scores[m].get(metric)
            cells += f"{val:>14.3f}" if isinstance(val, float) and not math.isnan(val) else f"{'-':>14s}"
        print(f"{metric:24s} {cells}{target:>12s}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["claude", "gemini"], help="evaluate one model")
    parser.add_argument("--limit", type=int, help="evaluate only the first N questions")
    parser.add_argument("--no-ragas", action="store_true", help="skip RAGAS (custom metrics only)")
    args = parser.parse_args()

    golden = load_golden_set()
    if args.limit:
        golden = golden[: args.limit]
    models = [args.model] if args.model else ["claude", "gemini"]

    scores = {m: evaluate_model(m, golden, use_ragas=not args.no_ragas) for m in models}
    _print_comparison(scores)
    print(f"\nLogged to MLflow ({SETTINGS.mlflow_tracking_uri}). View: mlflow ui")


if __name__ == "__main__":
    main()
