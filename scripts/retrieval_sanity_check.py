"""Manual retrieval sanity check (Days 3-5).

Runs a fixed set of questions through the retriever and prints the top chunks
with their relevance scores. The point is to eyeball, before any generation is
wired up, that:
  - in-scope hypertension questions retrieve relevant chunks at decent scores;
  - out-of-scope questions retrieve weak/irrelevant chunks (low top score).

This is a human-inspection tool, not an automated test. Threshold calibration
and scored metrics come later, in the evaluation phase.

Usage:
    python scripts/retrieval_sanity_check.py
"""

from __future__ import annotations

from clinrag.config import SETTINGS
from clinrag.retrieve import retrieve

# (question, expected_in_scope)
QUESTIONS: list[tuple[str, bool]] = [
    ("What blood pressure reading defines stage 1 hypertension?", True),
    ("How should a diagnosis of hypertension be confirmed before treatment?", True),
    ("What are the first-line drug classes for treating high blood pressure?", True),
    ("Which lifestyle changes are recommended to lower blood pressure?", True),
    ("What blood pressure target is recommended for adults with diabetes?", True),
    ("How often should adults be screened for high blood pressure?", True),
    ("What is ambulatory blood pressure monitoring used for?", True),
    ("What are the risk factors for developing high blood pressure?", True),
    ("What is the recommended insulin regimen for type 1 diabetes?", False),
    ("How is asthma diagnosed in young children?", False),
]

SNIPPET_LEN = 160


def main() -> None:
    print(f"Relevance threshold: {SETTINGS.relevance_threshold} | top_k: {SETTINGS.top_k}\n")
    correct = 0

    for question, expected_in_scope in QUESTIONS:
        result = retrieve(question)
        verdict = "IN-SCOPE" if result.in_scope else "OUT-OF-SCOPE"
        ok = result.in_scope == expected_in_scope
        correct += ok
        flag = "ok" if ok else "MISMATCH"
        expected = "in" if expected_in_scope else "out"

        print(f"Q: {question}")
        print(f"   verdict={verdict}  top_score={result.top_score:.3f}  "
              f"(expected={expected}-scope) [{flag}]")
        for rank, node in enumerate(result.nodes[:3], start=1):
            meta = node.metadata
            snippet = " ".join(node.get_content().split())[:SNIPPET_LEN]
            print(f"   {rank}. [{node.score:.3f}] {meta['doc_id']} "
                  f"({meta.get('location', '?')}) — {snippet}...")
        print()

    print(f"Scope verdict matched expectation on {correct}/{len(QUESTIONS)} questions.")
    print("Inspect the snippets above for relevance — this is a manual check.")


if __name__ == "__main__":
    main()
