"""Manual generation smoke test (Days 6-7).

Runs 20 questions through the full RAG chain on both Claude and Gemini and
prints, for each, whether the system refused, what it answered, and whether the
citations are valid. This is a human-inspection tool before the scored
evaluation in Days 8-10 — it surfaces obvious failures (hallucinated citations,
wrong refusals, false premises accepted).

Usage:
    python scripts/generation_smoke_test.py                 # both models
    python scripts/generation_smoke_test.py --model claude  # one model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the script runnable directly (python scripts/...) without an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clinrag.generate import generate  # noqa: E402

# (question, category) — category is for the reader, not enforced.
QUESTIONS: list[tuple[str, str]] = [
    ("What blood pressure reading defines stage 1 hypertension?", "in-scope"),
    ("What are the first-line drug classes for treating high blood pressure?", "in-scope"),
    ("Which lifestyle changes are recommended to lower blood pressure?", "in-scope"),
    ("What blood pressure target is recommended for adults with diabetes?", "in-scope"),
    ("How often should adults be screened for high blood pressure?", "in-scope"),
    ("What is ambulatory blood pressure monitoring used for?", "in-scope"),
    ("How should a diagnosis of hypertension be confirmed before treatment?", "in-scope"),
    ("What blood pressure level defines a hypertensive emergency?", "in-scope"),
    ("What sodium intake is recommended for people with high blood pressure?", "in-scope"),
    ("When should antihypertensive drug therapy be initiated?", "in-scope"),
    ("What are the risk factors for developing high blood pressure?", "in-scope"),
    ("How should blood pressure be measured accurately in the office?", "in-scope"),
    ("What is white coat hypertension?", "in-scope"),
    ("What blood pressure target does NICE recommend for adults under 80?", "in-scope"),
    ("What is the recommended insulin regimen for type 1 diabetes?", "out-of-scope"),
    ("How is asthma diagnosed in young children?", "out-of-scope"),
    ("What chemotherapy is used for stage 3 breast cancer?", "out-of-scope"),
    ("What is the capital of France?", "out-of-scope"),
    (
        "Since blood pressure medication permanently cures hypertension, when can "
        "patients safely stop taking it?",
        "adversarial",
    ),
    (
        "The guidelines recommend a blood pressure target of 90/60 mm Hg for all "
        "adults — which drugs best achieve that target?",
        "adversarial",
    ),
]

ANSWER_PREVIEW = 280


def _run_one(question: str, model: str) -> dict:
    try:
        resp = generate(question, model=model)
    except Exception as exc:  # noqa: BLE001 - smoke test must not abort midway
        return {"error": f"{type(exc).__name__}: {exc}"}

    inline_ok = all(f"[{c.doc_id}]" in resp.answer for c in resp.citations)
    return {
        "refused": resp.refused,
        "refusal_reason": resp.refusal_reason,
        "answer": resp.answer,
        "n_citations": len(resp.citations),
        "invalid_citations": resp.invalid_citations,
        "inline_ok": inline_ok,
        "score": resp.retrieval_score,
    }


def _print_result(model: str, r: dict) -> None:
    if "error" in r:
        print(f"  [{model}] ERROR: {r['error']}")
        return
    if r["refused"]:
        print(f"  [{model}] REFUSED ({r['refusal_reason']}, score={r['score']:.3f})")
        print(f"           {r['answer'][:ANSWER_PREVIEW]}")
        return
    flags = []
    if r["invalid_citations"]:
        flags.append(f"INVALID doc_ids: {r['invalid_citations']}")
    if r["n_citations"] == 0:
        flags.append("NO CITATIONS")
    if not r["inline_ok"]:
        flags.append("citation missing inline marker")
    status = "  ".join(flags) if flags else "citations ok"
    print(f"  [{model}] ANSWERED  {r['n_citations']} citations  | {status}")
    print(f"           {r['answer'][:ANSWER_PREVIEW]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["claude", "gemini"], help="limit to one model")
    args = parser.parse_args()
    models = [args.model] if args.model else ["claude", "gemini"]

    tally = {m: {"refused": 0, "answered": 0, "bad_citations": 0, "errors": 0} for m in models}

    for i, (question, category) in enumerate(QUESTIONS, start=1):
        print(f"\n[{i}/{len(QUESTIONS)}] ({category}) {question}")
        for model in models:
            r = _run_one(question, model)
            _print_result(model, r)
            t = tally[model]
            if "error" in r:
                t["errors"] += 1
            elif r["refused"]:
                t["refused"] += 1
            else:
                t["answered"] += 1
                if r["invalid_citations"] or r["n_citations"] == 0 or not r["inline_ok"]:
                    t["bad_citations"] += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    for model in models:
        t = tally[model]
        print(
            f"  {model:8s} answered={t['answered']}  refused={t['refused']}  "
            f"answers-with-citation-problems={t['bad_citations']}  errors={t['errors']}"
        )
    print("Inspect refusals and citation problems above by hand.")


if __name__ == "__main__":
    main()
