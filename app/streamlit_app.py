"""ClinCite-HTN — Streamlit demo.

Two views:
  - Ask: citation-verified Q&A over the hypertension corpus, with a model
    selector (Claude / Gemini), hover-able inline citations, and a retrieval
    debug panel.
  - Evaluation: the golden-set results for both models, read from the
    per-question CSVs produced by `python -m clinrag.evaluate`.

Run locally:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import csv
import html
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the `clinrag` package importable no matter where the host places this
# file (repo root, an app/ subdir, or a src/ subdir, as on HuggingFace Spaces):
# walk up from here until we find the directory that contains the package.
_HERE = Path(__file__).resolve()
for _candidate in (_HERE.parent, *_HERE.parents):
    if (_candidate / "clinrag" / "__init__.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise ModuleNotFoundError(
        "Could not find the 'clinrag' package near "
        f"{_HERE}. Make sure the clinrag/ folder is uploaded alongside this app."
    )

from clinrag.config import PATHS, SETTINGS  # noqa: E402

st.set_page_config(page_title="ClinCite-HTN", page_icon="🩺", layout="wide")

CITATION_RE = re.compile(r"\[([a-z0-9][a-z0-9\-]+)\]")

EXAMPLE_QUESTIONS = [
    "What blood pressure reading defines stage 1 hypertension?",
    "What are the first-line drug classes for treating high blood pressure?",
    "How often should adults be screened for high blood pressure?",
    "What blood pressure target does NICE recommend for adults under 80?",
    "What is white coat hypertension?",
]


@st.cache_data
def load_manifest() -> dict[str, dict[str, str]]:
    """doc_id -> {title, source, url} for rendering source links."""
    with PATHS.manifest.open(newline="", encoding="utf-8") as fh:
        return {row["doc_id"]: row for row in csv.DictReader(fh)}


def render_answer_html(answer: str, citations) -> str:
    """Escape the answer and turn each inline [doc_id] into a hover badge."""
    tooltips: dict[str, list[str]] = {}
    for c in citations:
        tooltips.setdefault(c.doc_id, []).append(f"{c.location}: “{c.quote}”")

    def badge(match: re.Match) -> str:
        doc_id = match.group(1)
        title = html.escape(" | ".join(tooltips.get(doc_id, ["source"])), quote=True)
        return (
            f'<abbr title="{title}" style="text-decoration:none;cursor:help;">'
            f'<span style="background:#e8f0fe;color:#1a4fbf;border-radius:4px;'
            f'padding:1px 5px;font-size:0.8em;font-weight:600;white-space:nowrap;">'
            f"[{html.escape(doc_id)}]</span></abbr>"
        )

    escaped = html.escape(answer)
    return CITATION_RE.sub(badge, escaped).replace("\n", "<br>")


def ask_view() -> None:
    st.title("🩺 ClinCite-HTN")
    st.caption("Citation-verified Q&A over public hypertension clinical guidelines.")
    st.warning(
        "**Educational project — not medical advice.** Answers come from public "
        "guidelines and must not be used for patient care. Consult a clinician.",
        icon="⚠️",
    )

    with st.sidebar:
        st.header("Settings")
        model = st.radio(
            "Model",
            ["claude", "gemini"],
            format_func=lambda m: {"claude": "Claude Sonnet 4.6", "gemini": "Gemini 2.5 Pro"}[m],
        )
        top_k = st.slider("Retrieved chunks (top-k)", 3, 12, SETTINGS.top_k)
        st.caption(f"Refusal threshold: {SETTINGS.relevance_threshold}")

    st.write("**Try an example:**")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, example in zip(cols, EXAMPLE_QUESTIONS, strict=True):
        if col.button(example, use_container_width=True):
            st.session_state["question"] = example

    question = st.text_input(
        "Ask a hypertension question",
        key="question",
        placeholder="e.g. When should pharmacological treatment be started?",
    )
    ask = st.button("Ask", type="primary")

    if not (ask and question.strip()):
        return

    if not (SETTINGS.anthropic_api_key if model == "claude" else SETTINGS.google_api_key):
        st.error(f"No API key configured for {model}. Set it in your environment / Space secrets.")
        return

    if not PATHS.lancedb.exists():
        st.error("No index found. Build it first: `python -m clinrag.ingest`")
        return

    from clinrag.generate import generate

    with st.spinner(f"Retrieving and generating with {model}… (first call loads the embedder)"):
        resp = generate(question, model=model, top_k=top_k)

    manifest = load_manifest()

    score_col, scope_col = st.columns(2)
    score_col.metric("Top retrieval score", f"{resp.retrieval_score:.3f}")
    scope_col.metric("Verdict", "Refused" if resp.refused else "Answered")

    if resp.refused:
        reason = {
            "retrieval_below_threshold": "Nothing relevant enough was retrieved — out of scope.",
            "llm_insufficient_context": "The retrieved guidelines do not contain the answer.",
        }.get(resp.refusal_reason, resp.refusal_reason or "")
        st.info(f"**The system refused to answer.** {reason}", icon="🚫")
        st.write(resp.answer)
    else:
        st.markdown("### Answer")
        st.markdown(render_answer_html(resp.answer, resp.citations), unsafe_allow_html=True)
        st.caption("Hover a [doc_id] badge to see the supporting quote.")

        if resp.citations:
            st.markdown("### Sources")
            for c in resp.citations:
                meta = manifest.get(c.doc_id, {})
                title = meta.get("title", c.doc_id)
                url = meta.get("url", "")
                head = f"**{title}** — {c.location}"
                head += f"  ·  [source]({url})" if url else ""
                st.markdown(head)
                st.markdown(f"> {c.quote}")

    with st.expander(f"🔎 Retrieval details ({len(resp.contexts)} chunks)"):
        if not resp.contexts:
            st.write("No chunks retrieved above the relevance threshold.")
        for i, (doc_id, text) in enumerate(zip(resp.context_doc_ids, resp.contexts), 1):
            st.markdown(f"**{i}. {doc_id}**")
            st.caption(text[:400] + ("…" if len(text) > 400 else ""))


# --- Evaluation view --------------------------------------------------------

def _aggregate(df: pd.DataFrame) -> dict[str, float]:
    in_scope = df[df.category == "in-scope"]
    answered_in = in_scope[~in_scope.refused]
    oos = df[df.category == "out-of-scope"]
    adv = df[df.category == "adversarial"]
    answered = df[(df.category.isin(["in-scope", "adversarial"])) & (~df.refused)]
    cited = answered_in.n_citations.sum()

    def mean(frame, col):
        return pd.to_numeric(frame[col], errors="coerce").mean()

    adv_pass = (adv.refused | ~adv.premise_endorsed.fillna(False)).mean() if len(adv) else float("nan")
    return {
        "Faithfulness": mean(answered_in, "ragas_faithfulness"),
        "Answer Relevancy": mean(answered_in, "ragas_answer_relevancy"),
        "Context Precision": mean(answered_in, "ragas_llm_context_precision_with_reference"),
        "Context Recall": mean(answered_in, "ragas_context_recall"),
        "Citation Accuracy": answered_in.n_good_citations.sum() / cited if cited else float("nan"),
        "Hallucination Rate": answered.hallucinates.mean() if len(answered) else float("nan"),
        "Refusal Rate (OOS)": oos.refused.mean() if len(oos) else float("nan"),
        "Adversarial Pass Rate": adv_pass,
        "False Refusal Rate": in_scope.refused.mean() if len(in_scope) else float("nan"),
    }


def eval_view() -> None:
    st.title("📊 Evaluation")
    st.caption("40-question golden set: 25 in-scope, 10 out-of-scope, 5 adversarial.")

    frames = {}
    for model in ["claude", "gemini"]:
        path = PATHS.eval / f"results_{model}.csv"
        if path.exists():
            frames[model] = pd.read_csv(path)
    if not frames:
        st.info("No results yet. Run `python -m clinrag.evaluate` to generate them.")
        return

    labels = {"claude": "Claude Sonnet 4.6", "gemini": "Gemini 2.5 Pro"}
    summary = pd.DataFrame({labels[m]: _aggregate(df) for m, df in frames.items()})
    summary["Target"] = [
        "> 0.85", "> 0.80", "> 0.70", "> 0.75", "> 0.90",
        "< 0.05", "> 0.95", "> 0.90", "~ 0.00",
    ]

    st.subheader("Metrics")
    st.dataframe(
        summary.style.format({c: "{:.3f}" for c in summary.columns if c != "Target"}),
        use_container_width=True,
    )

    st.subheader("Model comparison")
    chart_metrics = ["Faithfulness", "Answer Relevancy", "Context Precision",
                     "Context Recall", "Citation Accuracy"]
    st.bar_chart(summary.loc[chart_metrics, [labels[m] for m in frames]])

    st.caption(
        "Judge: Claude Haiku 4.5 (a third model, not under test). "
        "Full methodology and failure modes in docs/evaluation.md."
    )


def main() -> None:
    page = st.sidebar.radio("View", ["Ask", "Evaluation"], label_visibility="collapsed")
    if page == "Ask":
        ask_view()
    else:
        eval_view()


if __name__ == "__main__":
    main()
