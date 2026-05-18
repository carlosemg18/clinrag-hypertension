# ClinCite-HTN — Project Plan

> **A citation-verified clinical Q&A system over hypertension guidelines, with measured faithfulness scores.**

| Meta | |
|---|---|
| **Project #** | 1 of 5 (Healthcare) |
| **Portfolio role target** | Data Scientist — Gen AI / Agentic AI |
| **Target duration** | 14 days (~20–25 hrs) |
| **Repo name (working)** | `clinrag-hypertension` |
| **Domain** | Hypertension |
| **LLMs under test** | Claude Sonnet 4.6 + Gemini 2.5 Pro |

---

## The Thesis

A clinical Q&A system over public hypertension guidelines that **refuses to hallucinate**, **cites every claim inline**, and **publishes measured faithfulness scores** for two leading LLMs.

This is not "another RAG demo." The differentiator is evaluation rigor — measuring hallucination, not hoping it doesn't happen.

---

## Scope

### In scope (MVP must-haves)
- [ ] Curated corpus of ~75–100 hypertension guideline documents
- [ ] Ingestion → chunking → embedding → vector store → retrieval pipeline
- [ ] RAG generation with **enforced inline citations** (no claim ships without `[doc_id]`)
- [ ] Out-of-scope **refusal logic** (refuses non-hypertension questions)
- [ ] **RAGAS evaluation** (faithfulness, answer relevancy, context precision, context recall)
- [ ] **Custom metrics**: citation accuracy, refusal rate, hallucination rate
- [ ] **Head-to-head comparison**: Claude Sonnet 4.6 vs GPT-4o-mini on all metrics
- [ ] MLflow experiment tracking
- [ ] Streamlit UI deployed to HuggingFace Spaces
- [ ] README with architecture diagram + eval results
- [ ] 3-minute Loom demo

### Out of scope (defer)
- ~~Chart abstraction / clinical note extraction~~ → future project
- ~~Fine-tuning~~ → not needed for this thesis
- ~~Multi-step agentic reasoning~~ → save for project 3 (finance)
- ~~Real patient data~~ → never; public guidelines only
- ~~Multi-domain (diabetes, etc.)~~ → keep corpus narrow

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Embeddings | `BAAI/bge-large-en-v1.5` (local) | Strong on medical text, free |
| Vector store | LanceDB | Local, fast, no infra |
| LLM (A) | Claude Sonnet 4.6 via API | Strong citation discipline |
| LLM (B) | Gemini 2.5 Pro via API | Frontier counterpart, generous free tier |
| Framework | LlamaIndex | RAG-focused, less boilerplate than LangChain |
| Evaluation | RAGAS + custom metrics | Industry standard for RAG eval |
| Tracking | MLflow | Local-first, resume-friendly |
| UI | Streamlit | Fastest path to demo |
| Hosting | HuggingFace Spaces | Free, easy, persistent URL |
| CI | GitHub Actions | Lint + eval-on-PR |

**Estimated API cost**: $2–5 total (Gemini 2.5 Pro free tier covers most eval runs; Anthropic API is the main cost).

---

## 14-Day Plan

### Days 1–2 (Weekend 1) — Setup + Corpus
- [x] Initialize repo with `pyproject.toml`, `uv` or `poetry`, `.env.example`
- [x] Set up MLflow locally
- [x] Download corpus (`scripts/download_corpus.py`):
  - [x] ACC/AHA 2017 Hypertension Guidelines
  - [x] USPSTF Hypertension Screening recommendations
  - [x] JNC 8 guidelines
  - [x] NICE NG136 (UK)
  - [x] CDC hypertension resources
  - [~] WHO hypertension treatment guidelines — in manifest; needs manual download (WHO IRIS bot wall, see `docs/corpus.md`)
- [x] Document corpus inventory in `data/corpus_manifest.csv` (doc_id, source, date, URL, license)
- [x] Write `docs/corpus.md` explaining provenance

### Days 3–5 (Weekdays) — Indexing
- [x] PDF → text extraction (per-page; HTML per-section) — `clinrag/parsing.py`
- [x] Chunking strategy (recursive 512 tokens, 50 overlap via `SentenceSplitter`)
- [x] Embed all chunks with BGE-large
- [x] Load into LanceDB with metadata (doc_id, source, location) — 1566 chunks
- [x] Manual retrieval sanity check on 10 questions — 10/10 scope verdicts correct
  - Set provisional relevance threshold to 0.50 (in-scope ~0.62-0.73, OOS ~0.42-0.46)
  - Known issue: JNC 8 PDF has glyph-encoding garbling — revisit in Day 8-10 iteration

### Days 6–7 (Weekend 2) — Generation + Citations
- [x] Write citation-enforcing system prompt — `clinrag/prompts/system_v1.md`
- [x] Pydantic schema for structured response — `clinrag/schema.py` (`LLMAnswer`)
- [x] Wire RAG chain — `clinrag/generate.py` (LlamaIndex retrieval + Claude/Gemini
      generation via native structured output); one corrective retry on bad citations
- [x] Implement out-of-scope refusal — two gates: retrieval-score + LLM `answerable`
- [x] Manual smoke test on 20 questions across Claude Sonnet 4.6 and Gemini 2.5 Pro
  - Claude: 14/14 answered with clean citations, all 6 refusals correct
  - Gemini: 14/14 answered, 6 refusals correct; 1 residual missing-inline-marker
    case after retry — Gemini's citation discipline is weaker (a Day 8-10 finding)
  - Gemini 2.5 Pro needed a raised output-token budget (thinking model)

### Days 8–10 (Weekdays) — **Evaluation** ⭐
- [x] Curate 40-question golden test set — `data/eval/golden_set.jsonl`
  - [x] 25 in-scope (with reference doc_ids + ground-truth answers)
  - [x] 10 out-of-scope (should refuse)
  - [x] 5 adversarial (false-premise questions)
- [x] RAGAS pipeline: faithfulness, answer_relevancy, context_precision, context_recall
- [x] Custom metrics: citation_accuracy, hallucination_rate, refusal_rate, adversarial_pass
- [x] Run full eval on Claude Sonnet 4.6 and Gemini 2.5 Pro, log to MLflow
      (judge = Claude Haiku 4.5, a third model — no self-judging)
- [x] Iterate on weakest metric — PyMuPDF parser (fixed JNC 8 glyph garbling),
      top_k 5→8 (halved false refusals), whitespace-insensitive quote grounding
  - Result: 7/9 metrics clear target; hallucination rate + context precision miss
  - Full methodology, before/after, and failure modes in `docs/evaluation.md`

### Days 11–12 (Weekend 3) — UI + Deploy
- [ ] Streamlit app with citation highlighting on hover
- [ ] Model selector (toggle Claude/GPT)
- [ ] Eval results page (charts from MLflow data)
- [ ] Deploy to HuggingFace Spaces
- [ ] Architecture diagram (Mermaid)

### Days 13–14 (Weekdays) — Polish
- [ ] Final README pass (problem, approach, results, limitations)
- [ ] Eval writeup with charts in `docs/evaluation.md`
- [ ] 3-minute Loom walkthrough
- [ ] Tag v1.0.0 release
- [ ] Add to portfolio website queue

---

## Evaluation Framework

### Golden test set composition (40 questions)
- **25 In-scope factual**: each has a verified answer + supporting doc_ids
- **10 Out-of-scope**: diabetes, oncology, pediatrics, etc. → expect refusal
- **5 Adversarial**: leading questions, false premises, citation traps

### Metrics
| Metric | Target | Why it matters |
|---|---|---|
| Faithfulness (RAGAS) | > 0.85 | Answer grounded in retrieved context |
| Citation accuracy | > 0.90 | Every cited doc_id actually supports the claim |
| Hallucination rate | < 5% | Unsupported claims per response |
| Refusal rate (OOS) | > 95% | System knows when it doesn't know |
| Context precision | > 0.70 | Retrieval brings back relevant chunks |

Honest, measured numbers beat unverified high scores. Report the truth.

---

## Decision Log

| Date | Decision | Rationale |
|---|---|---|
| Day 0 | Hypertension as sole domain | Narrow corpus → better retrieval quality |
| Day 0 | Two-model comparison | Stronger eval story than single-model |
| Day 0 | Gemini 2.5 Pro over GPT-4o-mini | Frontier-vs-frontier comparison; Gemini free tier reduces cost; less common in tutorials → more differentiation |
| Day 0 | LanceDB over Chroma | Lighter footprint, no server process |
| Day 0 | LlamaIndex over LangChain | RAG-focused, less abstraction overhead |

---

## Stretch Goals (only if ahead of schedule)

- Hybrid retrieval (BM25 + dense) comparison in eval
- Query rewriting / decomposition for complex questions
- Confidence calibration plot
- One additional embedding model in the comparison

---

## Open Questions

- [ ] License check on each source document (most are public domain or CC; document anyway)
- [ ] Which Mermaid layout for the architecture diagram?
- [ ] Should the demo include a "show retrieval" debug panel?

---

## Definition of Done

You ship v1.0.0 when:
1. Live HuggingFace Spaces URL works
2. README has architecture diagram + eval results table with real numbers
3. `docs/evaluation.md` explains methodology and failure modes
4. 3-minute Loom recorded
5. Repo is public on GitHub with clean commit history

---

## Next Project (preview)

**Supply Chain — Agentic demand forecasting & procurement copilot** (weeks 3–4 from project 1 start)
