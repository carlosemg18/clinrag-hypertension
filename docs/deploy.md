# Deploying to HuggingFace Spaces

The Streamlit app deploys to a free HuggingFace **Space** (Streamlit SDK). The
Space serves the `Ask` and `Evaluation` views. Two things make this deploy a
little different from a vanilla Streamlit app:

1. The **prebuilt LanceDB index** must ship with the Space — rebuilding it on
   startup needs the source PDFs (git-ignored) and is slow.
2. The **BGE-large embedding model** (~1.3 GB) downloads on the first query, so
   the first request on a cold Space is slow. Subsequent queries are fast.

## 1. Create the Space

On https://huggingface.co/new-space: choose **Streamlit** as the SDK, CPU
hardware (free tier is enough), and clone the empty Space repo locally.

## 2. Add the Space metadata

HuggingFace reads a YAML header at the top of the Space's `README.md`. Create it
with:

```yaml
---
title: ClinCite-HTN
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.40.1"
app_file: app/streamlit_app.py
pinned: false
---
```

## 3. Copy the project into the Space

Copy these into the Space repo (everything the app imports at runtime, plus the
index and manifest):

```
app/streamlit_app.py
clinrag/                      # the package (config, retrieve, generate, llm, ...)
data/corpus_manifest.csv
data/lancedb/                 # the prebuilt index (~13 MB) — force-add, it is git-ignored here
requirements.txt
```

`data/lancedb/` is git-ignored in the source repo, so add it explicitly when
committing to the Space:

```bash
git add -f data/lancedb
```

`requirements.txt` (the inference subset — no ragas/mlflow) is what the Space
installs; do **not** copy `pyproject.toml` as the Space build file.

## 4. Set the API keys as Space secrets

In the Space's **Settings → Variables and secrets**, add:

- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

These arrive as environment variables; `clinrag/config.py` reads them directly
(there is no `.env` on the Space). Never commit real keys.

## 5. Push

```bash
git push   # to the Space remote
```

The Space builds, installs `requirements.txt`, and starts the app. Watch the
build logs for the first BGE-large download.

## 6. Update the live links

Once the Space URL is live, replace the `YOUR_USERNAME` placeholders in the main
[README.md](../README.md) badges and the demo link.

## Notes and limits

- **Cold start**: the first query downloads BGE-large and loads it into memory
  (~1.3 GB). Free CPU Spaces sleep after inactivity, so the first query after a
  sleep is slow. This is expected.
- **Cost**: generation calls hit the Anthropic / Google APIs using the Space
  secrets — the keys you provide are billed. The retriever and embeddings run
  locally on the Space (no API cost).
- **Index freshness**: if you re-ingest the corpus, re-copy `data/lancedb/` to
  the Space and push again.
