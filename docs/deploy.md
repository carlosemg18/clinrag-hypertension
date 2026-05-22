# Deploying to HuggingFace Spaces

This is a step-by-step guide to putting the Streamlit app online for free. No
prior HuggingFace experience assumed.

## How this works (the mental model)

A **HuggingFace Space is its own git repository** hosted on huggingface.co.
Whatever you commit to that repo, HuggingFace automatically builds and runs.
For a Streamlit Space, HuggingFace:

1. reads a YAML header at the top of the Space's `README.md` to learn it's a
   Streamlit app and which file to run;
2. installs the Python packages listed in `requirements.txt`;
3. runs your app and serves it at `https://huggingface.co/spaces/<you>/<name>`.

So deploying = "create a second git repo on HuggingFace, copy the app + the
prebuilt search index + the dependency list into it, set the API keys as
secrets, and push." That's the whole thing. The sections below are just that,
in detail.

Two project-specific wrinkles:

- The **prebuilt LanceDB index** (`data/lancedb/`, ~13 MB) must be shipped with
  the Space. Rebuilding it on the server would need the source PDFs (which are
  git-ignored) and is slow, so we ship the finished index instead.
- The **BGE-large embedding model** (~1.3 GB) downloads automatically the first
  time someone asks a question. The first request on a cold Space is therefore
  slow (a minute or two); everything after that is fast.

---

## Prerequisites

- A free HuggingFace account: https://huggingface.co/join
- This repo cloned locally with a **working built index** — i.e. you've run
  `python -m clinrag.ingest` and `data/lancedb/` exists. Check:
  ```bash
  ls data/lancedb        # should list a .lance table directory
  ```
- Your `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY` (the ones in your local `.env`).

---

## Step 1 — Create a HuggingFace access token

HuggingFace no longer accepts your account password for git pushes; you push
with an **access token** instead (think of it as an app-specific password).

1. Go to https://huggingface.co/settings/tokens
2. Click **New token**.
3. Name it e.g. `clinrag-deploy`, set the role to **Write**, and create it.
4. Copy the token (starts with `hf_...`) and keep it somewhere safe — you'll
   paste it in Step 5. Treat it like a password; never commit it.

---

## Step 2 — Create the Space

1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Owner**: your username (`cemg19`)
   - **Space name**: `clinrag-htn` (this becomes part of the URL)
   - **License**: MIT
   - **SDK**: select **Streamlit**
   - **Hardware**: **CPU basic** (free) — enough for this app
   - **Visibility**: Public
3. Click **Create Space**. You now have an empty repo at
   `https://huggingface.co/spaces/cemg19/clinrag-htn`.

---

## Step 3 — Clone the (empty) Space repo locally

Clone it **next to** this project, in a separate folder — the Space is a
different repo from your GitHub repo.

```bash
cd ..                                  # leave the clinrag-hypertension folder
git clone https://huggingface.co/spaces/cemg19/clinrag-htn
cd clinrag-htn
```

It will contain just a starter `README.md` and `.gitignore`.

---

## Step 4 — Copy the app and the index into the Space

From inside the `clinrag-htn` folder, copy these from your project. The runtime
needs the app, the `clinrag` package, the index, the manifest, and the
dependency list — **but not** the source PDFs (only ingestion needs those).

```bash
SRC=../clinrag-hypertension

cp -r  "$SRC/app"                     .
cp -r  "$SRC/clinrag"                 .       # includes clinrag/prompts/system_v1.md
cp     "$SRC/requirements.txt"        .
mkdir -p data/eval
cp     "$SRC/data/corpus_manifest.csv" data/
cp     "$SRC/data/eval/"results_*.csv  data/eval/   # for the Evaluation tab
cp -r  "$SRC/data/lancedb"            data/         # the prebuilt index
```

> The index is git-ignored in the source repo. Here it lives in a fresh repo
> with no such ignore rule, so a normal `git add` will pick it up. (If your
> source `.gitignore` got copied over, force it: `git add -f data/lancedb`.)

Your Space folder should now look like:

```
clinrag-htn/
├── README.md                 # the Space config (edit in Step 5)
├── requirements.txt
├── app/streamlit_app.py
├── clinrag/                   # config, retrieve, generate, llm, prompts/, ...
└── data/
    ├── corpus_manifest.csv
    ├── lancedb/               # ~13 MB prebuilt index
    └── eval/results_*.csv
```

> **Golden rule:** `clinrag/`, `data/`, and `requirements.txt` must sit in the
> **same folder as the app file**. The app auto-finds the `clinrag` package by
> searching upward from itself, so a few layouts work — but if you started from
> a template that puts the app in a `src/` folder, put `clinrag/` and `data/`
> inside that `src/` folder too. A missing `clinrag/` is the #1 deploy error.

---

## Step 5 — Configure the Space README

HuggingFace reads a YAML header at the very top of the Space's `README.md` to
configure the app. **`app_file` is the important line** — our entry point is
`app/streamlit_app.py`, not the default `app.py`, so it must be set explicitly.

Open the Space's `README.md` and make the top look exactly like this:

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
license: mit
---
```

(You can write anything you like below the closing `---`; that's just the
Space's description page.)

---

## Step 6 — Set the API keys as secrets

Do **not** put your keys in any file. Add them through the website:

1. Open your Space → **Settings** tab.
2. Scroll to **Variables and secrets** → **New secret**.
3. Add two secrets:
   - Name `ANTHROPIC_API_KEY`, value = your Anthropic key
   - Name `GOOGLE_API_KEY`, value = your Google key

HuggingFace injects these as environment variables when the app runs.
`clinrag/config.py` reads them directly from the environment (there is no
`.env` on the Space).

---

## Step 7 — Commit and push

You need to authenticate with the token from Step 1. The simplest way is to put
it in the remote URL once:

```bash
git remote set-url origin https://cemg19:hf_YOUR_TOKEN@huggingface.co/spaces/cemg19/clinrag-htn

git add -A
git commit -m "Deploy ClinCite-HTN Streamlit app"
git push
```

(Alternatively, run `pip install huggingface_hub` then `huggingface-cli login`,
paste the token, and answer "yes" to save it as a git credential — then a plain
`git push` works without putting the token in the URL.)

---

## Step 8 — Watch the build and test it

1. Open your Space and click the **Logs** (or **App**) tab. You'll see
   HuggingFace install `requirements.txt`, then start Streamlit. The first start
   downloads BGE-large — expect a wait.
2. When the status turns **Running**, open the **App** tab.
3. Ask an example question (e.g. *"What defines stage 1 hypertension?"*). The
   **first** answer is slow (model loading); later ones are fast.
4. Toggle the model and open the **Evaluation** view to confirm both load.

---

## Step 9 — Put the live URL in the README

Once the Space works, update the placeholders in the main project
[README.md](../README.md):

- the `Live Demo` badge and link → your real Space URL
- the `loom.com/PLACEHOLDER` walkthrough link → your recorded Loom

Commit those to the **GitHub** repo (not the Space).

---

## Updating the app later

The Space is just a git repo, so to ship a change: copy the updated files into
the Space folder again and `git push`. If you re-ingested the corpus, re-copy
`data/lancedb/` too.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'clinrag'` | The `clinrag/` package folder isn't sitting next to the app in the Space (or wasn't uploaded). Open the Space **Files** tab and confirm `clinrag/` is in the **same folder as `streamlit_app.py`**. If your app lives in a `src/` folder, put `clinrag/` and `data/` inside `src/` too. |
| Build error: app file not found | `app_file:` in the Space README must point at wherever `streamlit_app.py` actually is (e.g. `app/streamlit_app.py` or `streamlit_app.py`), not the default `app.py` |
| App loads but every question errors about a missing key | The `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` secrets aren't set (Step 6), or the name is misspelled |
| "No index found" message in the app | `data/lancedb/` wasn't pushed — confirm it's in the Space repo (`git add -f data/lancedb`) |
| `git push` rejected: authentication failed | Use a **write**-role token (Step 1), not your password; check the remote URL has the token |
| Push rejected: file too large | If any single file exceeds ~10 MB, track it with git-LFS (`git lfs track "*.lance"`); our index files are normally under that |
| First question hangs for a minute | Expected — BGE-large is downloading. It's cached after the first run |
| App sleeps / is slow after idle | Free CPU Spaces sleep after inactivity; the first request after waking reloads the model |
| Evaluation tab says "No results yet" | Copy `data/eval/results_claude.csv` and `results_gemini.csv` into the Space (Step 4) |

## Cost note

Generation calls bill the Anthropic / Google keys you set as secrets. Retrieval
and embeddings run locally on the Space (no API cost). A public Space means
anyone can ask questions against your keys — keep an eye on usage, or make the
Space private if that's a concern.
