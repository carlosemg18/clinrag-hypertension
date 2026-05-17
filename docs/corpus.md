# Corpus — Hypertension Clinical Guidelines

This document describes the source corpus for ClinCite-HTN: what is in it, where it
came from, and the licensing under which each document is used.

## Why these documents

The corpus is deliberately **narrow** — hypertension only. A focused corpus produces
higher retrieval quality than a broad one, which directly improves faithfulness and
citation accuracy (the metrics this project measures). Documents are drawn from
authoritative, publicly available clinical guidelines and public-health resources.

## Source inventory

The authoritative inventory is [`data/corpus_manifest.csv`](../data/corpus_manifest.csv).
Each row records: `doc_id`, `title`, `source`, `publisher`, `pub_date`, `url`
(canonical reference), `download_url` (fetchable location, when it differs),
`format`, `license`, and `local_path`.

| doc_id | Source | What it is |
|---|---|---|
| `htn-accaha-2017` | ACC/AHA | 2017 US clinical practice guideline — prevention, detection, evaluation, and management of high blood pressure in adults |
| `htn-jnc8-2014` | JNC 8 | 2014 evidence-based guideline (Eighth Joint National Committee panel) — treatment thresholds, goals, and drug selection |
| `htn-nice-ng136` | NICE | NG136 — UK guideline for diagnosis and management of hypertension in adults |
| `htn-nice-ng136-visual` | NICE | NG136 visual summary — diagnosis and treatment algorithm |
| `htn-uspstf-2021` | USPSTF | 2021 final recommendation statement — screening for hypertension in adults |
| `htn-who-2021` | WHO | 2021 guideline for the pharmacological treatment of hypertension in adults |
| `htn-cdc-about` | CDC | Public-health overview — what high blood pressure is |
| `htn-cdc-facts` | CDC | Prevalence and burden statistics for the US |
| `htn-cdc-risk` | CDC | Risk factors for high blood pressure |

These nine source documents are the *ingestion units*. During indexing (Days 3–5) they
are parsed and split into ~75–100 retrievable chunks (512 tokens, 50 overlap), each
carrying `doc_id` and section metadata so every generated claim can cite its origin.

## Provenance and licensing

Licenses were assessed from the publisher of each document. Verify before any
redistribution beyond this educational project.

- **USPSTF, CDC** — works of the US federal government; **public domain**.
- **WHO** — published under **CC BY-NC-SA 3.0 IGO** (non-commercial, share-alike).
- **NICE** — © NICE, reusable under the NICE UK Open Content Licence.
- **ACC/AHA (2017 guideline)** — © American Heart Association, published in
  *Hypertension*. Free to access for clinical/educational use; not an open license.
- **JNC 8 (2014 guideline)** — © American Medical Association, published in *JAMA*.
  Free to access for educational use; not an open license.

The ACC/AHA and JNC 8 documents are copyrighted. They are used here only for
non-commercial, educational retrieval-and-citation research. The system surfaces short,
attributed excerpts with links back to the original publication — it does not
redistribute the documents themselves.

## Reproducing the corpus

```bash
python scripts/download_corpus.py            # fetch everything missing
python scripts/download_corpus.py --force    # re-download all
python scripts/download_corpus.py --only htn-who-2021
```

The script reads the manifest, fetches each `download_url`, validates the response
(PDF rows must return PDF bytes), and writes to `data/corpus/`. It is idempotent —
existing files are skipped. For publishers that fingerprint and block the Python HTTP
client, it automatically retries the download with `curl`.

Downloaded files in `data/corpus/` are **git-ignored** — the manifest, not the binaries,
is the version-controlled record of the corpus.

### Known manual download

- **`htn-who-2021`** — the WHO IRIS repository (`iris.who.int`) sits behind a bot
  challenge that blocks automated fetches. Download the PDF manually from the
  publication page (`url` in the manifest) and save it to
  `data/corpus/htn-who-2021.pdf`.

## Limitations

- **Guidelines are dated.** Recommendations evolve; JNC 8 (2014) and ACC/AHA (2017)
  in particular predate later evidence. Each citation in the system includes the
  publication date so users can weigh recency.
- **Mixed formats.** Guidelines are PDFs; CDC and USPSTF sources are HTML pages.
  The ingestion pipeline handles both.
- **English-language, US/UK/global mix.** Thresholds and drug recommendations differ
  across these bodies (e.g. the ACC/AHA 130/80 vs. NICE 140/90 definitions). This is
  intentional — the system cites which body a claim comes from rather than reconciling
  them.
