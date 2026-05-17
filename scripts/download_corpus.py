"""Download the hypertension guideline corpus listed in data/corpus_manifest.csv.

Each manifest row names a source document, its URL, and a local path under
data/corpus/. Files already present are skipped, so the script is safe to re-run.

Usage:
    python scripts/download_corpus.py            # download everything missing
    python scripts/download_corpus.py --force    # re-download even if present
    python scripts/download_corpus.py --only htn-who-2021 htn-cdc-facts
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "data" / "corpus_manifest.csv"

# Some publisher sites reject requests without a browser-like User-Agent.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
TIMEOUT = 60


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _is_valid(body: bytes, fmt: str) -> bool:
    """A pdf row must return PDF bytes; html may return anything non-empty."""
    if not body:
        return False
    if fmt == "pdf":
        return body[:5].startswith(b"%PDF")
    return True


def _fetch_requests(url: str, referer: str) -> bytes:
    headers = {**HEADERS, "Referer": referer}
    resp = requests.get(url, headers=headers, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.content


def _fetch_curl(url: str, referer: str) -> bytes:
    """Fallback for sites that fingerprint and block the requests/urllib3 client."""
    curl = shutil.which("curl")
    if curl is None:
        return b""
    proc = subprocess.run(
        [curl, "-sL", "--max-time", str(TIMEOUT), "-A", HEADERS["User-Agent"],
         "-e", referer, url],
        capture_output=True,
    )
    return proc.stdout if proc.returncode == 0 else b""


def download(row: dict[str, str], force: bool) -> str:
    dest = REPO_ROOT / row["local_path"]
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        return f"SKIP  {row['doc_id']} (already present)"

    # download_url is the fetchable location; url is the canonical reference.
    fetch_url = row.get("download_url") or row["url"]
    # A same-origin Referer unblocks publishers that reject hotlinked downloads.
    parts = urlsplit(fetch_url)
    referer = f"{parts.scheme}://{parts.netloc}/"
    fmt = row["format"]

    try:
        body = _fetch_requests(fetch_url, referer)
    except requests.RequestException:
        body = b""

    # Some publishers fingerprint the Python HTTP client; retry via curl.
    if not _is_valid(body, fmt):
        body = _fetch_curl(fetch_url, referer)

    if not _is_valid(body, fmt):
        return (
            f"FAIL  {row['doc_id']}: could not fetch a valid {fmt} from {fetch_url} "
            f"(publisher block or moved URL) - download manually"
        )

    dest.write_bytes(body)
    return f"OK    {row['doc_id']} -> {row['local_path']} ({len(body) / 1024:.0f} KB)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--only", nargs="*", metavar="DOC_ID", help="limit to these doc_ids")
    args = parser.parse_args()

    rows = load_manifest()
    if args.only:
        wanted = set(args.only)
        rows = [r for r in rows if r["doc_id"] in wanted]
        missing = wanted - {r["doc_id"] for r in rows}
        for doc_id in sorted(missing):
            print(f"WARN  unknown doc_id: {doc_id}")

    failures = 0
    for row in rows:
        result = download(row, force=args.force)
        print(result)
        if result.startswith(("FAIL", "WARN")):
            failures += 1

    print(f"\n{len(rows) - failures}/{len(rows)} documents available.")
    if failures:
        print(
            "Some documents could not be fetched automatically (publisher blocks, "
            "moved URLs). Download them manually to the local_path in the manifest."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
