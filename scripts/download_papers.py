#!/usr/bin/env python3
"""Re-download the paper corpus catalogued in papers/catalog.json.

PDFs are gitignored, so a fresh clone has the catalog but not the binaries.
Run from the workspace root:  python scripts/download_papers.py
"""
import json
import os
import time

import requests

CATALOG = "papers/catalog.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (research literature review)"}


def main():
    papers = json.load(open(CATALOG))
    os.makedirs("papers", exist_ok=True)
    ok = failed = 0
    for p in papers:
        path = p["file"]
        if os.path.exists(path) and os.path.getsize(path) > 20000:
            ok += 1
            continue
        urls = []
        if p.get("arxiv"):
            urls.append(f"https://arxiv.org/pdf/{p['arxiv']}")
        if p.get("pdf"):
            urls.append(p["pdf"])
        for url in urls:
            try:
                resp = requests.get(url, headers=HEADERS, timeout=90)
            except requests.RequestException:
                continue
            if resp.status_code == 200 and resp.content[:4] == b"%PDF":
                open(path, "wb").write(resp.content)
                ok += 1
                break
            time.sleep(1)
        else:
            failed += 1
            print("FAILED:", p["title"])
        time.sleep(1.2)
    print(f"{ok} available, {failed} failed")


if __name__ == "__main__":
    main()
