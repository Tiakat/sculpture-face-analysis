#!/usr/bin/env python3
"""Corpus ingestion CLI.

    python scripts/ingest.py met artic wikimedia            # no API keys needed
    python scripts/ingest.py europeana smithsonian harvard  # need EUROPEANA_API_KEY / SMITHSONIAN_API_KEY / HARVARD_API_KEY
    python scripts/ingest.py met --query "moche portrait" --query "roman portrait bust" --limit 200
    python scripts/ingest.py all

After ingestion:  python scripts/build_reference_index.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from connectors.museums import CONNECTORS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", help=f"one or more of: {', '.join(CONNECTORS)} | all")
    ap.add_argument("--query", action="append", help="search query (repeatable); default = generic face/portrait queries")
    ap.add_argument("--limit", type=int, default=300, help="max objects per query per source")
    args = ap.parse_args()
    sources = list(CONNECTORS) if "all" in args.sources else args.sources
    total = 0
    for s in sources:
        fn = CONNECTORS.get(s)
        if not fn:
            print(f"unknown source {s}")
            continue
        print(f"=== {s} ===")
        try:
            total += fn(args.query, args.limit) if s != "wikimedia" else fn(None, args.limit)
        except KeyboardInterrupt:
            print("interrupted")
            break
    print(f"\nTotal new records: {total}. Next: python scripts/build_reference_index.py && python scripts/load_corpus_to_db.py")


if __name__ == "__main__":
    main()
