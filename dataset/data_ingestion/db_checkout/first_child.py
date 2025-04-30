#!/usr/bin/env python3
"""
Extract N parent‐level chunks along with their first child chunk
and export the result as JSON.
"""

import argparse
import json
from datetime import datetime
from pymongo import MongoClient

def iso(dt):
    return dt.isoformat() if isinstance(dt, datetime) else dt

def main():
    p = argparse.ArgumentParser(
        description="Dump parent chunks + their first child to JSON"
    )
    p.add_argument(
        "--uri", default="mongodb://localhost:27017",
        help="MongoDB URI (default: mongodb://localhost:27017)"
    )
    p.add_argument(
        "--db", default="anxiety",
        help="Database name (default: anxiety)"
    )
    p.add_argument(
        "--col", default="guidelines",
        help="Collection name (default: guidelines)"
    )
    p.add_argument(
        "--limit", type=int, default=2,
        help="Number of parent chunks to extract (default: 2)"
    )
    p.add_argument(
        "--out", default="parents_with_first_child.json",
        help="Output JSON filename"
    )
    args = p.parse_args()

    client = MongoClient(args.uri)
    col = client[args.db][args.col]

    # 1) Fetch up to N parent docs
    parents = list(
        col.find({"doc_level": "parent"})
           .sort([("parent_id", 1)])
           .limit(args.limit)
    )

    result = []
    for parent in parents:
        # 2) Fetch the first child for this parent
        child = col.find_one(
            {
                "doc_level": "child",
                "slug": parent["slug"],
                "parent_id": parent["parent_id"]
            },
            sort=[("child_id", 1)]
        )

        entry = {
            "parent": {
                "parent_id": parent["parent_id"],
                "slug": parent["slug"],
                "title": parent.get("title"),
                "text": parent.get("text"),
                "ingested_at": iso(parent.get("ingested_at"))
            },
            "first_child": None
        }

        if child:
            entry["first_child"] = {
                "child_id": child["child_id"],
                "text": child.get("text"),
                "embedding": child.get("embedding"),
                "ingested_at": iso(child.get("ingested_at"))
            }

        result.append(entry)

    # 3) Write to JSON file
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print(f"Exported {len(result)} entries to {args.out}")

if __name__ == "__main__":
    main()