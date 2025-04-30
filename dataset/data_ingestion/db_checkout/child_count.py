#!/usr/bin/env python3
"""
Export each parent‐chunk along with its child count and child IDs
from the `anxiety.guidelines` collection into a JSON file.
"""

import argparse
import json
from pymongo import MongoClient

def main():
    parser = argparse.ArgumentParser(
        description="Export parent‐child counts from MongoDB to JSON"
    )
    parser.add_argument(
        "--uri",
        default="mongodb://localhost:27017",
        help="MongoDB connection URI",
    )
    parser.add_argument(
        "--db",
        default="anxiety",
        help="Database name (default: anxiety)",
    )
    parser.add_argument(
        "--col",
        default="guidelines",
        help="Collection name (default: guidelines)",
    )
    parser.add_argument(
        "--out",
        default="parent_child_counts.json",
        help="Output JSON filename",
    )
    args = parser.parse_args()

    client = MongoClient(args.uri)
    col = client[args.db][args.col]

    pipeline = [
        { "$match": { "doc_level": "parent" } },
        { "$lookup": {
            "from": args.col,
            "let": { "slug": "$slug", "pid": "$parent_id" },
            "pipeline": [
                { "$match": {
                    "$expr": {
                        "$and": [
                            { "$eq": ["$doc_level", "child"] },
                            { "$eq": ["$slug",      "$$slug"] },
                            { "$eq": ["$parent_id","$$pid"] }
                        ]
                    }
                }},
                { "$group": {
                    "_id":        None,
                    "child_count": { "$sum": 1 },
                    "child_ids":   { "$push": "$child_id" }
                }}
            ],
            "as": "childInfo"
        }},
        { "$unwind": {
            "path": "$childInfo",
            "preserveNullAndEmptyArrays": True
        }},
        { "$project": {
            "_id":         0,
            "slug":        1,
            "title":       1,
            "parent_id":   1,
            "child_count": { "$ifNull": ["$childInfo.child_count", 0] },
            "child_ids":   { "$ifNull": ["$childInfo.child_ids",   []] }
        }}
    ]

    cursor = col.aggregate(pipeline)
    results = list(cursor)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(results)} parent‐chunks to {args.out}")

if __name__ == "__main__":
    main()