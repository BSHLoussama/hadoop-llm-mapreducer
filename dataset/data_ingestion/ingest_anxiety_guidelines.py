#!/usr/bin/env python
"""
Ingest anxiety‑guideline PDFs into MongoDB (parents + embedded children).

Index strategy
──────────────
1. slug + parent_id            UNIQUE   for doc_level="parent"
2. slug + child_id             UNIQUE   for doc_level="child"
"""
from __future__ import annotations
import argparse, pathlib, re, datetime, os, json, requests, sys
from urllib.parse import urlparse

from pymongo import MongoClient, errors
from sentence_transformers import SentenceTransformer
import numpy as np

# ───────────────────────── Embeddings ─────────────────────────
EMBED = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed(txt: str) -> list[float]:
    return EMBED.encode(txt[:4096]).astype(np.float32).tolist()

# ─────────────────────── PDF → text helpers ───────────────────
try:
    from pdfminer.high_level import extract_text as pdf2txt
except ImportError:
    from pypdf import PdfReader
    def pdf2txt(path: pathlib.Path) -> str:
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)

# ─────────────────────── Catalogue loading ───────────────────
with open("guidelines_catalogue.json", encoding="utf-8") as fh:
    data = json.load(fh)
GUIDELINES = data["guidelines"] if isinstance(data, dict) else data

SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
def safe_name(title: str) -> str:
    return SAFE.sub("_", title)[:80] + ".pdf"

# ─────────────────────── Text splitting ──────────────────────
def split_text(text: str, max_len: int, overlap: int) -> list[str]:
    out, start = [], 0
    while start < len(text):
        end = min(start + max_len, len(text))
        out.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return out

def hierarchical_split(text: str) -> list[dict]:
    parents: list[dict] = []
    for i, medium in enumerate(split_text(text, 2000, 200)):
        children: list[dict] = []
        for j, small in enumerate(split_text(medium, 500, 50)):
            children.append({
                "text": small,
                "embedding": embed(small),
                "doc_level": "child",
                "child_id": f"child_{i}_{j}",
            })
        parents.append({
            "text": medium,
            "doc_level": "parent",
            "parent_id": f"parent_{i}",
            "children": children,
        })
    return parents

# ─────────────────────── Index management ────────────────────
def ensure_indexes(col):
    existing = {ix["name"] for ix in col.list_indexes()}

    # Drop any old/colliding indexes
    for bad in (
        "parent_id_1", "child_id_1",
        "slug_parent_doc_uni", "slug_child_uni",
        "slug_parent_id_1", "slug_child_id_1",
    ):
        if bad in existing:
            print(f"[IDX] drop {bad}")
            col.drop_index(bad)

    # Unique on (slug, parent_id) only for parents
    col.create_index(
        [("slug", 1), ("parent_id", 1)],
        name="slug_parent_uni",
        unique=True,
        partialFilterExpression={"doc_level": "parent"},
    )

    # Unique on (slug, child_id) only for children
    col.create_index(
        [("slug", 1), ("child_id", 1)],
        name="slug_child_uni",
        unique=True,
        partialFilterExpression={"doc_level": "child"},
    )

    print("[IDX] ensured partial unique indexes")

# ─────────────────────── Ingestion routine ───────────────────
def ingest(uri, dbn, coln, folder: pathlib.Path, overwrite=False):
    client = MongoClient(uri)
    col = client[dbn][coln]
    ensure_indexes(col)

    folder.mkdir(exist_ok=True)

    for g in GUIDELINES:
        slug  = urlparse(g["url"]).netloc + urlparse(g["url"]).path
        title = g["title"].strip()
        pdf_path = folder / safe_name(title)

        # Skip if already ingested and not overwriting
        if col.find_one({"slug": slug}) and not overwrite:
            print(f"[SKIP] {title}")
            continue

        # Download PDF
        if overwrite or not pdf_path.exists():
            print(f"[DL  ] {title}")
            try:
                r = requests.get(g["url"], timeout=60)
                r.raise_for_status()
                pdf_path.write_bytes(r.content)
            except Exception as e:
                print(f"[ERR] download {title}: {e}")
                continue

        # Extract text
        try:
            text = pdf2txt(pdf_path)
        except Exception as e:
            print(f"[ERR] parse {title}: {e}")
            continue

        # Split and upsert
        for parent in hierarchical_split(text):
            # Parent document
            col.replace_one(
                {"slug": slug,
                 "parent_id": parent["parent_id"],
                 "doc_level": "parent"},
                {
                    **g,
                    "slug": slug,
                    "text": parent["text"],
                    "doc_level": "parent",
                    "parent_id": parent["parent_id"],
                    "ingested_at": datetime.datetime.utcnow(),
                },
                upsert=True,
            )

            # Child documents
            for child in parent["children"]:
                col.replace_one(
                    {"slug": slug,
                     "child_id": child["child_id"],
                     "doc_level": "child"},
                    {
                        **g,
                        "slug": slug,
                        "text": child["text"],
                        "embedding": child["embedding"],
                        "doc_level": "child",
                        "child_id": child["child_id"],
                        "parent_id": parent["parent_id"],
                        "ingested_at": datetime.datetime.utcnow(),
                    },
                    upsert=True,
                )

        print(f"[OK  ] inserted {title}")

# ───────────────────────── CLI entry‑point ─────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mongo",  default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    ap.add_argument("--db",     default="anxiety")
    ap.add_argument("--col",    default="guidelines")
    ap.add_argument("--folder", default="guidelines", type=pathlib.Path)
    ap.add_argument("--overwrite", action="store_true",
                    help="Redownload PDFs and overwrite existing MongoDB docs")
    args = ap.parse_args()

    try:
        ingest(
            uri=args.mongo,
            dbn=args.db,
            coln=args.col,
            folder=args.folder,
            overwrite=args.overwrite,
        )
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user")