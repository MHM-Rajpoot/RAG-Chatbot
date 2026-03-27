"""Notebook 1B: load processed chunks, embed them, and build the vector DB."""

# ===========
# Cell 1: Install dependencies
# ===========
import subprocess
import sys


def pip_install(*packages: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


INSTALL_PACKAGES = False  # Set to True on first Colab GPU run.

if INSTALL_PACKAGES:
    pip_install("chromadb", "sentence-transformers", "tqdm")


# ===========
# Cell 2: Imports and Google Drive mount
# ===========
import json
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)


# ===========
# Cell 3: Configuration
# ===========
BASE_DIR = (
    "/content/drive/MyDrive/salford_rag"
    if IN_COLAB
    else os.path.abspath("./salford_rag")
)
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")

CHUNKS_PATH = os.path.join(PROCESSED_DIR, "chunks.jsonl")
VECTOR_DB_MANIFEST_PATH = os.path.join(METADATA_DIR, "vector_db_manifest.json")

COLLECTION_NAME = "salford_pages"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 64
UPSERT_BATCH_SIZE = 128
RESET_COLLECTION = True

os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)


# ===========
# Cell 4: Helpers
# ===========
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def batches(items: List[Dict], batch_size: int) -> Iterable[List[Dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def save_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


# ===========
# Cell 5: Load processed chunks
# ===========
chunks = load_jsonl(CHUNKS_PATH)
print(f"Loaded {len(chunks)} processed chunks from {CHUNKS_PATH}")

if not chunks:
    raise ValueError("No chunks found. Run notebook_1a_scrape_process.py first.")


# ===========
# Cell 6: Load embedding model
# ===========
embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
print("Embedding device:", embedding_device)

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=embedding_device)


# ===========
# Cell 7: Build persistent Chroma database
# ===========
client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

if RESET_COLLECTION:
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

for batch in tqdm(list(batches(chunks, UPSERT_BATCH_SIZE)), desc="Embedding and storing"):
    texts = [item["text"] for item in batch]
    embeddings = embedding_model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()

    ids = [item["chunk_id"] for item in batch]
    metadatas = [
        {
            "page_id": item["page_id"],
            "url": item["url"],
            "title": item["title"],
            "section": item["section"],
            "word_count": int(item["word_count"]),
            "crawl_timestamp": item["crawl_timestamp"],
        }
        for item in batch
    ]

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

print("Stored chunk count in Chroma:", collection.count())


# ===========
# Cell 8: Save vector DB manifest
# ===========
save_json(
    VECTOR_DB_MANIFEST_PATH,
    {
        "created_at": now_utc_iso(),
        "base_dir": BASE_DIR,
        "collection_name": COLLECTION_NAME,
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "vector_db_dir": VECTOR_DB_DIR,
        "chunks_path": CHUNKS_PATH,
        "chunk_count": collection.count(),
    },
)

print("Saved vector DB manifest to:", VECTOR_DB_MANIFEST_PATH)
print("Vector DB directory:", VECTOR_DB_DIR)
