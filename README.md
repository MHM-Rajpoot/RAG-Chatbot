# University of Salford RAG Chatbot

A fully self-contained **Retrieval-Augmented Generation (RAG)** pipeline that answers questions about the University of Salford using content scraped directly from [salford.ac.uk](https://www.salford.ac.uk).

Instead of relying on a language model's internal knowledge (which may be outdated), every answer is grounded in real, recently scraped content from the university's own website. If the answer isn't in the retrieved sources, the model says so.

---

## What This Project Does

1. **Scrapes** official University of Salford web pages and cleans the HTML
2. **Chunks** the cleaned text into overlapping word-level segments
3. **Embeds** each chunk into a vector using a sentence transformer model
4. **Stores** those vectors in a persistent ChromaDB vector database
5. **Answers** user questions by retrieving the most relevant chunks and passing them to a large language model

The idea is: **scrape once, index once, chat many times.** Only re-run scraping and indexing when you want fresh data.

---

## Architecture

```
salford.ac.uk pages
        │
        ▼
 Clean & chunk text          ← notebook_1a_scrape_process.py
        │
        ▼
 Generate embeddings
 Store in ChromaDB           ← notebook_1b_embed_vector_db.py
        │
        ▼
 Embed user question
 Retrieve top-K chunks
 Generate answer with LLM   ← notebook_2_chat_cli.py
        │
        ▼
 Answer + source URLs
```

---

## Project Files

| File | Purpose |
|------|---------|
| `notebook_1a_scrape_process.py` | Crawl salford.ac.uk, clean HTML, chunk text, save to disk |
| `notebook_1b_embed_vector_db.py` | Embed chunks and build the persistent Chroma vector database |
| `notebook_2_chat_cli.py` | Load vector DB, retrieve context, generate answers with the LLM |
| `README.md` | This file |

---

## Technologies Used

### Core Pipeline

| Technology | Used In | Purpose |
|------------|---------|---------|
| Python 3 | All stages | Core programming language |
| Google Colab | All stages | Cloud notebook runtime with free GPU |
| Google Drive | All stages | Persistent storage for data and vector DB |

### Stage 1 — Scraping & Processing

| Technology | Purpose |
|------------|---------|
| `requests` | HTTP page fetching |
| `BeautifulSoup4` + `lxml` | HTML parsing and noise removal |
| `tqdm` | Progress bar during crawl |
| `urllib.robotparser` | Respects robots.txt rules |

### Stage 2 — Embedding & Indexing

| Technology | Purpose |
|------------|---------|
| `sentence-transformers` | Generates vector embeddings from text |
| `all-MiniLM-L6-v2` | Lightweight 384-dimension embedding model |
| `ChromaDB` | Persistent local vector database (cosine similarity) |
| `PyTorch` | Backend for embedding inference |
| `tqdm` | Progress bar during batch embedding |

### Stage 3 — Retrieval & Generation

| Technology | Purpose |
|------------|---------|
| `ChromaDB` | Retrieves top-K most similar chunks to a query |
| `sentence-transformers` | Embeds the user question at query time |
| `Qwen/Qwen2.5-7B-Instruct` | 7B parameter LLM for answer generation |
| `Hugging Face Transformers` | Model loading and text generation |
| `bitsandbytes` | 4-bit quantisation to reduce GPU memory usage |
| `accelerate` | Automatic device mapping across GPU/CPU |

---

## How Each Stage Works

### Stage 1 — Scrape and Process (`notebook_1a_scrape_process.py`)

- Starts a breadth-first crawl from three seed URLs
- Respects `robots.txt` and adds a 1-second delay between requests
- Skips non-HTML resources (PDFs, images, `.docx`, etc.)
- Removes noise: `<nav>`, `<footer>`, `<header>`, `<script>`, cookie banners
- Extracts clean text and splits it into overlapping word-level chunks
- Saves raw pages to `pages.jsonl` and processed chunks to `chunks.jsonl`

Key settings:

```python
SEED_URLS = [
    "https://www.salford.ac.uk/",
    "https://www.salford.ac.uk/study",
    "https://www.salford.ac.uk/courses",
]
MAX_PAGES = 50
MAX_DEPTH = 2
CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
REQUEST_DELAY_SECONDS = 1.0
```

### Stage 2 — Embed and Index (`notebook_1b_embed_vector_db.py`)

- Loads the chunks saved by Stage 1
- Passes them through `all-MiniLM-L6-v2` in batches to produce 384-dim embeddings
- Stores embeddings, raw text, and metadata (URL, title, section) in ChromaDB
- Saves a `vector_db_manifest.json` so Stage 3 can reload the correct model and collection

Key settings:

```python
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "salford_pages"
EMBED_BATCH_SIZE = 64
RESET_COLLECTION = True   # ⚠️ deletes and rebuilds the collection if True
```

### Stage 3 — Retrieve and Answer (`notebook_2_chat_cli.py`)

- Loads the saved ChromaDB collection and embedding model
- Embeds the user's question using the same model as Stage 2
- Retrieves the top-K most similar chunks by cosine similarity
- Builds a context block with title, section, URL, and chunk text
- Passes the context and question to `Qwen2.5-7B-Instruct`
- Returns a grounded answer and a list of source URLs

Key settings:

```python
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TOP_K = 4
MAX_NEW_TOKENS = 350
TEMPERATURE = 0.2
USE_4BIT = True
```

The system prompt instructs the model to:
- Answer only from the retrieved context
- Say it does not know if the answer is not in the context
- Keep answers concise and factual
- Always include a Sources section with the retrieved URLs

---

## Run Order

```
1. notebook_1a_scrape_process.py   ← run once to scrape
2. notebook_1b_embed_vector_db.py  ← run once to index
3. notebook_2_chat_cli.py          ← run whenever you want to chat
```

Re-run steps 1 and 2 only when you want to refresh the knowledge base with updated website content.

---

## Storage Layout

```
salford_rag/
├── raw/
│   └── pages.jsonl                 ← raw extracted page records
├── processed/
│   └── chunks.jsonl                ← cleaned chunks used for embedding
├── metadata/
│   ├── crawl_manifest.json         ← crawl settings and page/chunk counts
│   └── vector_db_manifest.json     ← embedding model name and DB config
└── vector_db/
    └── ...                         ← persistent ChromaDB files
```

Default base paths:

- **Colab:** `/content/drive/MyDrive/salford_rag/`
- **Local:** `./salford_rag/`

---

## Colab Setup Notes

- Set `INSTALL_PACKAGES = True` the first time you run each notebook in Colab
- Google Drive is mounted automatically when running inside Colab
- Stage 1 runs fine on a **CPU** instance
- Stage 2 benefits from a **GPU** for large embedding jobs
- Stage 3 **requires a GPU** — without one, the 7B model will be very slow or may not fit in RAM
- To use the interactive chat loop in Stage 3, set `RUN_CHAT_LOOP = True`

---

## Dependencies

**Stage 1 (`notebook_1a`)**
```
requests
beautifulsoup4
lxml
tqdm
```

**Stage 2 (`notebook_1b`)**
```
chromadb
sentence-transformers
tqdm
```

**Stage 3 (`notebook_2`)**
```
chromadb
sentence-transformers
transformers
accelerate
bitsandbytes
```

---

## Sample Output

```
Question: What postgraduate study options are available?

The University of Salford offers a wide range of postgraduate programmes
across business, engineering, health sciences, and the arts. Options include
taught Masters degrees (MSc, MA, MBA), research degrees (MPhil, PhD), and
professional doctorates...

Sources:
- https://www.salford.ac.uk/study/postgraduate
- https://www.salford.ac.uk/courses
```

---

## Current Limitations

- No reranking step — chunks are used in cosine similarity order only
- No conversation memory — each question is answered independently with no history
- No FastAPI server — currently CLI only
- No scheduled re-scraping — the knowledge base must be refreshed manually
- Section label uses only the first heading found on a page
- `MAX_PAGES = 50` covers a small fraction of the full Salford website

---

## License

This project is for educational and research purposes. Web scraping is performed respectfully with `robots.txt` compliance and rate limiting. All scraped content remains the property of the University of Salford.
