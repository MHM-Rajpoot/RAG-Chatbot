# University of Salford RAG Chatbot

This project is a Colab-friendly Retrieval-Augmented Generation (RAG) pipeline for answering questions about the University of Salford using content scraped from official `salford.ac.uk` pages.

The current code is split into three notebook-style Python files:

- [notebook_1a_scrape_process.py](./notebook_1a_scrape_process.py)
- [notebook_1b_embed_vector_db.py](./notebook_1b_embed_vector_db.py)
- [notebook_2_chat_cli.py](./notebook_2_chat_cli.py)

Each file is written like a notebook and uses `# ===========` markers to separate cells.

## What This Project Does

The pipeline works in 3 stages:

1. Scrape University of Salford pages, clean them, and split them into chunks.
2. Turn those chunks into embeddings and store them in a persistent Chroma vector database.
3. Load the saved vector database, retrieve relevant chunks for a question, and use a 7B language model to generate an answer with sources.

The main idea is:

- scrape once
- store once
- reuse the database many times
- only re-scrape when you want to refresh the knowledge base

## Current Models Used

### Embedding Model

The embedding model used in both indexing and query retrieval is:

- `sentence-transformers/all-MiniLM-L6-v2`

Where it is used:

- [notebook_1b_embed_vector_db.py](./notebook_1b_embed_vector_db.py) to embed stored chunks
- [notebook_2_chat_cli.py](./notebook_2_chat_cli.py) to embed each user question

Notes:

- It runs on `cuda` if a GPU is available.
- It falls back to `cpu` if no GPU is available.
- Embeddings are normalized before storing and querying.

### LLM

The answer-generation model used in chat is:

- `Qwen/Qwen2.5-7B-Instruct`

Where it is used:

- [notebook_2_chat_cli.py](./notebook_2_chat_cli.py)

How it is loaded:

- `device_map="auto"`
- 4-bit quantization is enabled by default with `USE_4BIT = True`
- `bitsandbytes` is used when a GPU is available
- compute dtype is `bfloat16` when supported, otherwise `float16`

Generation settings in the current code:

- `TOP_K = 4`
- `MAX_INPUT_CHARS = 6000`
- `MAX_NEW_TOKENS = 350`
- `TEMPERATURE = 0.2`

## Vector Database

The vector database is Chroma with persistent local storage.

Current configuration:

- collection name: `salford_pages`
- similarity space: `cosine`

Where it is built:

- [notebook_1b_embed_vector_db.py](./notebook_1b_embed_vector_db.py)

Where it is used:

- [notebook_2_chat_cli.py](./notebook_2_chat_cli.py)

Default storage location:

- Colab: `/content/drive/MyDrive/salford_rag/vector_db`
- local fallback: `./salford_rag/vector_db`

## Storage Layout

By default the project stores data in:

- Colab: `/content/drive/MyDrive/salford_rag`
- local fallback: `./salford_rag`

Generated files and folders:

- `raw/pages.jsonl`
  - raw extracted page records
- `processed/chunks.jsonl`
  - cleaned chunk records used for embeddings
- `metadata/crawl_manifest.json`
  - crawl settings and output counts
- `metadata/vector_db_manifest.json`
  - vector DB settings and embedding model info
- `vector_db/`
  - persistent Chroma database files

## How The Pipeline Works

### Notebook 1A: Scrape and Process

File:

- [notebook_1a_scrape_process.py](./notebook_1a_scrape_process.py)

Purpose:

- scrape official Salford pages
- clean the HTML
- extract readable text
- chunk the text
- save the results to Google Drive

Current crawl settings:

- allowed domains:
  - `www.salford.ac.uk`
  - `salford.ac.uk`
- seed URLs:
  - `https://www.salford.ac.uk/`
  - `https://www.salford.ac.uk/study`
  - `https://www.salford.ac.uk/courses`
- `MAX_PAGES = 50`
- `MAX_DEPTH = 2`
- `REQUEST_TIMEOUT = 20`
- `REQUEST_DELAY_SECONDS = 1.0`
- `USE_ROBOTS_TXT = True`
- `MIN_PAGE_WORDS = 80`
- `CHUNK_SIZE_WORDS = 220`
- `CHUNK_OVERLAP_WORDS = 40`

What it removes from pages:

- scripts
- styles
- nav
- footer
- header
- forms
- cookie banner style elements

Output of Notebook 1A:

- raw page records in `pages.jsonl`
- processed chunks in `chunks.jsonl`
- a crawl manifest in `crawl_manifest.json`

GPU need:

- not required

### Notebook 1B: Embed and Build Vector DB

File:

- [notebook_1b_embed_vector_db.py](./notebook_1b_embed_vector_db.py)

Purpose:

- load processed chunks from Notebook 1A
- generate embeddings
- create the persistent Chroma database
- save vector DB metadata

Current embedding settings:

- embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- `EMBED_BATCH_SIZE = 64`
- `UPSERT_BATCH_SIZE = 128`
- `RESET_COLLECTION = True`

Important behavior:

- if `RESET_COLLECTION = True`, the existing Chroma collection is deleted and rebuilt
- the code stores:
  - chunk text as `documents`
  - metadata like URL, title, section, word count, and crawl timestamp
  - precomputed embeddings

GPU need:

- optional, but helpful for large numbers of chunks

### Notebook 2: Chat and Retrieval

File:

- [notebook_2_chat_cli.py](./notebook_2_chat_cli.py)

Purpose:

- load the persistent Chroma vector DB
- embed a user question
- retrieve the most relevant chunks
- build a grounded prompt
- generate an answer using the 7B model
- print the answer and source URLs

How retrieval works:

1. Load `vector_db_manifest.json`
2. Reuse the same embedding model used during indexing
3. Embed the user question
4. Query Chroma for the top `TOP_K` chunks
5. Build a context block that includes:
   - title
   - section
   - URL
   - chunk text

How generation works:

1. A system prompt tells the model to answer only from retrieved context
2. The prompt includes the user question and retrieved chunks
3. The model generates a concise answer
4. The code also prints a `Sources` list from retrieved metadata

The system prompt currently instructs the model to:

- use only the retrieved context
- say it does not know if the answer is not in the context
- keep the answer concise and factual
- include a final sources section using the provided URLs only

Interactive usage:

- Cell 8 runs a single test question
- Cell 9 contains a CLI-style loop
- set `RUN_CHAT_LOOP = True` to chat interactively

GPU need:

- recommended
- if no GPU is available, the code warns that a 7B model may be too slow or may not fit in memory

## Run Order

Run the notebooks in this order:

1. [notebook_1a_scrape_process.py](./notebook_1a_scrape_process.py)
2. [notebook_1b_embed_vector_db.py](./notebook_1b_embed_vector_db.py)
3. [notebook_2_chat_cli.py](./notebook_2_chat_cli.py)

Typical workflow:

1. Run Notebook 1A once to scrape and prepare the data.
2. Run Notebook 1B once to build the vector database.
3. Run Notebook 2 whenever you want to ask questions.
4. Re-run 1A and 1B only when you want fresh website data.

## Colab Usage Notes

The files are written for Google Colab first, but they also have a local fallback path.

Important details:

- Google Drive is mounted automatically when running in Colab
- package install cells are present in each file
- `INSTALL_PACKAGES = False` by default in all files
- set `INSTALL_PACKAGES = True` the first time you run each notebook in Colab

Recommended environment split:

- Notebook 1A: CPU is fine
- Notebook 1B: GPU is useful for large embedding jobs
- Notebook 2: GPU is recommended for the 7B chat model

## Main Python Dependencies

Notebook 1A:

- `requests`
- `beautifulsoup4`
- `lxml`
- `tqdm`

Notebook 1B:

- `chromadb`
- `sentence-transformers`
- `tqdm`

Notebook 2:

- `chromadb`
- `sentence-transformers`
- `transformers`
- `accelerate`
- `bitsandbytes`

## What The Answer Output Looks Like

The current chat notebook returns:

- the generated answer
- a list of source URLs from the retrieved chunks
- the retrieval payload inside Python if you want to inspect it further

This means answers are grounded in stored Salford page chunks instead of relying only on the LLM's internal knowledge.

## Current Limitations

This is a strong MVP base, but a few things are still simple in the current code:

- scraping uses general HTML extraction, not a Salford-specific content parser
- there is no reranking step yet
- the first heading is used as the chunk `section`
- there is no FastAPI server yet
- there is no conversation memory yet
- there is no automatic scheduled re-scraping yet

## Summary

Current architecture in one line:

`salford.ac.uk pages -> cleaned text chunks -> all-MiniLM-L6-v2 embeddings -> Chroma vector DB -> Qwen2.5-7B-Instruct answer generation`

That gives you a reusable, Colab-friendly RAG pipeline where scraping and indexing happen once, and chat can run repeatedly against the saved vector database.
