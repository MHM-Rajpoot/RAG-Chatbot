"""Notebook 1A: scrape, clean, chunk, and save processed data to Google Drive.

Each section below is meant to behave like a Colab notebook cell.
"""

# ===========
# Cell 1: Install dependencies
# ===========
import subprocess
import sys


def pip_install(*packages: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


INSTALL_PACKAGES = False  # Set to True on first Colab run.

if INSTALL_PACKAGES:
    pip_install("requests", "beautifulsoup4", "lxml", "tqdm")


# ===========
# Cell 2: Imports and Google Drive mount
# ===========
import hashlib
import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
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
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")

RAW_PAGES_PATH = os.path.join(RAW_DIR, "pages.jsonl")
CHUNKS_PATH = os.path.join(PROCESSED_DIR, "chunks.jsonl")
CRAWL_MANIFEST_PATH = os.path.join(METADATA_DIR, "crawl_manifest.json")

ALLOWED_DOMAINS = {"www.salford.ac.uk", "salford.ac.uk"}
SEED_URLS = [
    "https://www.salford.ac.uk/",
    "https://www.salford.ac.uk/study",
    "https://www.salford.ac.uk/courses",
]

USER_AGENT = "SalfordRAGBot/0.1 (+https://www.salford.ac.uk/)"
MAX_PAGES = 50
MAX_DEPTH = 2
REQUEST_TIMEOUT = 20
REQUEST_DELAY_SECONDS = 1.0
USE_ROBOTS_TXT = True
MIN_PAGE_WORDS = 80
CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)


# ===========
# Cell 4: Helpers
# ===========
SKIP_FILE_SUFFIXES = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".zip",
    ".mp4",
    ".mp3",
)

NOISE_SELECTORS = [
    "script",
    "style",
    "noscript",
    "svg",
    "form",
    "nav",
    "footer",
    "header",
    "aside",
    ".cookie-banner",
    ".cookie-consent",
    ".newsletter-signup",
]


@dataclass
class PageRecord:
    page_id: str
    url: str
    title: str
    text: str
    headings: List[str]
    crawl_timestamp: str


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    cleaned = parsed._replace(fragment="", query="")
    return urlunparse(cleaned).rstrip("/")


def is_allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.netloc.lower() not in ALLOWED_DOMAINS:
        return False
    if parsed.path.lower().endswith(SKIP_FILE_SUFFIXES):
        return False
    return True


def load_robot_parser(domain: str) -> Optional[RobotFileParser]:
    if not USE_ROBOTS_TXT:
        return None

    parser = RobotFileParser()
    parser.set_url(f"https://{domain}/robots.txt")
    try:
        parser.read()
        return parser
    except Exception:
        return None


ROBOT_PARSERS = {domain: load_robot_parser(domain) for domain in ALLOWED_DOMAINS}


def can_fetch(url: str) -> bool:
    if not USE_ROBOTS_TXT:
        return True

    parser = ROBOT_PARSERS.get(urlparse(url).netloc.lower())
    if parser is None:
        return True

    try:
        return parser.can_fetch(USER_AGENT, url)
    except Exception:
        return True


def fetch_html(url: str) -> Optional[str]:
    headers = {"User-Agent": USER_AGENT}
    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        if "text/html" not in response.headers.get("Content-Type", ""):
            return None
        return response.text
    except requests.RequestException:
        return None


def clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    for selector in NOISE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()
    return soup


def extract_page(html: str, url: str) -> Optional[PageRecord]:
    soup = BeautifulSoup(html, "lxml")
    clean_soup(soup)

    container = soup.find("main") or soup.find("article") or soup.body or soup
    title = ""
    if soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(" ", strip=True)
    elif soup.find("h1"):
        title = soup.find("h1").get_text(" ", strip=True)
    else:
        title = url

    headings = [
        heading.get_text(" ", strip=True)
        for heading in container.find_all(["h1", "h2", "h3"])
        if heading.get_text(strip=True)
    ]

    lines = []
    for line in container.get_text("\n", strip=True).splitlines():
        normalized = re.sub(r"\s+", " ", line).strip()
        if normalized:
            lines.append(normalized)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text.split()) < MIN_PAGE_WORDS:
        return None

    return PageRecord(
        page_id=sha1_text(url)[:16],
        url=url,
        title=title,
        text=text,
        headings=headings[:20],
        crawl_timestamp=now_utc_iso(),
    )


def extract_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    links: List[str] = []
    for anchor in soup.find_all("a", href=True):
        joined = normalize_url(urljoin(base_url, anchor["href"]))
        if is_allowed_url(joined):
            links.append(joined)
    return links


def chunk_text(text: str, chunk_size_words: int, chunk_overlap_words: int) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size_words - chunk_overlap_words)

    while start < len(words):
        end = min(len(words), start + chunk_size_words)
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start += step

    return chunks


def save_jsonl(path: str, rows: Iterable[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


# ===========
# Cell 5: Crawl site
# ===========
def crawl_site(seed_urls: List[str], max_pages: int, max_depth: int) -> List[PageRecord]:
    queue: deque[Tuple[str, int]] = deque((normalize_url(url), 0) for url in seed_urls)
    visited: Set[str] = set()
    collected_pages: List[PageRecord] = []

    with tqdm(total=max_pages, desc="Crawling pages") as progress:
        while queue and len(collected_pages) < max_pages:
            current_url, depth = queue.popleft()
            if current_url in visited:
                continue
            if not is_allowed_url(current_url):
                continue
            if not can_fetch(current_url):
                continue

            visited.add(current_url)
            html = fetch_html(current_url)
            if html is None:
                continue

            page = extract_page(html, current_url)
            if page is not None:
                collected_pages.append(page)
                progress.update(1)

            if depth < max_depth:
                for link in extract_links(html, current_url):
                    if link not in visited:
                        queue.append((link, depth + 1))

            time.sleep(REQUEST_DELAY_SECONDS)

    return collected_pages


pages = crawl_site(SEED_URLS, MAX_PAGES, MAX_DEPTH)
print(f"Collected {len(pages)} pages.")


# ===========
# Cell 6: Build chunks
# ===========
chunk_rows: List[Dict] = []

for page in pages:
    page_chunks = chunk_text(
        text=page.text,
        chunk_size_words=CHUNK_SIZE_WORDS,
        chunk_overlap_words=CHUNK_OVERLAP_WORDS,
    )

    for chunk_index, chunk in enumerate(page_chunks):
        chunk_rows.append(
            {
                "chunk_id": f"{page.page_id}-chunk-{chunk_index:04d}",
                "page_id": page.page_id,
                "url": page.url,
                "title": page.title,
                "section": page.headings[0] if page.headings else "main",
                "text": chunk,
                "word_count": len(chunk.split()),
                "crawl_timestamp": page.crawl_timestamp,
            }
        )

print(f"Built {len(chunk_rows)} chunks.")
if chunk_rows:
    print("Sample chunk:")
    print(chunk_rows[0]["text"][:500], "...")


# ===========
# Cell 7: Save outputs to Google Drive
# ===========
page_rows = [
    {
        "page_id": page.page_id,
        "url": page.url,
        "title": page.title,
        "text": page.text,
        "headings": page.headings,
        "crawl_timestamp": page.crawl_timestamp,
    }
    for page in pages
]

save_jsonl(RAW_PAGES_PATH, page_rows)
save_jsonl(CHUNKS_PATH, chunk_rows)
save_json(
    CRAWL_MANIFEST_PATH,
    {
        "created_at": now_utc_iso(),
        "base_dir": BASE_DIR,
        "seed_urls": SEED_URLS,
        "allowed_domains": sorted(ALLOWED_DOMAINS),
        "page_count": len(page_rows),
        "chunk_count": len(chunk_rows),
        "raw_pages_path": RAW_PAGES_PATH,
        "chunks_path": CHUNKS_PATH,
        "max_pages": MAX_PAGES,
        "max_depth": MAX_DEPTH,
        "chunk_size_words": CHUNK_SIZE_WORDS,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
    },
)

print("Saved raw pages to:", RAW_PAGES_PATH)
print("Saved chunks to:", CHUNKS_PATH)
print("Saved crawl manifest to:", CRAWL_MANIFEST_PATH)
