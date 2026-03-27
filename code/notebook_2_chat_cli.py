"""Notebook 2: load the vector DB, retrieve context, and run a CLI-style chat."""

# ===========
# Cell 1: Install dependencies
# ===========
import subprocess
import sys


def pip_install(*packages: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


INSTALL_PACKAGES = False  # Set to True on first Colab GPU run.

if INSTALL_PACKAGES:
    pip_install(
        "chromadb",
        "sentence-transformers",
        "transformers",
        "accelerate",
        "bitsandbytes",
    )


# ===========
# Cell 2: Imports and Google Drive mount
# ===========
import json
import os
from typing import Dict, List

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
VECTOR_DB_MANIFEST_PATH = os.path.join(METADATA_DIR, "vector_db_manifest.json")

LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TOP_K = 4
MAX_INPUT_CHARS = 6000
MAX_NEW_TOKENS = 350
TEMPERATURE = 0.2
USE_4BIT = True


# ===========
# Cell 4: Helpers
# ===========
def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dedupe_keep_order(values: List[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def get_model_device(model: AutoModelForCausalLM) -> torch.device:
    return next(model.parameters()).device


manifest = load_json(VECTOR_DB_MANIFEST_PATH)
COLLECTION_NAME = manifest["collection_name"]
EMBEDDING_MODEL_NAME = manifest["embedding_model_name"]


# ===========
# Cell 5: Load embedding model and Chroma collection
# ===========
embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
print("Query embedding device:", embedding_device)

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=embedding_device)

client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
collection = client.get_collection(COLLECTION_NAME)
print("Loaded collection:", COLLECTION_NAME)
print("Stored chunk count:", collection.count())


# ===========
# Cell 6: Load the LLM
# ===========
if not torch.cuda.is_available():
    print("Warning: no GPU detected. A 7B model may be slow or may not fit in memory.")

bnb_dtype = torch.bfloat16
if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
    bnb_dtype = torch.float16

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

model_kwargs = {"device_map": "auto"}
if USE_4BIT and torch.cuda.is_available():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=bnb_dtype,
    )
    model_kwargs["quantization_config"] = quant_config
    model_kwargs["torch_dtype"] = bnb_dtype
else:
    model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, **model_kwargs)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loaded LLM:", LLM_MODEL_NAME)


# ===========
# Cell 7: Retrieval and generation functions
# ===========
SYSTEM_PROMPT = """You answer questions about the University of Salford.
Use only the retrieved context.
If the context does not contain the answer, say that you do not know based on the retrieved sources.
Keep the answer concise and factual.
Always include a final Sources section using the provided URLs only."""


def retrieve_context(question: str, top_k: int = TOP_K) -> Dict:
    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]

    return {
        "documents": documents,
        "metadatas": metadatas,
        "distances": distances,
    }


def build_context_block(retrieval_result: Dict) -> str:
    parts: List[str] = []

    for index, (document, metadata) in enumerate(
        zip(retrieval_result["documents"], retrieval_result["metadatas"]),
        start=1,
    ):
        parts.append(
            "\n".join(
                [
                    f"Source {index}",
                    f"Title: {metadata.get('title', '')}",
                    f"Section: {metadata.get('section', '')}",
                    f"URL: {metadata.get('url', '')}",
                    "Content:",
                    document,
                ]
            )
        )

    joined = "\n\n".join(parts)
    return joined[:MAX_INPUT_CHARS]


def build_messages(question: str, retrieval_result: Dict) -> List[Dict[str, str]]:
    context_block = build_context_block(retrieval_result)
    user_prompt = f"""Question:
{question}

Retrieved context:
{context_block}
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def generate_answer(question: str, retrieval_result: Dict) -> str:
    messages = build_messages(question, retrieval_result)

    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = SYSTEM_PROMPT + "\n\n" + messages[-1]["content"] + "\n\nAnswer:"

    model_device = get_model_device(llm)
    model_inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
    )
    model_inputs = {key: value.to(model_device) for key, value in model_inputs.items()}

    with torch.inference_mode():
        generated = llm.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated[0][model_inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def answer_question(question: str, top_k: int = TOP_K) -> Dict:
    retrieval_result = retrieve_context(question, top_k=top_k)
    answer = generate_answer(question, retrieval_result)
    sources = dedupe_keep_order(
        [metadata.get("url", "") for metadata in retrieval_result["metadatas"]]
    )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieval_result": retrieval_result,
    }


# ===========
# Cell 8: Single-question test
# ===========
TEST_QUESTION = "What postgraduate study options are available?"
test_result = answer_question(TEST_QUESTION, top_k=TOP_K)

print("Question:", test_result["question"])
print()
print(test_result["answer"])
print()
print("Sources:")
for source_url in test_result["sources"]:
    print("-", source_url)


# ===========
# Cell 9: CLI-style chat loop
# ===========
RUN_CHAT_LOOP = False  # Set to True when you want an interactive loop.

if RUN_CHAT_LOOP:
    print("Type 'exit' to stop the chat.")
    while True:
        user_question = input("\nYou: ").strip()
        if not user_question:
            continue
        if user_question.lower() in {"exit", "quit"}:
            break

        result = answer_question(user_question, top_k=TOP_K)
        print("\nAssistant:")
        print(result["answer"])
        print("\nSources:")
        for source_url in result["sources"]:
            print("-", source_url)
