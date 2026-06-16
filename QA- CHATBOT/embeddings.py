# =============================================================
# embeddings.py  —  MEMORY-FIX VERSION (uses HF Inference API)
# =============================================================
#
# WHY THIS CHANGED:
#
#   BEFORE: HuggingFaceEmbeddings loaded the full BGE model
#           (~400MB+) into RAM on every startup.
#           This is what crashed Render's free 512MB instance.
#
#   AFTER:  We call Hugging Face's free Inference API instead.
#           The model runs on HF's servers, not yours.
#           Your app only sends text and receives back vectors.
#           Near-zero RAM usage for embeddings.
#
# NEW PACKAGE NEEDED:
#   pip install huggingface_hub
#
# NEW ENV VARIABLE NEEDED:
#   HF_TOKEN=hf_xxxxxxxxxxxx   (get free from huggingface.co/settings/tokens)
#
# =============================================================

import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# NOTE: api-inference.huggingface.co is DEPRECATED (shut down).
# Hugging Face moved everything to router.huggingface.co
API_URL = f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL_NAME}/pipeline/feature-extraction"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


def create_embedding(text):
    """
    Sends text to Hugging Face's hosted model and gets back
    the embedding vector. Same output format as before —
    nothing else in your code needs to change.
    """
    response = requests.post(
        API_URL,
        headers=HEADERS,
        json={"inputs": text, "options": {"wait_for_model": True}},
        timeout=30
    )

    if response.status_code != 200:
        raise Exception(f"HF Embedding API error: {response.status_code} - {response.text}")

    embedding = response.json()

    # Some HF models return a nested list (token-level embeddings).
    # If so, we average them into a single sentence-level vector.
    if isinstance(embedding[0], list):
        embedding = [sum(col) / len(col) for col in zip(*embedding)]

    return embedding


# =============================================================
# BATCH VERSION — embeds multiple texts in ONE network call
# =============================================================
# Used by vector_store.py when indexing many chunks at once.
# Much faster than calling create_embedding() in a loop.

def create_embeddings_batch(texts):
    response = requests.post(
        API_URL,
        headers=HEADERS,
        json={"inputs": texts, "options": {"wait_for_model": True}},
        timeout=60
    )

    if response.status_code != 200:
        raise Exception(f"HF Embedding API error: {response.status_code} - {response.text}")

    embeddings = response.json()

    # Average token-level embeddings per text, if needed
    result = []
    for emb in embeddings:
        if isinstance(emb[0], list):
            emb = [sum(col) / len(col) for col in zip(*emb)]
        result.append(emb)

    return result


# =============================================================
# LANGCHAIN-COMPATIBLE WRAPPER
# =============================================================
# Your vector_store.py passes embedding_model directly into FAISS
# (embedding=embedding_model). FAISS/LangChain expects an object
# that inherits from langchain_core.embeddings.Embeddings — a plain
# class with the right method names is NOT enough, it must inherit
# from this base class or FAISS internals fail with
# "object is not callable".

from langchain_core.embeddings import Embeddings


class HFAPIEmbeddings(Embeddings):
    def embed_query(self, text):
        return create_embedding(text)

    def embed_documents(self, texts):
        return create_embeddings_batch(texts)


embedding_model = HFAPIEmbeddings()


if __name__ == "__main__":
    sample_text = "Artificial Intelligence improves education"
    embedding = create_embedding(sample_text)
    print("VECTOR LENGTH:")
    print(len(embedding))
    print("\nFIRST 10 VALUES:")
    print(embedding[:10])
