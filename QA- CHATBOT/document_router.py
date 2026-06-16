# =============================================================
# document_router.py  —  MEMORY-FIX VERSION (uses HF Inference API)
# =============================================================
#
# WHY THIS CHANGED:
#
#   BEFORE: SentenceTransformer("all-MiniLM-L6-v2") loaded a
#           second local model into RAM, on top of the embedding
#           and reranker models. This was the THIRD model adding
#           to the memory crash on Render's free 512MB instance.
#
#   AFTER:  We call Hugging Face's Inference API for this model
#           too. Same model, same outputs — just running on HF's
#           servers instead of yours.
#
# EVERYTHING ELSE IN THIS FILE — routing logic, keyword scoring,
# hybrid scoring, sorting, filtering — is UNCHANGED.
#
# =============================================================

import os
import pickle
import re
import requests
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
ROUTER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# NOTE: api-inference.huggingface.co is DEPRECATED (shut down).
# Hugging Face moved everything to router.huggingface.co
API_URL = f"https://router.huggingface.co/hf-inference/models/{ROUTER_MODEL_NAME}/pipeline/feature-extraction"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


# =============================================================
# API-BASED ENCODE FUNCTION
# =============================================================
# Replaces embedding_model.encode(texts, convert_to_numpy=True)
# Same input/output shape: takes a list of texts, returns a numpy array.

def encode_texts(texts):
    response = requests.post(
        API_URL,
        headers=HEADERS,
        json={"inputs": texts, "options": {"wait_for_model": True}},
        timeout=60
    )

    if response.status_code != 200:
        raise Exception(f"HF Router Embedding API error: {response.status_code} - {response.text}")

    embeddings = response.json()

    # Average token-level embeddings into sentence-level vectors if needed
    result = []
    for emb in embeddings:
        if isinstance(emb[0], list):
            emb = [sum(col) / len(col) for col in zip(*emb)]
        result.append(emb)

    return np.array(result)


# =========================
# LOAD DOCUMENT REGISTRY  (unchanged)
# =========================

if os.path.exists("document_registry.pkl"):
    with open("document_registry.pkl", "rb") as f:
        document_registry = pickle.load(f)
else:
    document_registry = {}


# =========================
# DOCUMENT PROFILES  (unchanged)
# =========================

document_profiles = []

for document_name, profile in document_registry.items():
    summary = profile.get("summary", "")
    keywords = profile.get("keywords", [])
    sample_chunks = profile.get("sample_chunks", [])

    profile_text = " ".join([
        document_name,
        summary,
        " ".join(keywords),
        " ".join(sample_chunks)
    ])

    document_profiles.append({
        "source": document_name,
        "profile_text": profile_text,
        "keywords": keywords
    })


# =========================
# CREATE DOCUMENT EMBEDDINGS  (now via API, same logic)
# =========================

if len(document_profiles) > 0:
    profile_texts = [item["profile_text"] for item in document_profiles]
    document_embeddings = encode_texts(profile_texts)
else:
    document_embeddings = np.array([])


# =========================
# TOKENIZE TEXT  (unchanged)
# =========================

def tokenize(text):
    if text is None:
        return set()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    return set(text.split())


# =========================
# KEYWORD MATCH SCORE  (unchanged)
# =========================

def keyword_match_score(query, document_keywords, profile_text=""):
    query_tokens = tokenize(query)
    keyword_tokens = tokenize(" ".join(document_keywords))
    profile_tokens = tokenize(profile_text)
    all_tokens = keyword_tokens.union(profile_tokens)

    if len(query_tokens) == 0:
        return 0.0

    overlap = query_tokens.intersection(all_tokens)
    return len(overlap) / len(query_tokens)


# =========================
# ROUTE DOCUMENTS  (unchanged logic, API-based embedding call)
# =========================

def route_documents(query, top_k=10):

    if len(document_profiles) == 0:
        return []

    # QUERY EMBEDDING (now via API)
    query_embedding = encode_texts([query])

    # SEMANTIC SIMILARITY
    semantic_scores = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    matched_documents = []

    for idx, profile in enumerate(document_profiles):
        semantic_score = float(semantic_scores[idx])

        keyword_score = keyword_match_score(
            query,
            profile["keywords"],
            profile["profile_text"]
        )

        print("\nDEBUG ROUTER")
        print("QUERY:", query)
        print("DOCUMENT:", profile["source"])
        print("KEYWORD SCORE:", keyword_score)

        hybrid_score = (
            (0.7 * semantic_score)
            +
            (0.3 * keyword_score)
        )

        matched_documents.append({
            "source": profile["source"],
            "semantic_score": semantic_score,
            "keyword_score": keyword_score,
            "score": hybrid_score
        })

    matched_documents = sorted(
        matched_documents,
        key=lambda x: x["score"],
        reverse=True
    )

    print("\nAFTER SORTING")
    print("TOTAL DOCUMENTS:", len(matched_documents))
    for doc in matched_documents:
        print(doc["source"], doc["score"])

    BEST_SCORE_THRESHOLD = 0.25

    matched_documents = [
        doc for doc in matched_documents
        if doc["score"] >= BEST_SCORE_THRESHOLD
    ]

    matched_documents = matched_documents[:top_k]
    print("\nALL DOCUMENT SCORES\n")
    for doc in matched_documents:
        print(doc)

    print("\nDOCUMENT ROUTING RESULTS:\n")
    for doc in matched_documents:
        print(f"SOURCE: {doc['source']}")
        print(f"SEMANTIC SCORE: {doc['semantic_score']}")
        print(f"KEYWORD SCORE: {doc['keyword_score']}")
        print(f"FINAL HYBRID SCORE: {doc['score']}")
        print("=" * 50)

    return matched_documents[:top_k]
