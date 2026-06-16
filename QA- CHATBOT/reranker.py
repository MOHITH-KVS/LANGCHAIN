# =============================================================
# reranker.py  —  MEMORY-FIX VERSION (uses HF Inference API)
# =============================================================
#
# WHY THIS CHANGED:
#
#   BEFORE: CrossEncoder(RERANKER_MODEL) loaded the full
#           BGE reranker model (~280MB+) into RAM.
#           Combined with the embedding model, this exceeded
#           Render's free 512MB limit.
#
#   AFTER:  We call Hugging Face's Inference API for reranking.
#           The model runs on HF's servers.
#           All your existing logic (lexical boost, sorting,
#           debug printing) stays EXACTLY the same — only the
#           scoring call itself changed.
#
# NOTHING ELSE IN YOUR PROJECT NEEDS TO CHANGE.
# chatbot_engine.py calls rerank_chunks() exactly the same way.
#
# =============================================================

import os
import re
import requests
from dotenv import load_dotenv

from config import (
    RERANKER_MODEL,
    LEXICAL_BOOST_WEIGHT
)
from config import DEBUG

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# NOTE: api-inference.huggingface.co is DEPRECATED (shut down).
# Hugging Face moved everything to router.huggingface.co
API_URL = f"https://router.huggingface.co/hf-inference/models/{RERANKER_MODEL}/pipeline/sentence-similarity"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


# =========================
# TEXT NORMALIZATION  (unchanged from your original)
# =========================

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# TOKEN EXTRACTION  (unchanged from your original)
# =========================

def extract_keywords(text):
    normalized = normalize_text(text)
    return set(normalized.split())


# =========================
# LEXICAL OVERLAP SCORE  (unchanged from your original)
# =========================

def lexical_overlap_score(query, content):
    query_words = extract_keywords(query)
    content_words = extract_keywords(content)
    if len(query_words) == 0:
        return 0
    overlap = query_words.intersection(content_words)
    return len(overlap)


# =============================================================
# CROSS-ENCODER SCORING VIA HF API  (replaces local model)
# =============================================================
# HF's cross-encoder reranker API expects this format:
#   {"inputs": {"source_sentence": query, "sentences": [doc1, doc2, ...]}}
# Returns a list of similarity scores, one per sentence,
# in the SAME ORDER they were sent.

def get_rerank_scores_from_api(query, documents_text):
    response = requests.post(
        API_URL,
        headers=HEADERS,
        json={
            "inputs": {
                "source_sentence": query,
                "sentences": documents_text
            },
            "options": {"wait_for_model": True}
        },
        timeout=30
    )

    if response.status_code != 200:
        raise Exception(f"HF Reranker API error: {response.status_code} - {response.text}")

    scores = response.json()
    return scores


# =========================
# RERANK CHUNKS  (same structure as your original, API-based scoring)
# =========================

def rerank_chunks(query, retrieved_chunks):
    if not retrieved_chunks:
        return []

    # =========================
    # EXTRACT DOCUMENTS
    # =========================

    docs = []
    for item in retrieved_chunks:
        if isinstance(item, tuple):
            doc = item[0]
        else:
            doc = item
        docs.append(doc)

    # =========================
    # PREPARE TEXTS FOR API CALL
    # =========================

    documents_text = [
        doc.page_content + " " + doc.metadata.get("section", "")
        for doc in docs
    ]

    # =========================
    # CROSS ENCODER SCORES (via API, ONE batched call)
    # =========================

    scores = get_rerank_scores_from_api(query, documents_text)

    # =========================
    # HYBRID RERANK SCORE  (unchanged logic)
    # =========================

    scored_results = []
    for doc, score in zip(docs, scores):
        lexical_score = lexical_overlap_score(query, doc.page_content)

        final_score = (
            float(score)
            +
            (LEXICAL_BOOST_WEIGHT * lexical_score)
        )

        scored_results.append((doc, final_score))

    # =========================
    # SORT RESULTS
    # =========================

    scored_results = sorted(
        scored_results,
        key=lambda x: x[1],
        reverse=True
    )

    # =========================
    # DEBUG OUTPUT  (unchanged from your original)
    # =========================

    if DEBUG:
        print("\n")
        print("=" * 80)
        print("RERANKER SCORES")
        print("=" * 80)
        for doc, score in scored_results[:10]:
            print(
                round(float(score), 4),
                "|",
                doc.metadata.get("section", "unknown")
            )
        print("=" * 80)

    return scored_results
