# =========================
# GOAL OF THIS FILE
# =========================

# This file should ONLY:
#
# | Responsibility              | Why                         |
# | --------------------------- | --------------------------- |
# | Score retrieved chunks      | relevance estimation        |
# | Rerank chunks               | improve retrieval precision |
# | Select best chunks          | stronger context quality    |
# | Improve retrieval relevance | reduce noisy retrieval      |


# =========================
# IMPORTS
# =========================

from sentence_transformers import CrossEncoder

import re


# =========================
# LOAD CROSS-ENCODER MODEL
# =========================

reranker_model = CrossEncoder(

    "BAAI/bge-reranker-base"
)


# =========================
# TEXT NORMALIZATION
# =========================

def normalize_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================
# TOKEN EXTRACTION
# =========================

def extract_keywords(text):

    normalized = normalize_text(text)

    return set(normalized.split())


# =========================
# LEXICAL OVERLAP SCORE
# =========================

def lexical_overlap_score(

    query,

    content
):

    query_words = extract_keywords(query)

    content_words = extract_keywords(content)


    if len(query_words) == 0:

        return 0


    overlap = query_words.intersection(content_words)


    return len(overlap)


# =========================
# RERANK CHUNKS
# =========================

def rerank_chunks(
    query,
    retrieved_chunks
):

    results = []

    for item in retrieved_chunks:

        if isinstance(item, tuple):
            doc = item[0]
            score = item[1]
        else:
            doc = item
            score = 0

        results.append((doc, score))

    return results