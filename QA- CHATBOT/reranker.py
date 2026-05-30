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

    if not retrieved_chunks:
        return []

    docs = []

    for item in retrieved_chunks:
        if isinstance(item, tuple):
            doc = item[0]
        else:
            doc = item
        docs.append(doc)

    pairs = [
        [query, doc.page_content]
        for doc in docs
    ]

    scores = reranker_model.predict(pairs)

    scored_results = list(zip(docs, scores))

    scored_results = sorted(
        scored_results,
        key=lambda x: x[1],
        reverse=True
    )

    print("\nRERANKER SCORES:")
    for doc, score in scored_results[:5]:
        print(round(float(score), 4), "|", doc.metadata.get("section", ""))

    return scored_results