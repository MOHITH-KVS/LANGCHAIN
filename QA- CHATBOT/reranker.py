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


    # =========================
    # EMPTY CHECK
    # =========================

    if len(retrieved_chunks) == 0:

        return []


    # =========================
    # PREPARE INPUTS
    # =========================

    pairs = []

    documents = []


    for item in retrieved_chunks:


        # =========================
        # HANDLE TUPLE FORMAT
        # =========================

        if isinstance(item, tuple):

            chunk = item[0]

        else:

            chunk = item


        pairs.append(

            [query, chunk.page_content]
        )

        documents.append(chunk)


    # =========================
    # CROSS-ENCODER SCORING
    # =========================

    semantic_scores = reranker_model.predict(pairs)


    reranked_results = []


    # =========================
    # HYBRID SCORING
    # =========================

    for doc, semantic_score in zip(

        documents,

        semantic_scores
    ):


        lexical_score = lexical_overlap_score(

            query,

            doc.page_content
        )


        # =========================
        # NORMALIZED HYBRID SCORE
        # =========================

        normalized_semantic = float(semantic_score)

        normalized_lexical = lexical_score / 10


        final_score = (

            normalized_semantic * 0.85

            +

            normalized_lexical * 0.15
        )


        reranked_results.append(

            (

                doc,

                final_score
            )
        )


    # =========================
    # FINAL SORTING
    # =========================

    reranked_results = sorted(

        reranked_results,

        key=lambda x: x[1],

        reverse=True
    )


    return reranked_results