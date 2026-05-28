import re

from rank_bm25 import BM25Okapi


# =========================================================
# ADVANCED TOKENIZER
# =========================================================
# Industrial-grade preprocessing for:
# - resumes
# - certifications
# - skills
# - technical terms
# - entities
# - abbreviations
#
# Handles:
# - punctuation
# - symbols
# - extra spaces
# - case normalization
#
# Example:
# "AWS," -> "aws"
# "Certification:" -> "certification"
# "Machine-Learning" -> "machine learning"
# =========================================================

def tokenize(text):

    # =========================
    # SAFETY CHECK
    # =========================

    if not text:

        return []

    # =========================
    # LOWERCASE
    # =========================

    text = text.lower()

    # =========================
    # REPLACE SPECIAL SYMBOLS
    # =========================

    text = re.sub(

        r"[^a-z0-9\s]",

        " ",

        text
    )

    # =========================
    # REMOVE EXTRA SPACES
    # =========================

    text = re.sub(

        r"\s+",

        " ",

        text
    ).strip()

    # =========================
    # TOKENIZE
    # =========================

    tokens = text.split()

    return tokens


# =========================================================
# CREATE BM25 INDEX
# =========================================================
# Builds lexical retrieval index
# used alongside semantic retrieval
# for hybrid search.
# =========================================================

def create_bm25_index(chunks):

    tokenized_chunks = []

    for chunk in chunks:

        content = chunk.get(

            "content",

            ""
        )

        tokens = tokenize(content)

        tokenized_chunks.append(tokens)

    bm25 = BM25Okapi(tokenized_chunks)

    return bm25


# =========================================================
# BM25 SEARCH
# =========================================================
# Performs lexical retrieval
# using exact keyword matching.
#
# Extremely important for:
# - certifications
# - names
# - skills
# - technologies
# - abbreviations
# - resume queries
# =========================================================

def bm25_search(

    bm25,

    chunks,

    query,

    top_k=5
):

    # =========================
    # TOKENIZE QUERY
    # =========================

    tokenized_query = tokenize(query)

    # =========================
    # EMPTY QUERY SAFETY
    # =========================

    if not tokenized_query:

        return []

    # =========================
    # GET BM25 SCORES
    # =========================

    scores = bm25.get_scores(

        tokenized_query
    )

    # =========================
    # ATTACH SCORES
    # =========================

    scored_chunks = list(

        zip(chunks, scores)
    )

    # =========================
    # REMOVE ZERO-SCORE RESULTS
    # =====================================================
    # Critical industrial fix:
    # prevents irrelevant chunks
    # from polluting retrieval.
    # =====================================================

    scored_chunks = [

        (chunk, score)

        for chunk, score in scored_chunks

        if score > 0
    ]

    # =========================
    # SORT BY SCORE
    # =========================

    scored_chunks = sorted(

        scored_chunks,

        key=lambda x: x[1],

        reverse=True
    )

    # =========================
    # RETURN TOP RESULTS
    # =========================

    return scored_chunks[:top_k]