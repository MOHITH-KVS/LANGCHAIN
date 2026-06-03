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

from config import (
    RERANKER_MODEL,
    LEXICAL_BOOST_WEIGHT
)

from config import DEBUG

# =========================
# LOAD CROSS-ENCODER MODEL
# =========================

reranker_model = CrossEncoder(

    RERANKER_MODEL
)


# =========================
# TEXT NORMALIZATION
# =========================

def normalize_text(text):

    text = text.lower()

    text = re.sub(

        r"[^a-zA-Z0-9\.\s]",

        " ",

        text
    )
    text = re.sub(

        r"\s+",

        " ",

        text
    ).strip()

    return text


# =========================
# TOKEN EXTRACTION
# =========================

def extract_keywords(text):

    normalized = normalize_text(text)

    return set(

        normalized.split()
    )


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

    overlap = query_words.intersection(

        content_words
    )

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
    # CROSS ENCODER INPUT
    # =========================

    pairs = [

        [query, doc.page_content]

        for doc in docs
    ]


    # =========================
    # CROSS ENCODER SCORES
    # =========================

    scores = reranker_model.predict(

        pairs
    )


    # =========================
    # HYBRID RERANK SCORE
    # =========================

    scored_results = []

    for doc, score in zip(

        docs,

        scores
    ):

        lexical_score = lexical_overlap_score(

            query,

            doc.page_content
        )

        # ====================================
        # FINAL SCORE
        # ====================================
        # CrossEncoder
        # +
        # Lexical overlap boost
        # ====================================

        final_score = (

            float(score)

            +

            (LEXICAL_BOOST_WEIGHT * lexical_score)
        )

        scored_results.append(

            (

                doc,

                final_score
            )
        )


    # =========================
    # SORT RESULTS
    # =========================

    scored_results = sorted(

        scored_results,

        key=lambda x: x[1],

        reverse=True
    )


    # =========================
    # DEBUG OUTPUT
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
                doc.metadata.get(
                    "section",
                    "unknown"
                )
            )

        print("=" * 80)


    return scored_results