# GOAL OF THIS FILE

# This file should ONLY:

#| Responsibility              | Why                              |
#| --------------------------- | -------------------------------- |
#| Load document registry      | document-level retrieval         |
#| Create document embeddings  | hierarchical retrieval           |
#| Match query to documents    | semantic document routing        |
#| Return relevant documents   | scalable enterprise RAG          |


import os
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# LOAD EMBEDDING MODEL
# =========================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# LOAD DOCUMENT REGISTRY
# =========================

if os.path.exists("document_registry.pkl"):

    with open("document_registry.pkl", "rb") as f:

        document_registry = pickle.load(f)

else:

    document_registry = {}


# =========================
# DOCUMENT PROFILES
# =========================

document_profiles = []


for document_name, profile in document_registry.items():

    summary = profile.get(
        "summary",
        ""
    )

    keywords = profile.get(
        "keywords",
        []
    )

    sample_chunks = profile.get(
        "sample_chunks",
        []
    )

    # =========================
    # BUILD DOCUMENT PROFILE
    # =========================

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
# CREATE DOCUMENT EMBEDDINGS
# =========================

if len(document_profiles) > 0:

    profile_texts = [

        item["profile_text"]

        for item in document_profiles
    ]


    document_embeddings = embedding_model.encode(

        profile_texts,

        convert_to_numpy=True
    )

else:

    document_embeddings = np.array([])


# =========================
# TOKENIZE TEXT
# =========================

import re

def tokenize(text):
    if text is None:
        return set()

    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())

    return set(text.split())

# =========================
# KEYWORD MATCH SCORE
# =========================

def keyword_match_score(
    query,
    document_keywords,
    profile_text=""
):
    query_tokens = tokenize(query)

    keyword_tokens = tokenize(
        " ".join(document_keywords)
    )

    profile_tokens = tokenize(
        profile_text
    )

    all_tokens = keyword_tokens.union(
        profile_tokens
    )

    if len(query_tokens) == 0:
        return 0.0

    overlap = query_tokens.intersection(
        all_tokens
    )

    return len(overlap) / len(query_tokens)


# =========================
# ROUTE DOCUMENTS
# =========================

def route_documents(

    query,

    top_k=10
):


    # =========================
    # NO DOCUMENTS AVAILABLE
    # =========================

    if len(document_profiles) == 0:

        return []


    # =========================
    # QUERY EMBEDDING
    # =========================

    query_embedding = embedding_model.encode(

        [query],

        convert_to_numpy=True
    )


    # =========================
    # SEMANTIC SIMILARITY
    # =========================

    semantic_scores = cosine_similarity(

        query_embedding,

        document_embeddings
    )[0]


    matched_documents = []


    # =========================
    # HYBRID SCORING
    # =========================

    for idx, profile in enumerate(document_profiles):

        semantic_score = float(

            semantic_scores[idx]
        )


        keyword_score = keyword_match_score(
            query,
            profile["keywords"],
            profile["profile_text"]
        )

        print("\nDEBUG ROUTER")
        print("QUERY:", query)
        print("DOCUMENT:", profile["source"])
        print("KEYWORD SCORE:", keyword_score)


        # =========================
        # FINAL HYBRID SCORE
        # =========================

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


    # =========================
    # SORT DOCUMENTS
    # =========================

    matched_documents = sorted(

        matched_documents,

        key=lambda x: x["score"],

        reverse=True
    )

    print("\nAFTER SORTING")
    print("TOTAL DOCUMENTS:", len(matched_documents))

    for doc in matched_documents:
        print(doc["source"], doc["score"])


    # =========================
    # FILTER LOW QUALITY MATCHES
    # =========================

    BEST_SCORE_THRESHOLD = 0.25


    matched_documents = [

        doc

        for doc in matched_documents

        if doc["score"] >= BEST_SCORE_THRESHOLD
    ]


    # =========================
    # KEEP ONLY BEST DOCUMENT
    # =========================

    matched_documents = matched_documents[:top_k]
    print("\nALL DOCUMENT SCORES\n")

    for doc in matched_documents:
        print(doc)


    # =========================
    # DEBUG LOGS
    # =========================

    print("\nDOCUMENT ROUTING RESULTS:\n")


    for doc in matched_documents:

        print(f"SOURCE: {doc['source']}")

        print(f"SEMANTIC SCORE: {doc['semantic_score']}")

        print(f"KEYWORD SCORE: {doc['keyword_score']}")

        print(f"FINAL HYBRID SCORE: {doc['score']}")

        print("=" * 50)


    return matched_documents[:top_k]