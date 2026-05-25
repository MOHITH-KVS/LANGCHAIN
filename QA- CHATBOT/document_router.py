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

from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np


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
# DOCUMENT NAMES
# =========================

document_names = list(document_registry.keys())


# =========================
# DOCUMENT TEXTS
# =========================

document_texts = list(document_registry.values())


# =========================
# CREATE DOCUMENT EMBEDDINGS
# =========================

if len(document_texts) > 0:

    document_embeddings = embedding_model.encode(

        document_texts
    )

else:

    document_embeddings = np.array([])


# =========================
# ROUTE DOCUMENTS
# =========================

def route_documents(

    query,

    top_k=1
):


    # =========================
    # NO DOCUMENTS AVAILABLE
    # =========================

    if len(document_names) == 0:

        return []


    # =========================
    # QUERY EMBEDDING
    # =========================

    query_embedding = embedding_model.encode([query])


    # =========================
    # SIMILARITY SEARCH
    # =========================

    similarities = cosine_similarity(

        query_embedding,

        document_embeddings
    )[0]


    # =========================
    # SORT DOCUMENTS
    # =========================

    ranked_indices = np.argsort(similarities)[::-1]


    matched_documents = []


    # =========================
    # TOP DOCUMENTS
    # =========================

    for idx in ranked_indices[:top_k]:

        matched_documents.append(

            {

                "source": document_names[idx],

                "score": float(similarities[idx])
            }
        )


    return matched_documents