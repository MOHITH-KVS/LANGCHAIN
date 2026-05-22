# GOAL OF THIS FILE

# This file should ONLY:

#| Responsibility              | Why                              |
#| --------------------------- | -------------------------------- |
#| Load document registry      | document-level retrieval         |
#| Create document embeddings  | hierarchical retrieval           |
#| Match query to documents    | semantic document routing        |
#| Return relevant documents   | scalable enterprise RAG          |


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

with open("document_registry.pkl", "rb") as f:

    document_registry = pickle.load(f)


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

document_embeddings = embedding_model.encode(

    document_texts
)


# =========================
# ROUTE DOCUMENTS
# =========================

def route_documents(

    query,

    top_k=1
):

    query_embedding = embedding_model.encode([query])


    similarities = cosine_similarity(

        query_embedding,

        document_embeddings
    )[0]


    ranked_indices = np.argsort(similarities)[::-1]


    matched_documents = []


    for idx in ranked_indices[:top_k]:

        matched_documents.append(

            {

                "source": document_names[idx],

                "score": similarities[idx]
            }
        )


    return matched_documents