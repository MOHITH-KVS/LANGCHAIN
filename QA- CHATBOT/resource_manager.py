# =========================
# IMPORTS
# =========================

import os
import pickle

from langchain_community.vectorstores import FAISS

from embeddings import embedding_model

from engine.bm25_retriever import create_bm25_index


# =========================
# GLOBAL RESOURCES
# =========================

VECTOR_STORE = None

ALL_CHUNKS = None

BM25_INDEX = None


# =========================
# LOAD RESOURCES
# =========================

def initialize_resources():

    global VECTOR_STORE
    global ALL_CHUNKS
    global BM25_INDEX


    # =========================
    # LOAD CHUNKS
    # =========================

    if os.path.exists("chunks.pkl"):

        with open("chunks.pkl", "rb") as f:

            ALL_CHUNKS = pickle.load(f)

    else:

        raise Exception(

            "chunks.pkl not found"
        )


    # =========================
    # LOAD FAISS
    # =========================

    if os.path.exists("faiss_index"):

        VECTOR_STORE = FAISS.load_local(

            "faiss_index",

            embedding_model,

            allow_dangerous_deserialization=True
        )

    else:

        raise Exception(

            "faiss_index not found"
        )


    # =========================
    # CREATE BM25 INDEX
    # =========================

    BM25_INDEX = create_bm25_index(

        ALL_CHUNKS
    )


    print("\nRESOURCES INITIALIZED SUCCESSFULLY")


# =========================
# GETTERS
# =========================

def get_vector_store():

    return VECTOR_STORE


def get_all_chunks():

    return ALL_CHUNKS


def get_bm25_index():

    return BM25_INDEX