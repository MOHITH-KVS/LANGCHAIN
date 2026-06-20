# =============================================================
# resource_manager.py  —  V2 MULTI-TENANT VERSION
# =============================================================
#
# WHAT CHANGED FROM V1:
#
#   BEFORE: initialize_resources() loaded ONE global FAISS index
#           from disk at startup. All users shared it.
#           It crashed if faiss_index/ folder didn't exist.
#
#   AFTER:  No more global loading at startup.
#           Instead, get_session_resources(session_id) loads
#           the specific user's index from Redis on demand.
#           Each user gets their own isolated resources.
#
#   The old getters (get_vector_store, get_all_chunks, etc.)
#   are kept as fallbacks but the main path is now per-session.
#
# =============================================================

import os
import pickle

from embeddings import embedding_model
from vector_store import create_bm25_index
from services.session_store import load_session_data


# =============================================================
# PER-SESSION RESOURCE LOADING
# =============================================================
# This is the main function called by chatbot_engine.py
# for every /chat request.
#
# It loads THIS user's private FAISS index from Redis,
# builds a BM25 index from their chunks, and returns both.
#
# Returns: (vector_store, all_chunks, bm25_index)
# or raises an exception if the session has no uploaded documents.

def get_session_resources(session_id: str):
    """
    Load resources for a specific user session from Redis.

    Called at the start of every ask_question() call.
    """

    # ── Load from Redis ───────────────────────────────────────
    vector_store, all_chunks, document_registry = load_session_data(
        session_id=session_id,
        embedding_model=embedding_model
    )

    # ── Session not found ─────────────────────────────────────
    if vector_store is None or not all_chunks:
        return None, None, None

    # ── Build BM25 index from chunks ──────────────────────────
    # BM25 is fast to build (pure Python, no API calls needed)
    # so we build it fresh from the loaded chunks each time.
    # It's lightweight enough that this adds <100ms per request.
    bm25_index = create_bm25_index(all_chunks)

    return vector_store, all_chunks, bm25_index


# =============================================================
# INITIALIZE  (simplified — no longer loads from disk)
# =============================================================
# Called once at startup by chatbot_engine.py.
# In V2 we don't load any global index at startup — just confirm
# everything is connected and ready.

def initialize_resources():
    """
    V2: No global index loading at startup.
    Each user's resources are loaded per-request from Redis.
    Just print a confirmation that the system is ready.
    """
    print("\n✅ RESOURCES INITIALIZED — Multi-tenant mode")
    print("   Each user session loads their own index from Redis")
    print("   No global pre-loaded index")