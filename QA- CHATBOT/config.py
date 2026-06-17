# =========================
# GOAL OF THIS FILE
# =========================

# Centralized project configuration
# Single source of truth
# Production-ready settings


# =========================
# RETRIEVAL SETTINGS
# =========================
#
# CHANGED: TOP_K_RETRIEVAL lowered from implicit 150 (hardcoded in
# vector_store.py's similarity_search_with_score call) — see note below.
# This value here controls the FINAL number of chunks returned after
# hybrid fusion, which was already fine at 15.

TOP_K_RETRIEVAL = 15

TOP_K_RERANK = 10

SIMILARITY_THRESHOLD = 0.10


# =========================
# CONTEXT SETTINGS
# =========================

MAX_CONTEXT_CHUNKS = 10

MAX_CONTEXT_TOKENS = 2500


# =========================
# LLM SETTINGS
# =========================

TEMPERATURE = 0


# =========================
# DEBUG SETTINGS
# =========================
#
# CHANGED: DEBUG = False (was True)
#
# WHY THIS MATTERS FOR MEMORY:
# With DEBUG=True, every single /chat request prints EVERY chunk's
# full content, every semantic result, every BM25 result, and builds
# multiple large temporary lists just for printing. On Render's free
# 512MB instance, this print-heavy debug path during each request was
# a major contributor to memory spikes that crashed the server.
#
# Keep DEBUG=True only when running locally for development/learning.
# Always set DEBUG=False before deploying to production.

DEBUG = False


# =========================
# CHUNKING SETTINGS
# =========================

MAX_CHUNK_LENGTH = 700

MIN_CHUNK_LENGTH = 30


# =========================
# RERANKER SETTINGS
# =========================

RERANKER_MODEL = "BAAI/bge-reranker-base"

LEXICAL_BOOST_WEIGHT = 0.30


# =========================
# GENERATION SETTINGS
# =========================

GENERATION_MODEL = "llama-3.1-8b-instant"
