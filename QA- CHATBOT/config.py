# =========================
# GOAL OF THIS FILE
# =========================

# Centralized project configuration
# Single source of truth
# Production-ready settings


# =========================
# RETRIEVAL SETTINGS
# =========================

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

DEBUG = True


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