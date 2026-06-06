# =========================
# CONTEXT COMPRESSION
# =========================

def compress_context(reranked_chunks):

    if not reranked_chunks:
        return []

    # Do not drop chunks based on relative score threshold.
    # The reranker scores vary wildly depending on query type.
    # Dropping by percentage causes incomplete answers when
    # content is split across many small chunks.
    # Instead return all chunks and let the token budget
    # in context packing control what fits.

    return reranked_chunks