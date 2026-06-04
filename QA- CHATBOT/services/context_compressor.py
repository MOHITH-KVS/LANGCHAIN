# =========================
# CONTEXT COMPRESSION
# =========================

def compress_context(reranked_chunks):

    if not reranked_chunks:
        return []

    top_score = reranked_chunks[0][1]

    threshold = top_score * 0.30

    compressed_chunks = []

    for chunk, score in reranked_chunks:

        if score >= threshold:

            compressed_chunks.append(
                (chunk, score)
            )

    return compressed_chunks