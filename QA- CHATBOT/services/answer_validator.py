# =============================================================
# services/answer_validator.py  —  PHASE 1 IMPROVED VERSION
# =============================================================
#
# WHAT CHANGED FROM YOUR ORIGINAL:
#
#   Your original version just checked if chunks list was empty or not.
#   It returned True (valid) even if context was very weak/irrelevant.
#
#   This version adds a HALLUCINATION GUARD:
#   - If no chunks found → invalid → chatbot says "I don't have info"
#   - If chunks found but all have very low similarity → invalid → same message
#   - Only if chunks are reasonably relevant → valid → LLM generates answer
#
# WHY THIS MATTERS:
#   Without this, your LLM can "hallucinate" — confidently make up answers
#   when the retrieved context is irrelevant to the question.
#
# =============================================================


# Minimum average similarity score to consider context "good enough".
# Scores typically range from 0.0 to 1.0 in cosine similarity.
# 0.20 is a conservative threshold — adjust based on your testing.
MINIMUM_SIMILARITY_THRESHOLD = 0.20


def validate_context(chunks):
    """
    Returns True if the retrieved chunks are good enough to answer from.
    Returns False if we should tell the user we don't have enough info.

    Parameters:
        chunks: list of LangChain Document objects (what your retriever returns)
    """

    # Case 1: No chunks at all
    if not chunks:
        return False

    # Case 2: Check if any chunk has a meaningful similarity score
    # LangChain stores similarity scores in metadata when using FAISS
    # If no scores available, we trust the chunks and return True
    scores = []
    for chunk in chunks:
    # chunk can be either a Document or a (Document, score) tuple
        if isinstance(chunk, tuple):
            doc, score_val = chunk
            scores.append(float(score_val))
        else:
            score = chunk.metadata.get("score", None)
            if score is not None:
                scores.append(float(score))

    # If we have scores, check they are above threshold
    if scores:
        avg_score = sum(scores) / len(scores)
        if avg_score < MINIMUM_SIMILARITY_THRESHOLD:
            return False   # context too weak, don't let LLM hallucinate

    return True


def get_fallback_message():
    """
    The message to show the user when we don't have enough context.
    This is shown instead of letting the LLM guess.
    """
    return (
        "I don't have enough information in the uploaded documents to answer this question. "
        "Please make sure the relevant document is uploaded, or try rephrasing your question."
    )