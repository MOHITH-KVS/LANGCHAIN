#GOAL OF THIS FILE

#This file should ONLY:
#| Responsibility              | Why                         |
#| --------------------------- | --------------------------- |
#| Score retrieved chunks      | relevance estimation        |
#| Rerank chunks               | improve retrieval precision |
#| Select best chunks          | stronger context quality    |
#| Improve retrieval relevance | reduce noisy retrieval      |

from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_chunks(query, retrieved_chunks):

    pairs = []

    for chunk in retrieved_chunks:

        pairs.append(
            [query, chunk.page_content]
        )

    scores = reranker_model.predict(pairs)

    scored_chunks = list(
        zip(retrieved_chunks, scores)
    )

    ranked_chunks = sorted(
        scored_chunks,
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_chunks

